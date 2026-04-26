"""
Step 2a — Corpus Preparation & LLM Annotation

Tokenize a corpus, extract residual-stream activations at the target layer,
and annotate each token position with binary feature labels using Claude.

Outputs:
    pipeline_data/tokens.pt        (N, seq_len) int64
    pipeline_data/activations.pt   (N, seq_len, d_model) float32
    pipeline_data/annotations.pt   (N, seq_len, n_features) float32

Usage:
    python -m pipeline.annotate
"""

# vLLM requires 'spawn' multiprocessing to avoid CUDA re-init in forks.
# Must be set before any CUDA initialization happens.
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

from .config import Config

# Module-global accumulator for silent-failure auditing. annotate_sequence_async
# appends a record per (seq_idx, chunk_start, chunk_end, last_err) when a
# chunk's labels are left zero after max retries. The top-level async entry
# point clears it at start, checks it at end, and raises if the failure rate
# exceeds a configurable threshold.
_ANNOTATION_FAILURE_COUNT: list = []


_NOT_PREFIX_RE = re.compile(r"^not\s+", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _format_feature_for_annotator(f: dict, max_exclusions: int = 2) -> str:
    """Build the description string the annotator actually sees for a
    feature. When the catalog entry carries `exclusions` (set by
    --step rewrite-catalog or by promote-loop's atom decomposer), they
    are appended as ", NOT X, NOT Y" clauses so the boundary is explicit
    in the prompt itself.

    SCOPE NOTE (v8.16 audit fix #5): only the `exclusions` field is
    propagated into the annotator suffix. The richer v8.14 fields
    `positive_examples` and `negative_examples` are NOT in the suffix
    by design. Reasoning:
      - Adding 3-5 example phrases per feature would multiply suffix
        length by ~5-10x, blowing past the prefix-cache budget and
        slowing annotation by the same factor.
      - Examples in the prompt invite the annotator to memorize them
        instead of generalizing the rule, which we want to avoid.
      - positive_examples / negative_examples are kept for human audit
        (--step audit-feature reads them) and for the decomposer's
        own internal use; they're catalog metadata, not annotator
        instructions.
    The v8.15 changelog overstated this as "richer fields wired into
    annotator" — only `exclusions` actually made it into the prompt.

    Without this, v8.14's exclusion metadata is dead weight: the catalog
    knows the boundary but the annotator doesn't. Appending up to two
    exclusions costs ~10-25 extra suffix tokens per feature; the local
    Qwen3-4B-Base annotator runs at ~600 dec/s with prefix caching, so
    the throughput hit is roughly proportional. Limit to 2 exclusions
    so a verbose Sonnet rewrite doesn't blow up the suffix budget.

    The formatting normalizes Sonnet's output: strips a leading "NOT "
    if the exclusion already has one, drops trailing parentheticals
    (often "(because Y)" rationale that's meaningful to the human
    auditor but noise to the annotator).
    """
    desc = (f.get("description") or "").rstrip(".").strip()
    excl_raw = f.get("exclusions") or []
    if not excl_raw or not desc:
        return desc
    cleaned = []
    for e in excl_raw[:max_exclusions]:
        s = str(e).strip()
        s = _NOT_PREFIX_RE.sub("", s)
        s = _PARENTHETICAL_RE.sub("", s)
        s = s.rstrip(".").strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return desc
    return f"{desc}, NOT {', NOT '.join(cleaned)}"


# ── Corpus preparation ──────────────────────────────────────────────────────

def prepare_corpus(model, cfg: Config) -> torch.Tensor:
    """Load corpus, tokenize with transformer_lens, return (N, seq_len) token tensor."""
    from datasets import load_dataset

    tokenizer = model.tokenizer
    print(f"Loading corpus: {cfg.corpus_dataset} [{cfg.corpus_split}]")
    dataset = load_dataset(cfg.corpus_dataset, split=cfg.corpus_split, streaming=True)

    sequences = []
    pbar = tqdm(total=cfg.n_sequences, desc="Tokenizing")
    for example in dataset:
        text = example.get("text", "")
        if len(text.strip()) < 80:
            continue
        ids = tokenizer.encode(text)
        if len(ids) >= cfg.seq_len:
            sequences.append(ids[: cfg.seq_len])
            pbar.update(1)
        if len(sequences) >= cfg.n_sequences:
            break
    pbar.close()

    tokens = torch.tensor(sequences, dtype=torch.long)
    print(f"  Corpus: {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")
    return tokens


# ── Activation extraction ───────────────────────────────────────────────────

def extract_activations(model, tokens: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Run model on tokenized corpus, extract residual stream at target layer.

    Returns: (N, seq_len, d_model) float32 tensor.
    """
    N = tokens.shape[0]
    all_resid = []

    print(f"Extracting activations at {cfg.hook_point}...")
    with torch.no_grad():
        for i in tqdm(range(0, N, cfg.corpus_batch_size), desc="Extracting"):
            batch = tokens[i : i + cfg.corpus_batch_size].to(cfg.device)
            _, cache = model.run_with_cache(
                batch, names_filter=cfg.hook_point, return_type=None
            )
            resid = cache[cfg.hook_point].float().cpu()
            all_resid.append(resid)

    activations = torch.cat(all_resid, dim=0)
    print(f"  Activations: {activations.shape}  "
          f"({activations.numel() * 4 / 1e9:.2f} GB)")
    return activations


# ── LLM annotation ─────────────────────────────────────────────────────────

def build_annotation_prompt(
    token_strs: list[str],
    feature_chunk: list[dict],
    chunk_offset: int,
) -> str:
    """Build a prompt for Claude to annotate one sequence for a chunk of features."""
    token_block = " ".join(f"[{i}]{t}" for i, t in enumerate(token_strs))

    feat_block = "\n".join(
        f"F{chunk_offset + k} ({f['id']}): {_format_feature_for_annotator(f)}"
        for k, f in enumerate(feature_chunk)
    )

    return (
        f"Token sequence (index before each token):\n{token_block}\n\n"
        f"Feature definitions:\n{feat_block}\n\n"
        f"For each feature, list the token indices where it is CLEARLY active "
        f"based on the description. A feature activates on a token when that token "
        f"(in its surrounding context) matches the feature's description.\n\n"
        f"Reply ONLY with JSON: "
        f"{{\"F{chunk_offset}\": [indices], \"F{chunk_offset+1}\": [indices], ...}}.\n"
        f"If no tokens match a feature, use an empty list."
    )


def _extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from text, handling nested braces and strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def annotate_sequence_async(
    client,
    token_strs: list[str],
    features: list[dict],
    n_features: int,
    seq_idx: int,
    cfg: Config,
    semaphore: asyncio.Semaphore,
) -> tuple[int, torch.Tensor]:
    """Annotate one sequence for all features (chunked if needed).

    Returns: (seq_idx, labels_tensor) where labels_tensor is (seq_len, n_features).
    """
    labels = torch.zeros(len(token_strs), n_features)
    chunk_size = cfg.features_per_annotation_call

    for chunk_start in range(0, n_features, chunk_size):
        chunk = features[chunk_start : chunk_start + chunk_size]
        prompt = build_annotation_prompt(token_strs, chunk, chunk_start)

        async with semaphore:
            last_err = None
            succeeded = False
            for attempt in range(cfg.annotation_max_retries):
                try:
                    # Scale max_tokens with feature count: ~30 tokens per feature
                    max_toks = max(500, len(chunk) * 30)
                    response = await client.chat.completions.create(
                        model=cfg.annotation_model,
                        max_tokens=max_toks,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.choices[0].message.content.strip()
                    result = _extract_json_object(text)
                    # Treat JSON-parse failure as retryable, not as "success
                    # with zero labels". Previously a None result silently
                    # fell through the if-guard and broke out of the retry
                    # loop, yielding an all-zero label block with no
                    # diagnostic — bad labels are worse than missing labels.
                    if not result:
                        raise ValueError(
                            f"annotation JSON parse failed: {text[:200]!r}"
                        )
                    for k in range(len(chunk)):
                        key = f"F{chunk_start + k}"
                        indices = result.get(key, [])
                        for idx in indices:
                            if isinstance(idx, int) and 0 <= idx < len(token_strs):
                                labels[idx, chunk_start + k] = 1.0
                    succeeded = True
                    break
                except Exception as e:
                    last_err = e
                    if attempt < cfg.annotation_max_retries - 1:
                        delay = cfg.annotation_retry_base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
            if not succeeded:
                logger.warning(
                    "Annotation failed for seq %d, chunk %d-%d after %d "
                    "retries; leaving labels zero. Last error: %s",
                    seq_idx, chunk_start, chunk_start + len(chunk),
                    cfg.annotation_max_retries, last_err,
                )
                _ANNOTATION_FAILURE_COUNT.append(
                    (seq_idx, chunk_start, chunk_start + len(chunk), str(last_err))
                )

    return seq_idx, labels


async def annotate_corpus_async(
    tokens: torch.Tensor,
    features: list[dict],
    tokenizer,
    cfg: Config,
) -> torch.Tensor:
    """Annotate all sequences concurrently using Claude.

    Returns: (N, seq_len, n_features) float32 tensor of binary labels.
    """
    N, T = tokens.shape
    n_features = len(features)

    from .llm import get_async_client
    client = get_async_client()
    semaphore = asyncio.Semaphore(cfg.max_annotation_concurrency)
    # Reset the global failure accumulator for this run.
    _ANNOTATION_FAILURE_COUNT.clear()

    # Decode all sequences to strings (use .tolist() to avoid per-element overhead)
    print("Decoding tokens to strings...")
    all_token_strs = []
    for i in range(N):
        ids = tokens[i].tolist()
        strs = [tokenizer.decode([t]) for t in ids]
        all_token_strs.append(strs)

    print(f"Annotating {N} sequences x {n_features} features "
          f"with {cfg.annotation_model}...")

    # Resume from partial annotations if available
    wave_size = 100
    partial_path = cfg.output_dir / "annotations_partial.pt"
    progress_path = cfg.output_dir / "annotations_progress.txt"
    resume_from = 0

    if partial_path.exists() and progress_path.exists():
        all_labels = torch.load(partial_path, weights_only=True)
        if all_labels.shape == (N, T, n_features):
            resume_from = int(progress_path.read_text().strip())
            print(f"Resuming annotation from sequence {resume_from}/{N}")
        else:
            print("Partial annotations shape mismatch, starting fresh")
            all_labels = torch.zeros(N, T, n_features)
    else:
        all_labels = torch.zeros(N, T, n_features)

    t0 = time.time()
    completed = resume_from

    # Process in waves, saving partial results after each wave
    for wave_start in range(resume_from, N, wave_size):
        wave_end = min(wave_start + wave_size, N)
        tasks = [
            annotate_sequence_async(
                client, all_token_strs[i], features, n_features, i, cfg, semaphore
            )
            for i in range(wave_start, wave_end)
        ]
        results = await asyncio.gather(*tasks)

        for seq_idx, labels in results:
            all_labels[seq_idx] = labels

        completed += len(tasks)
        elapsed = time.time() - t0
        rate = (completed - resume_from) / elapsed if elapsed > 0 else 0
        remaining = N - completed
        eta = remaining / rate if rate > 0 else 0
        total_pos = int(all_labels[:completed].sum().item())
        print(f"  {completed}/{N} sequences  "
              f"({rate:.1f} seq/s, ETA {eta:.0f}s, {total_pos} positives)")

        # Save partial results after each wave (crash recovery)
        torch.save(all_labels, partial_path)
        progress_path.write_text(str(wave_end))

    elapsed = time.time() - t0
    total_pos = int(all_labels.sum().item())

    # Post-run failure audit. With parse-failure-as-exception wired in, every
    # left-zero chunk is now logged to _ANNOTATION_FAILURE_COUNT. Surface
    # the rate and abort if it crosses a configurable ceiling — a few
    # stragglers are tolerable, but 10%+ silent failure means the prompt or
    # model is broken and the labels can't be trusted.
    n_failed_chunks = len(_ANNOTATION_FAILURE_COUNT)
    chunk_size = cfg.features_per_annotation_call
    n_chunks_per_seq = (n_features + chunk_size - 1) // chunk_size
    total_chunks = max(N * n_chunks_per_seq, 1)
    failure_rate = n_failed_chunks / total_chunks
    threshold = getattr(cfg, "annotation_max_failure_rate", 0.10)

    if n_failed_chunks > 0:
        print(
            f"\n  WARNING: {n_failed_chunks}/{total_chunks} chunks "
            f"({failure_rate:.1%}) left zero after {cfg.annotation_max_retries} "
            f"retries. See first few below; full list in "
            f"annotations_failures.json."
        )
        for rec in _ANNOTATION_FAILURE_COUNT[:5]:
            print(f"    seq={rec[0]}  features [{rec[1]}..{rec[2]})  err={rec[3][:120]}")
        # Persist the audit log
        import json as _json
        (cfg.output_dir / "annotations_failures.json").write_text(_json.dumps([
            {"seq": s, "feat_start": cs, "feat_end": ce, "error": err}
            for s, cs, ce, err in _ANNOTATION_FAILURE_COUNT
        ], indent=2))
        if failure_rate > threshold:
            raise RuntimeError(
                f"annotation failure rate {failure_rate:.1%} exceeds "
                f"threshold {threshold:.1%}; aborting rather than saving "
                f"a catalog with silently-zeroed labels. Inspect "
                f"annotations_failures.json and lower the prompt / "
                f"max_tokens / chunk_size, or raise the threshold via "
                f"cfg.annotation_max_failure_rate."
            )

    # Remove partial checkpoints on success
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    print(f"\nAnnotation complete in {elapsed:.1f}s")
    print(f"  Total positives: {total_pos}")

    # Per-feature stats
    print(f"\n  {'Feature':<40} {'Pos':>8} {'Rate':>8}")
    print("  " + "-" * 58)
    for k, feat in enumerate(features):
        n_pos = int(all_labels[:, :, k].sum().item())
        rate = all_labels[:, :, k].mean().item()
        tag = " [group]" if feat.get("type") == "group" else ""
        print(f"  {feat['id']:<40} {n_pos:>8} {rate:>7.4%}{tag}")

    return all_labels


def annotate_corpus(
    tokens: torch.Tensor,
    features: list[dict],
    tokenizer,
    cfg: Config,
) -> torch.Tensor:
    """Synchronous wrapper for async annotation."""
    return asyncio.run(
        annotate_corpus_async(tokens, features, tokenizer, cfg)
    )


# ── v2: Local model annotation (decomposed single-feature single-token) ───

def _format_annotator_context(token_strs: list[str], target_pos: int) -> str:
    """Format a token sequence with the target token marked by >><<."""
    parts = []
    for i, t in enumerate(token_strs):
        if i == target_pos:
            parts.append(f">>{t}<<")
        else:
            parts.append(t)
    return "".join(parts)


def annotate_local(
    tokens: torch.Tensor,
    features: list[dict],
    base_tokenizer,
    cfg: Config,
) -> torch.Tensor:
    """Local annotation with vLLM, optimized prefix caching.

    Architecture for maximum cache reuse:
      Level 0: System prompt with ALL feature definitions (~200 tokens)
               Computed ONCE, cached across every prompt in the run.
      Level 1: Sequence tokens up to position k (~k tokens)
               Grows by 1 token per position, cached across all features.
      Variable: Feature ID "\\nF12\\n" + assistant tag (~6 tokens)
               The only non-cached part per prompt.

    vLLM (primary), HF transformers (fallback).
    """
    N, T = tokens.shape
    n_features = len(features)

    # Decode all tokens to strings once
    print("Decoding tokens to strings...")
    all_token_strs = []
    for i in range(N):
        strs = [base_tokenizer.decode([t]) for t in tokens[i].tolist()]
        all_token_strs.append(strs)

    try:
        from vllm import LLM, SamplingParams
        if cfg.batch_positions:
            return _annotate_local_vllm_batch(
                tokens, features, all_token_strs, n_features, N, T, cfg,
            )
        else:
            return _annotate_local_vllm_pertoken(
                tokens, features, all_token_strs, n_features, N, T, cfg,
            )
    except ImportError:
        print("WARNING: vLLM not installed. Falling back to HuggingFace transformers.")
        print("  Install vLLM: pip install vllm")
        return _annotate_local_hf(
            tokens, features, all_token_strs, n_features, N, T, cfg,
        )


def _benchmark_vllm_ids(llm, params, sys_ids, tok_ids_per_seq, suffix_ids,
                         n_features, T, N):
    """4-run cache isolation test.

    Run 1: Cold — first time seeing these prompts
    Run 2: Full cache hit — exact same prompts (should be 5-10x faster)
    Run 3: Shared prefix — new suffixes, same sequence prefixes
    Run 4: Cache miss — completely different text

    If Run 2 isn't dramatically faster than Run 4, caching is broken.
    """
    import time as _time

    bench_feats = min(5, n_features)

    def _build_seq(seq_j, n_feats, feat_offset=0):
        prompts = []
        prefix = list(sys_ids)
        for tok_k in range(T):
            prefix = prefix + list(tok_ids_per_seq[seq_j][tok_k])
            for fi in range(feat_offset, min(feat_offset + n_feats, len(suffix_ids))):
                prompts.append({"prompt_token_ids": prefix + suffix_ids[fi]})
        return prompts

    batch1 = _build_seq(0, bench_feats, feat_offset=0)
    n = len(batch1)
    print(f"\n  Cache isolation test ({n} prompts per batch)...")

    # Run 1: Cold
    t0 = _time.time()
    llm.generate(batch1, params)
    t_cold = _time.time() - t0

    # Run 2: Full cache hit (exact same prompts)
    t0 = _time.time()
    llm.generate(batch1, params)
    t_hit = _time.time() - t0

    # Run 3: Same prefixes, different suffixes
    if n_features > bench_feats:
        batch2 = _build_seq(0, bench_feats, feat_offset=bench_feats)
    else:
        batch2 = batch1
    t0 = _time.time()
    llm.generate(batch2, params)
    t_shared = _time.time() - t0

    # Run 4: Completely different text (different sequence)
    diff_seq = min(1, N - 1)
    batch3 = _build_seq(diff_seq, bench_feats, feat_offset=0)
    t0 = _time.time()
    llm.generate(batch3, params)
    t_miss = _time.time() - t0

    r_cold = n / t_cold
    r_hit = n / t_hit
    r_shared = n / t_shared
    r_miss = n / t_miss

    print(f"    Run 1 (cold):           {r_cold:>7.0f} p/s  ({t_cold:.1f}s)")
    print(f"    Run 2 (full cache hit):  {r_hit:>7.0f} p/s  ({t_hit:.1f}s)")
    print(f"    Run 3 (shared prefix):   {r_shared:>7.0f} p/s  ({t_shared:.1f}s)")
    print(f"    Run 4 (cache miss):      {r_miss:>7.0f} p/s  ({t_miss:.1f}s)")

    hit_vs_miss = r_hit / r_miss if r_miss > 0 else 1
    shared_vs_miss = r_shared / r_miss if r_miss > 0 else 1

    if hit_vs_miss > 3:
        status = "EXCELLENT"
    elif hit_vs_miss > 1.2:
        status = "WORKING"
    else:
        status = "NO BENEFIT"

    print(f"    Hit/miss ratio: {hit_vs_miss:.1f}x — {status}")
    print(f"    Prefix reuse:   {shared_vs_miss:.1f}x")

    # Select chunk size — use the warm rate as the baseline
    # More sequences per chunk = more prompts per llm.generate() call
    # vLLM handles the internal batching, so bigger chunks are fine
    if r_hit > 500:
        chunk = min(5, N)   # fast model, can handle more
    elif r_hit > 100:
        chunk = min(3, N)
    else:
        chunk = min(2, N)

    print(f"    Selected: seq_chunk={chunk}")
    return chunk


def _annotate_local_vllm_pertoken(
    tokens: torch.Tensor,
    features: list[dict],
    all_token_strs: list[list[str]],
    n_features: int,
    N: int,
    T: int,
    cfg: Config,
) -> torch.Tensor:
    """vLLM per-token with guaranteed-perfect prefix caching.

    All prompts built as TOKEN ID lists (not strings) to guarantee
    token-level prefix alignment. BPE merges across string boundaries
    can't break the prefix match when IDs are pre-tokenized separately.

    Cache hierarchy:
      Level 0: sys_ids (~200 tokens) — shared across ALL prompts
      Level 1: + tok_ids[0:k+1] — grows by exact token IDs per position
      Variable: suffix_ids[fi] (~6 tokens) — only non-cached part
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading local annotator via vLLM: {cfg.local_annotator_model}")

    ann_tokenizer = AutoTokenizer.from_pretrained(cfg.local_annotator_model)
    tok_0_id = ann_tokenizer.encode("0", add_special_tokens=False)[0]
    tok_1_id = ann_tokenizer.encode("1", add_special_tokens=False)[0]
    print(f"  Token IDs: '0'={tok_0_id}, '1'={tok_1_id}")

    # ── Build system prefix and suffix caches ──────────────────────────
    # Both modes share the same cache structure:
    #   pos_prefix = sys + text:" + tok_0...tok_k + '"\nToken: "X". '  (CACHED)
    #   suffix = feature question only                                  (NOT cached)
    # F-index suffix: "F3? " (~3 tok).  Full-desc suffix: "description? " (~8-10 tok)
    use_findex = cfg.use_findex_suffix
    if use_findex:
        feat_defs = "\n".join(
            f"F{i}: {_format_feature_for_annotator(f)}" for i, f in enumerate(features)
        )
        SYS_STR = (
            f'Answer 0 (no) or 1 (yes).\n\n'
            f'Features:\n{feat_defs}\n\n'
            f'Text: "The cat sat on the mat ,"\n'
            f'Token: ",". F0? 1\n'
            f'Token: "mat". F0? 0\n'
            f'Token: "The". F5? 1\n'
            f'Token: "cat". F5? 0\n\n'
        )
    else:
        SYS_STR = (
            'Answer 0 (no) or 1 (yes).\n\n'
            'Text: "The cat sat on the mat ,"\nToken: ",". A comma? 1\n'
            'Text: "The cat sat on the mat ,"\nToken: "mat". A comma? 0\n'
            'Text: "The cat sat on the mat ,"\nToken: "The". Starts with uppercase? 1\n'
            'Text: "The cat sat on the mat ,"\nToken: "cat". Starts with uppercase? 0\n\n'
        )

    sys_ids = ann_tokenizer.encode(SYS_STR, add_special_tokens=False)

    # Pre-tokenize each GPT-2 token string
    print("  Pre-tokenizing sequence tokens...")
    tok_ids_per_seq = []
    for seq_j in range(N):
        seq_tok_ids = []
        for tok_k in range(T):
            piece = all_token_strs[seq_j][tok_k]
            ids = ann_tokenizer.encode(piece, add_special_tokens=False)
            seq_tok_ids.append(ids)
        tok_ids_per_seq.append(seq_tok_ids)

    # Token name cache: cached across features at same position.
    #
    # IMPORTANT: preserve the raw token string — do NOT strip whitespace or
    # lowercase. For GPT-2-style BPE the leading space on " The" is the
    # word-boundary marker, and the distinction between "US" and "us" is a
    # real feature. We JSON-encode the token so embedded quotes, newlines,
    # and tabs survive as escape sequences the model can read unambiguously.
    tok_name_cache = {}
    import json as _json
    for seq_j in range(N):
        for tok_k in range(T):
            tok_str = all_token_strs[seq_j][tok_k]
            if tok_str not in tok_name_cache:
                # Inline a closing quote + newline for the Text:"..." block,
                # then a json-encoded token literal that preserves whitespace.
                s = f'"\nToken: {_json.dumps(tok_str)}. '
                tok_name_cache[tok_str] = tuple(
                    ann_tokenizer.encode(s, add_special_tokens=False)
                )

    # Feature suffix: only non-cached part per prompt. Descriptions are
    # NOT lowercased any more — "Token is US" and "Token is us" are
    # different questions for any sensible annotator, and plenty of our
    # features are case-sensitive by design (capitalization, acronym,
    # code identifier).
    if use_findex:
        feat_suffix_list = [
            tuple(ann_tokenizer.encode(f'F{fi}? ', add_special_tokens=False))
            for fi in range(n_features)
        ]
    else:
        feat_suffix_list = [
            tuple(ann_tokenizer.encode(
                f'{_format_feature_for_annotator(f)}? ',
                add_special_tokens=False,
            ))
            for f in features
        ]

    text_open_ids = ann_tokenizer.encode('Text: "', add_special_tokens=False)

    # Compute actual max prompt length → set max_model_len tightly
    max_text_toks = max(
        sum(len(tok_ids_per_seq[j][k]) for k in range(T))
        for j in range(N)
    )
    max_name_toks = max(len(v) for v in tok_name_cache.values())
    max_sfx_toks = max(len(s) for s in feat_suffix_list)
    max_prompt = len(sys_ids) + len(text_open_ids) + max_text_toks + max_name_toks + max_sfx_toks
    # Round up to next multiple of 64
    computed_max_len = min(8192, ((max_prompt + 63) // 64) * 64)

    avg_sfx = sum(len(s) for s in feat_suffix_list) / max(len(feat_suffix_list), 1)
    mode_str = "F-index" if use_findex else "full-desc"
    print(f"  System prefix: {len(sys_ids)} tokens ({mode_str})")
    print(f"  Token names: {len(tok_name_cache)} unique (cached per-position)")
    print(f"  Feature suffixes: avg {avg_sfx:.1f} tok (NOT cached)")
    print(f"  Max prompt: {max_prompt} tok → max_model_len={computed_max_len}")

    # Debug: show sample prompt
    sample_tok = all_token_strs[0][0].strip()
    sample_prompt = (
        sys_ids + text_open_ids + list(tok_ids_per_seq[0][0])
        + list(tok_name_cache[sample_tok]) + list(feat_suffix_list[0])
    )
    print(f"\n  Sample prompt (seq=0, pos=0, feat=0):")
    print(f"  {ann_tokenizer.decode(sample_prompt)}")
    print()
    del ann_tokenizer

    # vLLM spawns a subprocess for the engine. If CUDA was already
    # initialized (e.g., by HookedTransformer), the fork fails.
    # Force 'spawn' start method so the subprocess gets a clean CUDA context.
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

    # Free any CUDA memory from previous steps
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    llm = LLM(
        model=cfg.local_annotator_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_model_len=computed_max_len,
        gpu_memory_utilization=0.80,
    )
    # Base model: no thinking, no chat template. Just completes text.
    # allowed_token_ids forces "0" or "1" — safe on base models.
    params = SamplingParams(
        max_tokens=1,
        temperature=0,
        allowed_token_ids=[tok_0_id, tok_1_id],
    )

    total_decisions = n_features * N * T
    t_start = time.time()

    seq_chunk = min(getattr(cfg, "local_annotation_seq_chunk", 2), N)

    # Resume from partial checkpoint (crash recovery)
    partial_path = cfg.output_dir / "annotations_local_partial.pt"
    progress_path = cfg.output_dir / "annotations_local_progress.txt"
    resume_from = 0

    if partial_path.exists() and progress_path.exists():
        annotations = torch.load(partial_path, weights_only=True)
        if annotations.shape == (N, T, n_features):
            resume_from = int(progress_path.read_text().strip())
            print(f"Resuming local annotation from sequence {resume_from}/{N}")
        else:
            print("Partial annotations shape mismatch, starting fresh")
            annotations = torch.zeros(N, T, n_features)
    else:
        annotations = torch.zeros(N, T, n_features)

    completed = resume_from * T * n_features

    print(f"\nAnnotating: {n_features} features x {N} sequences x {T} tokens "
          f"= {total_decisions:,} decisions "
          f"(vLLM, token-ID prefixes, chunk={seq_chunk})")

    # Convert to tuples for faster concatenation
    sys_tuple = tuple(sys_ids)
    text_open_tuple = tuple(text_open_ids)
    tok_id_tuples = [
        [tuple(tok_ids_per_seq[j][k]) for k in range(T)]
        for j in range(N)
    ]

    for seq_start in range(resume_from, N, seq_chunk):
        seq_end = min(seq_start + seq_chunk, N)

        prompts = []
        positions = []

        for seq_j in range(seq_start, seq_end):
            prefix = sys_tuple + text_open_tuple
            for tok_k in range(T):
                prefix = prefix + tok_id_tuples[seq_j][tok_k]
                tok_str = all_token_strs[seq_j][tok_k]  # preserve whitespace
                # Token name in prefix: cached across all features at this position
                pos_prefix = prefix + tok_name_cache[tok_str]
                for fi in range(n_features):
                    prompts.append({
                        "prompt_token_ids": list(pos_prefix + feat_suffix_list[fi])
                    })
                    positions.append((seq_j, tok_k, fi))

        outputs = llm.generate(prompts, params)

        batch_time = time.time()

        # Debug + cache verification on first chunk
        if seq_start == 0:
            print("\n  First 10 raw outputs:")
            for di in range(min(10, len(outputs))):
                sj, tk, fi_d = positions[di]
                raw = outputs[di].outputs[0].text
                tok = all_token_strs[sj][tk].strip()
                print(f"    [{di}] tok='{tok}' feat={fi_d} -> '{raw}'")

            # Prefix cache check: re-run same batch, should be 2-10x faster
            t_cold = batch_time - t_start
            t0 = time.time()
            llm.generate(prompts[:min(512, len(prompts))], params)
            t_warm = time.time() - t0
            speedup = t_cold / max(t_warm, 1e-6)
            n_check = min(512, len(prompts))
            print(f"\n  PREFIX CACHE CHECK ({n_check} prompts):")
            print(f"    Cold: {t_cold:.2f}s  Warm: {t_warm:.2f}s  Speedup: {speedup:.1f}x")
            if speedup < 1.5:
                print("    WARNING: prefix caching may not be working!")
            else:
                print("    OK — caching is active.")
            print()

        for idx, output in enumerate(outputs):
            seq_j, tok_k, fi = positions[idx]
            tid = output.outputs[0].token_ids[0]
            annotations[seq_j, tok_k, fi] = 1.0 if tid == tok_1_id else 0.0

        completed += len(prompts)
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta_sec = (total_decisions - completed) / rate if rate > 0 else 0
        print(f"  {completed:,}/{total_decisions:,} decisions  "
              f"rate={rate:.0f}/s  ETA {eta_sec/3600:.1f}h")

        # Save checkpoint after each chunk (crash recovery)
        torch.save(annotations, partial_path)
        progress_path.write_text(str(seq_end))

    # Clean up partial checkpoints on success
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()

    del llm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return annotations


def _annotate_local_vllm_batch(
    tokens: torch.Tensor,
    features: list[dict],
    all_token_strs: list[list[str]],
    n_features: int,
    N: int,
    T: int,
    cfg: Config,
) -> torch.Tensor:
    """vLLM batch-positions: full sequence + all features → JSON indices.

    Same prompt format as the API annotation path: show the full indexed
    sequence, list all feature definitions, ask for JSON with position
    indices per feature. One prompt per sequence (all features at once).

    This gives the model full context for every token — critical for
    features that depend on sentence structure (e.g., IOI name roles).
    """
    from vllm import LLM, SamplingParams

    print(f"Loading local annotator via vLLM: {cfg.local_annotator_model}")
    llm = LLM(
        model=cfg.local_annotator_model,
        dtype="auto",
        enable_prefix_caching=True,
        max_model_len=2048,
    )
    # Generous output for JSON with position lists
    max_out = max(500, n_features * 30)
    params = SamplingParams(max_tokens=max_out, temperature=0)

    annotations = torch.zeros(N, T, n_features)
    total_decisions = n_features * N * T
    t_start = time.time()

    seq_chunk = max(1, min(100, N))

    print(f"Annotating: {n_features} features x {N} sequences x {T} tokens "
          f"= {total_decisions:,} decisions "
          f"(vLLM, full-sequence JSON, {N} prompts)")

    for seq_start in range(0, N, seq_chunk):
        seq_end = min(seq_start + seq_chunk, N)

        prompts = []
        for seq_j in range(seq_start, seq_end):
            prompt_text = build_annotation_prompt(
                all_token_strs[seq_j], features, 0,
            )
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, params)

        for j, output in enumerate(outputs):
            seq_j = seq_start + j
            text = output.outputs[0].text.strip()
            result = _extract_json_object(text)
            if result:
                for k in range(n_features):
                    key = f"F{k}"
                    indices = result.get(key, [])
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < T:
                            annotations[seq_j, idx, k] = 1.0
            else:
                # Silent-zero was the pre-v8.5 behavior. Now: warn so the
                # failure is auditable, and track the rate across the run.
                _ANNOTATION_FAILURE_COUNT.append(
                    (seq_j, 0, n_features, f"vLLM parse fail: {text[:160]!r}")
                )

        elapsed = time.time() - t_start
        done = (seq_end) * n_features * T
        rate = done / elapsed if elapsed > 0 else 0
        eta_sec = (total_decisions - done) / rate if rate > 0 else 0
        print(f"  {seq_end}/{N} sequences  rate={rate:.0f} decisions/s  "
              f"ETA {eta_sec/3600:.1f}h")

    del llm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return annotations


def _annotate_local_hf(
    tokens: torch.Tensor,
    features: list[dict],
    all_token_strs: list[list[str]],
    n_features: int,
    N: int,
    T: int,
    cfg: Config,
) -> torch.Tensor:
    """HuggingFace transformers fallback (no prefix caching).

    Uses logit comparison (token "1" vs "0") instead of generate() for speed,
    but re-processes the full prompt for every decision. 10-50x slower than vLLM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local annotator via transformers: {cfg.local_annotator_model}")
    ann_tokenizer = AutoTokenizer.from_pretrained(cfg.local_annotator_model)
    ann_model = AutoModelForCausalLM.from_pretrained(
        cfg.local_annotator_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ann_model.eval()

    if ann_tokenizer.pad_token is None:
        ann_tokenizer.pad_token = ann_tokenizer.eos_token
    ann_tokenizer.padding_side = "left"

    tok_0 = ann_tokenizer.encode("0", add_special_tokens=False)[0]
    tok_1 = ann_tokenizer.encode("1", add_special_tokens=False)[0]

    annotations = torch.zeros(N, T, n_features)
    batch_size = cfg.local_annotation_batch_size
    total_decisions = n_features * N * T
    completed = 0
    t_start = time.time()

    print(f"Annotating: {n_features} features x {N} sequences x {T} tokens "
          f"= {total_decisions:,} decisions (HF transformers, no prefix caching)")

    for fi, feat in enumerate(features):
        feat_prefix = (
            "You are a feature annotator. "
            f"Feature: {_format_feature_for_annotator(feat)}\n"
            "Output only 0 or 1. 1 if the feature activates at the LAST token.\n\n"
        )

        for seq_start in range(0, N, batch_size):
            seq_end = min(seq_start + batch_size, N)
            batch_seqs = list(range(seq_start, seq_end))

            for tok_k in range(T):
                prompts = []
                for seq_j in batch_seqs:
                    # Incremental: show tokens 0..tok_k, annotate the last
                    context = "".join(all_token_strs[seq_j][:tok_k + 1])
                    prompts.append(f"{feat_prefix}{context}\nAnswer:")

                inputs = ann_tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                inputs = {k: v.to(ann_model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = ann_model(**inputs)
                    last_logits = outputs.logits[:, -1, :]
                    labels = (last_logits[:, tok_1] > last_logits[:, tok_0]).float()

                for j, seq_j in enumerate(batch_seqs):
                    annotations[seq_j, tok_k, fi] = labels[j].item()

                completed += len(batch_seqs)

        elapsed_total = time.time() - t_start
        rate = completed / elapsed_total if elapsed_total > 0 else 0
        eta_sec = (total_decisions - completed) / rate if rate > 0 else 0
        n_pos = int(annotations[:, :, fi].sum().item())
        print(f"  [{fi+1}/{n_features}] {feat['id']:<36} "
              f"pos={n_pos:>6} rate={rate:.0f}/s  "
              f"ETA {eta_sec/3600:.1f}h")

    del ann_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return annotations


# ── Propagate group labels ──────────────────────────────────────────────────

def filter_sparse_features(
    features: list[dict], annotations: torch.Tensor, min_rate: float
) -> tuple[list[dict], torch.Tensor, list[str]]:
    """Remove leaf features with positive rate below min_rate and orphaned groups.

    Returns (filtered_features, filtered_annotations, removed_ids).
    """
    rates = annotations.mean(dim=(0, 1))  # per-feature positive rate

    # Step 1: identify surviving leaves
    surviving_ids = set()
    removed_ids = []
    for i, feat in enumerate(features):
        if feat["type"] == "leaf":
            if rates[i].item() >= min_rate:
                surviving_ids.add(feat["id"])
            else:
                removed_ids.append(feat["id"])

    if not removed_ids:
        return features, annotations, []

    # Step 2: keep groups that still have at least one surviving descendant
    changed = True
    while changed:
        changed = False
        for feat in features:
            if feat["type"] == "group" and feat["id"] not in surviving_ids:
                has_child = any(
                    f.get("parent") == feat["id"] and f["id"] in surviving_ids
                    for f in features
                )
                if has_child:
                    surviving_ids.add(feat["id"])
                    changed = True

    for feat in features:
        if feat["type"] == "group" and feat["id"] not in surviving_ids:
            removed_ids.append(feat["id"])

    keep_indices = [i for i, f in enumerate(features) if f["id"] in surviving_ids]
    filtered_features = [features[i] for i in keep_indices]
    filtered_annotations = annotations[:, :, keep_indices]

    return filtered_features, filtered_annotations, removed_ids


def propagate_group_labels(annotations: torch.Tensor, features: list[dict]) -> torch.Tensor:
    """Set group feature labels = OR of children's labels.

    Processes groups bottom-up (leaf-ward groups first) so nested hierarchies
    propagate correctly: sub-group labels are computed before parent groups.
    """
    feat_id_to_idx = {f["id"]: i for i, f in enumerate(features)}

    # Build parent→children map
    children_of: dict[int, list[int]] = {}
    for f in features:
        if f.get("parent") and f["parent"] in feat_id_to_idx:
            parent_idx = feat_id_to_idx[f["parent"]]
            children_of.setdefault(parent_idx, []).append(feat_id_to_idx[f["id"]])

    # Topological sort: compute depth of each group (max distance to a leaf)
    group_idxs = [i for i, f in enumerate(features) if f["type"] == "group"]

    def depth(idx: int) -> int:
        kids = children_of.get(idx, [])
        group_kids = [k for k in kids if features[k]["type"] == "group"]
        if not group_kids:
            return 0
        return 1 + max(depth(k) for k in group_kids)

    # Process shallowest (closest to leaves) first
    group_idxs.sort(key=depth)

    for idx in group_idxs:
        child_idxs = children_of.get(idx, [])
        if child_idxs:
            annotations[:, :, idx] = (
                annotations[:, :, child_idxs].max(dim=-1).values
            )

    return annotations


# ── Main entry point ────────────────────────────────────────────────────────

def run(cfg: Config = None):
    """Run corpus preparation + activation extraction + LLM annotation."""
    if cfg is None:
        cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature catalog
    if not cfg.catalog_path.exists():
        raise FileNotFoundError(
            f"Feature catalog not found: {cfg.catalog_path}\n"
            "Run pipeline.inventory first."
        )
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    # Only annotate leaf features (groups are derived)
    leaf_features = [f for f in features if f["type"] == "leaf"]
    leaf_indices = [i for i, f in enumerate(features) if f["type"] == "leaf"]
    print(f"Feature catalog: {len(features)} total, {len(leaf_features)} leaves to annotate")

    # ── Tokenization + activation extraction ──────────────────────────
    # Run in a subprocess if needed so CUDA context doesn't corrupt vLLM.
    # The subprocess loads GPT-2, extracts, saves to disk, and exits cleanly.
    # Cache freshness is checked by sidecar meta BEFORE deciding to reuse —
    # a layer-9 activations.pt silently reused in a layer-6 run is the
    # exact silent corruption the sidecars are designed to catch.
    from .cache_meta import (
        load_or_die as _cache_load_or_die,
        write_cache_meta as _cache_write_meta,
    )
    tokens_ok = cfg.tokens_path.exists() and _cache_load_or_die(
        cfg.tokens_path, "tokens", cfg,
    )
    acts_ok = cfg.activations_path.exists() and _cache_load_or_die(
        cfg.activations_path, "activations", cfg,
    )
    need_tokens = not tokens_ok
    need_acts = not acts_ok

    if need_tokens or need_acts:
        import subprocess, sys
        print("Extracting tokens/activations in subprocess (isolates CUDA context)...")
        extract_script = f"""
import torch
from pipeline.annotate import prepare_corpus, extract_activations
from pipeline.inventory import load_target_model
from pipeline.config import Config

cfg = Config(
    model_name="{cfg.model_name}", device="{cfg.device}",
    model_dtype="{cfg.model_dtype}", n_sequences={cfg.n_sequences},
    seq_len={cfg.seq_len}, corpus_batch_size={cfg.corpus_batch_size},
    target_layer={cfg.target_layer}, output_dir="{cfg.output_dir}",
    hook_point="{cfg.hook_point}",
    sae_release="{cfg.sae_release}", sae_id="{cfg.sae_id}",
)
cfg.output_dir.mkdir(parents=True, exist_ok=True)

# Use the centralized loader so activations match the pretrained SAE's
# training distribution (no LayerNorm folding, etc.).
model = load_target_model(cfg)

from pipeline.cache_meta import write_cache_meta

if not cfg.tokens_path.exists():
    tokens = prepare_corpus(model, cfg)
    torch.save(tokens, cfg.tokens_path)
    write_cache_meta(cfg.tokens_path, "tokens", cfg)
    print(f"Saved tokens: {{cfg.tokens_path}}")
else:
    tokens = torch.load(cfg.tokens_path, weights_only=True)

if not cfg.activations_path.exists():
    activations = extract_activations(model, tokens, cfg)
    torch.save(activations, cfg.activations_path)
    write_cache_meta(cfg.activations_path, "activations", cfg)
    print(f"Saved activations: {{cfg.activations_path}}")
"""
        result = subprocess.run(
            [sys.executable, "-c", extract_script],
            check=True,
        )
        print("Subprocess complete. CUDA context cleaned up.")

    # Load cached data (extracted by subprocess or previous run)
    print(f"Loading cached tokens: {cfg.tokens_path}")
    tokens = torch.load(cfg.tokens_path, weights_only=True)
    print(f"Loading cached activations: {cfg.activations_path}")
    activations = torch.load(cfg.activations_path, weights_only=True)

    # Get tokenizer without loading the full model onto GPU
    from transformers import AutoTokenizer
    tokenizer_ref = AutoTokenizer.from_pretrained(cfg.model_name)

    # Annotate (or load cached). The cache is validated against the current
    # catalog by FEATURE ID via the `annotations_meta.json` sidecar — this
    # prevents silent misbinding when the catalog is reordered or features
    # are removed. A tensor without a sidecar is treated as "legacy positional"
    # and accepted only when shape matches exactly; any mismatch triggers a
    # warning and a full re-annotation because we can't recover the ID
    # sequence after the fact.
    N_tok, T_tok = tokens.shape
    n_features = len(features)
    current_ids = [f["id"] for f in features]

    annotations = None
    new_feature_indices: list[int] = []

    if cfg.annotations_path.exists():
        cached = torch.load(cfg.annotations_path, weights_only=True)
        cached_shape_ok = (cached.shape[:2] == (N_tok, T_tok))
        cached_ids: list[str] | None = None
        if cfg.annotations_meta_path.exists():
            try:
                meta = json.loads(cfg.annotations_meta_path.read_text())
                cached_ids = meta.get("feature_ids")
                if not isinstance(cached_ids, list) or len(cached_ids) != cached.shape[-1]:
                    print(
                        f"  annotations_meta.json is malformed (ids "
                        f"{len(cached_ids) if isinstance(cached_ids, list) else 'n/a'} "
                        f"vs tensor last-dim {cached.shape[-1]}). Discarding cache."
                    )
                    cached_ids = None
                    cached_shape_ok = False
            except json.JSONDecodeError:
                print("  annotations_meta.json is not valid JSON. Discarding cache.")
                cached_shape_ok = False
                cached_ids = None

        if not cached_shape_ok:
            print(
                f"  Cached annotations shape {tuple(cached.shape)} incompatible "
                f"with (N={N_tok}, T={T_tok}). Re-annotating from scratch."
            )
        elif cached_ids is not None:
            # ID-based remap: each current feature pulls its column from the
            # cache by ID. Missing IDs → all-zero placeholder + schedule for
            # re-annotation. Extra cached IDs (features no longer in catalog)
            # are silently dropped. Safe under any catalog reorder / insert /
            # delete.
            cached_by_id: dict[str, int] = {}
            for i, fid in enumerate(cached_ids):
                # If the same id appears twice (shouldn't happen, but guard),
                # the first occurrence wins.
                cached_by_id.setdefault(fid, i)

            annotations = torch.zeros(N_tok, T_tok, n_features)
            hit_count = 0
            for ki, fid in enumerate(current_ids):
                src = cached_by_id.get(fid)
                if src is not None:
                    annotations[:, :, ki] = cached[:, :, src]
                    hit_count += 1
                else:
                    new_feature_indices.append(ki)

            n_new = len(new_feature_indices)
            n_dropped = len(cached_ids) - hit_count
            print(
                f"ID-keyed cache reuse: {hit_count}/{n_features} features matched "
                f"by id, {n_new} new features scheduled for annotation"
                + (f", {n_dropped} cached-only features dropped" if n_dropped else "")
                + "."
            )
        else:
            # Legacy positional cache: no sidecar. Warn and fall back to the
            # old "catalog-grew-at-the-tail" behavior only when the shape is
            # consistent; otherwise discard.
            if cached.shape[-1] == n_features:
                print(
                    "  WARNING: cached annotations lack an id sidecar. "
                    "Assuming positional alignment; delete annotations.pt to "
                    "regenerate safely."
                )
                annotations = cached.clone()
            elif cached.shape[-1] < n_features:
                n_cached_features = cached.shape[-1]
                print(
                    f"  WARNING: cached annotations (n_features={n_cached_features}) "
                    f"lack an id sidecar. Falling back to legacy tail-append "
                    f"assumption; delete annotations.pt to regenerate safely."
                )
                annotations = torch.zeros(N_tok, T_tok, n_features)
                annotations[:, :, :n_cached_features] = cached
                new_feature_indices = list(range(n_cached_features, n_features))
            else:
                print(
                    f"  Cached annotations have more columns ({cached.shape[-1]}) "
                    f"than current catalog ({n_features}) and no id sidecar. "
                    f"Discarding cache."
                )

    if annotations is None:
        annotations = torch.zeros(N_tok, T_tok, n_features)
        new_feature_indices = list(range(n_features))

    # Narrow the annotation pass to leaves only (groups are derived via OR).
    new_leaf_features: list[dict] = []
    new_leaf_feat_indices: list[int] = []  # index into `features`
    for fi in new_feature_indices:
        feat = features[fi]
        if feat.get("type") == "leaf":
            new_leaf_features.append(feat)
            new_leaf_feat_indices.append(fi)

    if new_leaf_features:
        print(
            f"  Annotating {len(new_leaf_features)} new leaves "
            f"(out of {len(leaf_features)} total leaves in current catalog)"
        )
        if cfg.use_local_annotator:
            new_leaf_annotations = annotate_local(
                tokens, new_leaf_features, tokenizer_ref, cfg,
            )
        else:
            new_leaf_annotations = annotate_corpus(
                tokens, new_leaf_features, tokenizer_ref, cfg,
            )
        for k, fi in enumerate(new_leaf_feat_indices):
            annotations[:, :, fi] = new_leaf_annotations[:, :, k]

    # Group labels are always re-derived from their (possibly freshly
    # annotated) leaves — cheap and handles the reorder/remap case.
    annotations = propagate_group_labels(annotations, features)

    # Filter out features with very low positive rate
    if cfg.min_feature_positive_rate > 0:
        features, annotations, removed = filter_sparse_features(
            features, annotations, cfg.min_feature_positive_rate
        )
        if removed:
            print(f"\n  Filtered {len(removed)} sparse features "
                  f"(rate < {cfg.min_feature_positive_rate:.2%}):")
            for fid in removed:
                print(f"    - {fid}")
            # Update catalog on disk to match filtered annotations
            catalog["features"] = features
            cfg.catalog_path.write_text(json.dumps(catalog, indent=2))
            print(f"  Updated catalog: {len(features)} features remain")
            current_ids = [f["id"] for f in features]

    # Always (re)write the sidecar alongside the tensor so future runs can
    # re-align by id.
    torch.save(annotations, cfg.annotations_path)
    cfg.annotations_meta_path.write_text(json.dumps(
        {
            "feature_ids": current_ids,
            "shape": list(annotations.shape),
            "version": 1,
        },
        indent=2,
    ))
    print(
        f"Saved annotations: {cfg.annotations_path} "
        f"(+ sidecar {cfg.annotations_meta_path.name})"
    )

    return tokens, activations, annotations


if __name__ == "__main__":
    run()
