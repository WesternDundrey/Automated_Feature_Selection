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
        f"F{chunk_offset + k} ({f['id']}): {f['description']}"
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
                    if result:
                        for k in range(len(chunk)):
                            key = f"F{chunk_start + k}"
                            indices = result.get(key, [])
                            for idx in indices:
                                if isinstance(idx, int) and 0 <= idx < len(token_strs):
                                    labels[idx, chunk_start + k] = 1.0
                    break  # Success — exit retry loop
                except Exception as e:
                    last_err = e
                    if attempt < cfg.annotation_max_retries - 1:
                        delay = cfg.annotation_retry_base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
            else:
                # All retries exhausted → labels stay zero (safe default)
                logger.warning(
                    "Annotation failed for seq %d, chunk %d-%d after %d retries: %s",
                    seq_idx, chunk_start, chunk_start + len(chunk),
                    cfg.annotation_max_retries, last_err,
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
    """Benchmark vLLM with token-ID prompts. Verify caching, find optimal chunk.

    Tests:
      1. Cold vs warm (same prompts twice) — verifies cache persistence
      2. Ordered vs shuffled positions — verifies incremental prefix caching
      3. Chunk size sweep
    """
    import time as _time
    import random as _random

    bench_seqs = min(1, N)
    bench_feats = min(5, n_features)  # small subset for speed

    def _build_ordered(n_seqs, n_feats):
        """Build prompts in order: all features at pos 0, then pos 1, etc."""
        prompts = []
        for seq_j in range(n_seqs):
            prefix = list(sys_ids)
            for tok_k in range(T):
                prefix.extend(tok_ids_per_seq[seq_j][tok_k])
                for fi in range(n_feats):
                    prompts.append({"prompt_token_ids": prefix + suffix_ids[fi]})
        return prompts

    ordered = _build_ordered(bench_seqs, bench_feats)
    n = len(ordered)
    print(f"\n  Benchmarking vLLM ({n} prompts, token-ID mode)...")

    # Test 1: Cold vs warm
    t0 = _time.time()
    llm.generate(ordered, params)
    cold = _time.time() - t0

    t0 = _time.time()
    llm.generate(ordered, params)
    warm = _time.time() - t0

    cold_rate = n / cold
    warm_rate = n / warm
    speedup = warm_rate / cold_rate if cold_rate > 0 else 1
    print(f"    Cold: {cold_rate:.0f} p/s | Warm: {warm_rate:.0f} p/s | "
          f"Speedup: {speedup:.2f}x {'CACHED' if speedup > 1.2 else 'NO CACHE'}")

    # Test 2: Ordered vs shuffled (verifies incremental prefix caching)
    shuffled = list(ordered)
    _random.shuffle(shuffled)
    t0 = _time.time()
    llm.generate(shuffled, params)
    shuffled_time = _time.time() - t0
    shuffled_rate = n / shuffled_time

    order_benefit = warm_rate / shuffled_rate if shuffled_rate > 0 else 1
    print(f"    Ordered: {warm_rate:.0f} p/s | Shuffled: {shuffled_rate:.0f} p/s | "
          f"Order benefit: {order_benefit:.2f}x")

    # Sweep chunk sizes with full feature set
    best_rate = warm_rate
    best_chunk = min(5, N)
    candidates = [c for c in [1, 2, 5, 10] if c <= N]

    if len(candidates) > 1:
        print(f"    Chunk sweep: {candidates}")
        for chunk in candidates:
            chunk_seqs = min(chunk, bench_seqs)
            chunk_prompts = _build_ordered(chunk_seqs, bench_feats)
            t0 = _time.time()
            llm.generate(chunk_prompts, params)
            elapsed = _time.time() - t0
            rate = len(chunk_prompts) / elapsed if elapsed > 0 else 0
            print(f"      chunk={chunk:>3}: {rate:.0f} p/s")
            if rate > best_rate:
                best_rate = rate
                best_chunk = chunk

    print(f"    Selected: seq_chunk={best_chunk} ({best_rate:.0f} p/s)")
    return best_chunk


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

    # ── Pre-tokenize everything as token IDs ─────────────────────────
    # This guarantees prefix alignment: concatenating ID lists is exact,
    # unlike concatenating strings then tokenizing (BPE boundary merges).

    feat_defs = "\n".join(
        f"F{i}: {feat['description']}" for i, feat in enumerate(features)
    )
    SYS_STR = (
        "<|im_start|>system\n"
        "Feature annotator. For the last token shown, answer which "
        "feature ID matches. Answer 0 (no) or 1 (yes).\n\n"
        f"{feat_defs}<|im_end|>\n"
        "<|im_start|>user\n"
    )
    ASST_STR = "<|im_end|>\n<|im_start|>assistant\n"

    sys_ids = ann_tokenizer.encode(SYS_STR, add_special_tokens=False)
    print(f"  System prompt: {len(sys_ids)} tokens (cached L0)")

    # Pre-tokenize each GPT-2 token string individually
    print("  Pre-tokenizing sequence tokens...")
    tok_ids_per_seq = []
    for seq_j in range(N):
        seq_tok_ids = []
        for tok_k in range(T):
            piece = all_token_strs[seq_j][tok_k]
            ids = ann_tokenizer.encode(piece, add_special_tokens=False)
            seq_tok_ids.append(ids)
        tok_ids_per_seq.append(seq_tok_ids)

    # Pre-tokenize feature suffixes
    suffix_ids = []
    for fi in range(n_features):
        s = f"\nF{fi}{ASST_STR}"
        ids = ann_tokenizer.encode(s, add_special_tokens=False)
        suffix_ids.append(ids)

    print(f"  Suffix tokens: {len(suffix_ids[0])} per feature")
    del ann_tokenizer

    llm = LLM(
        model=cfg.local_annotator_model,
        dtype="auto",
        enable_prefix_caching=True,
        max_model_len=1024,
        gpu_memory_utilization=0.95,
    )
    params = SamplingParams(
        max_tokens=1,
        temperature=0,
        allowed_token_ids=[tok_0_id, tok_1_id],
    )

    annotations = torch.zeros(N, T, n_features)
    total_decisions = n_features * N * T
    completed = 0
    t_start = time.time()

    # ── Benchmark: verify caching with token IDs ─────────────────────
    seq_chunk = _benchmark_vllm_ids(
        llm, params, sys_ids, tok_ids_per_seq, suffix_ids,
        n_features, T, N,
    )

    print(f"\nAnnotating: {n_features} features x {N} sequences x {T} tokens "
          f"= {total_decisions:,} decisions "
          f"(vLLM, token-ID prefixes, chunk={seq_chunk})")

    # Convert suffix_ids to tuples for faster concatenation
    sys_tuple = tuple(sys_ids)
    suffix_tuples = [tuple(s) for s in suffix_ids]
    tok_id_tuples = [
        [tuple(tok_ids_per_seq[j][k]) for k in range(T)]
        for j in range(N)
    ]

    for seq_start in range(0, N, seq_chunk):
        seq_end = min(seq_start + seq_chunk, N)

        # Pre-build all prefix tuples for this chunk, then generate prompts
        prompts = []
        positions = []

        for seq_j in range(seq_start, seq_end):
            prefix = sys_tuple
            for tok_k in range(T):
                prefix = prefix + tok_id_tuples[seq_j][tok_k]
                for fi in range(n_features):
                    prompts.append({
                        "prompt_token_ids": list(prefix + suffix_tuples[fi])
                    })
                    positions.append((seq_j, tok_k, fi))

        outputs = llm.generate(prompts, params)

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
            f"Feature: {feat['description']}\n"
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

    # Load model for tokenization and activation extraction
    from transformer_lens import HookedTransformer

    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=cfg.device, dtype=cfg.model_dtype
    )
    model.eval()
    tokenizer = model.tokenizer

    # Tokenize corpus (or load cached)
    if cfg.tokens_path.exists():
        print(f"Loading cached tokens: {cfg.tokens_path}")
        tokens = torch.load(cfg.tokens_path, weights_only=True)
    else:
        tokens = prepare_corpus(model, cfg)
        torch.save(tokens, cfg.tokens_path)
        print(f"Saved tokens: {cfg.tokens_path}")

    # Extract activations (or load cached)
    if cfg.activations_path.exists():
        print(f"Loading cached activations: {cfg.activations_path}")
        activations = torch.load(cfg.activations_path, weights_only=True)
    else:
        activations = extract_activations(model, tokens, cfg)
        torch.save(activations, cfg.activations_path)
        print(f"Saved activations: {cfg.activations_path}")

    # Save tokenizer ref before freeing model
    tokenizer_ref = tokenizer

    # Free GPU
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Annotate (or load cached)
    if cfg.annotations_path.exists():
        print(f"Loading cached annotations: {cfg.annotations_path}")
        annotations = torch.load(cfg.annotations_path, weights_only=True)
    else:
        if cfg.use_local_annotator:
            # v2: Decomposed single-feature single-token with local model
            leaf_annotations = annotate_local(tokens, leaf_features, tokenizer_ref, cfg)
        else:
            # v1: API-based multi-feature annotation
            leaf_annotations = annotate_corpus(tokens, leaf_features, tokenizer_ref, cfg)
        # Map leaf annotations back to full feature tensor
        N_tok, T_tok = tokens.shape
        annotations = torch.zeros(N_tok, T_tok, len(features))
        for li, fi in enumerate(leaf_indices):
            annotations[:, :, fi] = leaf_annotations[:, :, li]
        # Propagate group labels (OR of children)
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

        torch.save(annotations, cfg.annotations_path)
        print(f"Saved annotations: {cfg.annotations_path}")

    return tokens, activations, annotations


if __name__ == "__main__":
    run()
