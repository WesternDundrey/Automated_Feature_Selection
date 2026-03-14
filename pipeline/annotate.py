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
import re
import time
from pathlib import Path

import anthropic
import torch
from tqdm.auto import tqdm

from .config import Config


# ── Corpus preparation ──────────────────────────────────────────────────────

def prepare_corpus(model, cfg: Config) -> torch.Tensor:
    """Load corpus, tokenize with transformer_lens, return (N, seq_len) token tensor."""
    from datasets import load_dataset

    tokenizer = model.tokenizer
    print(f"Loading corpus: {cfg.corpus_dataset} [{cfg.corpus_split}]")
    dataset = load_dataset(cfg.corpus_dataset, split=cfg.corpus_split, streaming=True)

    sequences = []
    for example in tqdm(dataset, desc="Tokenizing", total=cfg.n_sequences):
        text = example.get("text", "")
        if len(text.strip()) < 80:
            continue
        ids = tokenizer.encode(text)
        if len(ids) >= cfg.seq_len:
            sequences.append(ids[: cfg.seq_len])
        if len(sequences) >= cfg.n_sequences:
            break

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
    """Extract the first JSON object from text, handling nested braces."""
    # Find the first '{' and match braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def annotate_sequence_async(
    client: anthropic.AsyncAnthropic,
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
                    response = await client.messages.create(
                        model=cfg.annotation_model,
                        max_tokens=500,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
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
            # All retries exhausted → labels stay zero (safe default)

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
    all_labels = torch.zeros(N, T, n_features)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(cfg.max_annotation_concurrency)

    # Decode all sequences to strings
    print("Decoding tokens to strings...")
    all_token_strs = []
    for i in range(N):
        strs = [tokenizer.decode([t.item()]) for t in tokens[i]]
        all_token_strs.append(strs)

    print(f"Annotating {N} sequences x {n_features} features "
          f"with {cfg.annotation_model}...")

    t0 = time.time()
    completed = 0

    # Process in waves to show progress
    wave_size = 100
    for wave_start in range(0, N, wave_size):
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
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (N - completed) / rate if rate > 0 else 0
        total_pos = int(all_labels[:completed].sum().item())
        print(f"  {completed}/{N} sequences  "
              f"({rate:.1f} seq/s, ETA {eta:.0f}s, {total_pos} positives)")

    elapsed = time.time() - t0
    total_pos = int(all_labels.sum().item())
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


# ── Propagate group labels ──────────────────────────────────────────────────

def propagate_group_labels(annotations: torch.Tensor, features: list[dict]) -> torch.Tensor:
    """Set group feature labels = OR of children's labels."""
    feat_id_to_idx = {f["id"]: i for i, f in enumerate(features)}

    for idx, feat in enumerate(features):
        if feat["type"] == "group":
            child_idxs = [
                feat_id_to_idx[f["id"]]
                for f in features
                if f.get("parent") == feat["id"]
            ]
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
        annotations = annotate_corpus(tokens, features, tokenizer_ref, cfg)
        # Propagate group labels
        annotations = propagate_group_labels(annotations, features)
        torch.save(annotations, cfg.annotations_path)
        print(f"Saved annotations: {cfg.annotations_path}")

    return tokens, activations, annotations


if __name__ == "__main__":
    run()
