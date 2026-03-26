"""
Quick annotator validation on trivially checkable features.

Tests whether the local annotator (vLLM) can correctly label:
  - BOS token (first token in sequence)
  - Comma (token is ",")
  - Period (token is ".")
  - The word "the" (token is "the" or " the")
  - Capitalized token (starts with uppercase)

Ground truth is deterministic from the token text — no ambiguity.
Measures precision, recall, F1 per feature. If the annotator can't
get >0.8 F1 on "is this token a comma?", it's broken.

Usage:
    python -m pipeline.validate_annotator
    python -m pipeline.validate_annotator --annotator-model Qwen/Qwen3.5-9B
"""

import json
import time

import torch
from tqdm.auto import tqdm

from .config import Config


# ── Trivial features with deterministic ground truth ───────────────────────

VALIDATION_FEATURES = [
    {
        "id": "bos_token",
        "description": "Token is the very first token in the sequence (position 0)",
        "type": "leaf",
        "parent": None,
        "check": lambda tok_str, pos, all_strs: pos == 0,
    },
    {
        "id": "comma",
        "description": "Token is a comma",
        "type": "leaf",
        "parent": None,
        "check": lambda tok_str, pos, all_strs: tok_str.strip() == ",",
    },
    {
        "id": "period",
        "description": "Token is a period",
        "type": "leaf",
        "parent": None,
        "check": lambda tok_str, pos, all_strs: tok_str.strip() == ".",
    },
    {
        "id": "the_word_the",
        "description": "Token is the word 'the'",
        "type": "leaf",
        "parent": None,
        "check": lambda tok_str, pos, all_strs: tok_str.strip().lower() == "the",
    },
    {
        "id": "capitalized",
        "description": "Token starts with an uppercase letter",
        "type": "leaf",
        "parent": None,
        "check": lambda tok_str, pos, all_strs: (
            len(tok_str.strip()) > 0 and tok_str.strip()[0].isupper()
        ),
    },
]


def compute_ground_truth(all_token_strs, T, n_seqs):
    """Compute deterministic ground truth from token strings."""
    n_features = len(VALIDATION_FEATURES)
    labels = torch.zeros(n_seqs, T, n_features)

    for seq_j in range(n_seqs):
        for tok_k in range(T):
            tok_str = all_token_strs[seq_j][tok_k]
            for fi, feat in enumerate(VALIDATION_FEATURES):
                if feat["check"](tok_str, tok_k, all_token_strs[seq_j]):
                    labels[seq_j, tok_k, fi] = 1.0

    return labels


def run(cfg: Config = None):
    """Run annotator validation on trivial features."""
    if cfg is None:
        cfg = Config()

    from transformer_lens import HookedTransformer
    from datasets import load_dataset

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model + tokenize corpus
    print("Loading GPT-2 for tokenization...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device="cpu", dtype=cfg.model_dtype,
    )
    tokenizer = model.tokenizer
    del model

    n_seqs = min(50, cfg.n_sequences)  # small — just for validation
    T = cfg.seq_len

    print(f"Tokenizing {n_seqs} sequences...")
    dataset = load_dataset(cfg.corpus_dataset, split=cfg.corpus_split, streaming=True)
    sequences = []
    for example in dataset:
        text = example.get("text", "")
        if len(text.strip()) < 80:
            continue
        ids = tokenizer.encode(text)
        if len(ids) >= T:
            sequences.append(ids[:T])
        if len(sequences) >= n_seqs:
            break

    tokens = torch.tensor(sequences, dtype=torch.long)
    N = tokens.shape[0]

    # Decode to strings
    all_token_strs = []
    for i in range(N):
        strs = [tokenizer.decode([t]) for t in tokens[i].tolist()]
        all_token_strs.append(strs)

    # Ground truth
    gt = compute_ground_truth(all_token_strs, T, N)
    n_features = len(VALIDATION_FEATURES)

    print(f"\nGround truth positive counts:")
    for fi, feat in enumerate(VALIDATION_FEATURES):
        n_pos = int(gt[:, :, fi].sum().item())
        rate = n_pos / (N * T)
        print(f"  {feat['id']:<25} {n_pos:>6} ({rate:.3%})")

    # Build feature list for annotator (without the check lambda)
    features_for_annotator = [
        {"id": f["id"], "description": f["description"], "type": "leaf", "parent": None}
        for f in VALIDATION_FEATURES
    ]

    # Run LLM annotation
    from .annotate import annotate_local
    print(f"\nRunning annotator: {cfg.local_annotator_model}")
    t0 = time.time()
    llm_labels = annotate_local(tokens, features_for_annotator, tokenizer, cfg)
    elapsed = time.time() - t0

    # Compare
    print(f"\nAnnotation completed in {elapsed:.1f}s")
    print(f"  Rate: {N * T * n_features / elapsed:.0f} decisions/s")

    print(f"\n  {'Feature':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'GT+':>7} {'LLM+':>7}")
    print("  " + "-" * 63)

    gt_flat = gt.reshape(-1, n_features).numpy()
    pred_flat = llm_labels.reshape(-1, n_features).numpy()

    all_f1 = []
    for fi, feat in enumerate(VALIDATION_FEATURES):
        gt_pos = int(gt_flat[:, fi].sum())
        pred_pos = int(pred_flat[:, fi].sum())
        tp = int((gt_flat[:, fi].astype(bool) & pred_flat[:, fi].astype(bool)).sum())
        prec = tp / pred_pos if pred_pos > 0 else 0
        rec = tp / gt_pos if gt_pos > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        all_f1.append(f1)
        print(f"  {feat['id']:<25} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} "
              f"{gt_pos:>7} {pred_pos:>7}")

    mean_f1 = sum(all_f1) / len(all_f1)
    print(f"\n  Mean F1: {mean_f1:.4f}")

    if mean_f1 > 0.8:
        print("  PASS: Annotator works on trivial features.")
    elif mean_f1 > 0.5:
        print("  PARTIAL: Some features work, some don't.")
    else:
        print("  FAIL: Annotator can't label trivial features.")

    results = {
        "model": cfg.local_annotator_model,
        "n_sequences": N,
        "elapsed_seconds": round(elapsed, 1),
        "decisions_per_sec": round(N * T * n_features / elapsed, 0),
        "mean_f1": round(mean_f1, 4),
        "per_feature": {
            feat["id"]: round(all_f1[fi], 4)
            for fi, feat in enumerate(VALIDATION_FEATURES)
        },
    }
    results_path = cfg.output_dir / "annotator_validation.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    run()
