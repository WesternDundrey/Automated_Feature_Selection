"""
annotate.py — Programmatically label each token position for each feature.

Annotation rule: decode token to string, strip whitespace, lowercase.
If the result exactly matches a word in a feature's word list, the feature is active.
Group features are derived as the OR of their children's labels.

Outputs:
  data/labels.npy  bool  (N_SEQS, SEQ_LEN, N_FEATURES)

Requires: transformer_lens (tokenizer only), features.json, data/tokens.npy
"""

import json
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer

SAVE_DIR = Path("data")


def build_vocab_lookup(tokenizer: GPT2Tokenizer) -> dict[int, str]:
    """Map token_id -> lowercase stripped string for the full GPT-2 vocab."""
    vocab_size = tokenizer.vocab_size
    lookup = {}
    for tok_id in range(vocab_size):
        decoded = tokenizer.decode([tok_id]).strip().lower()
        lookup[tok_id] = decoded
    return lookup


def main():
    with open("features.json") as f:
        catalog = json.load(f)
    features = catalog["features"]
    feature_ids = [feat["id"] for feat in features]
    n_features = len(features)
    feature_id_to_idx = {feat["id"]: i for i, feat in enumerate(features)}

    # word -> list of feature indices that activate on this word
    word_to_feature_idxs: dict[str, list[int]] = {}
    for idx, feat in enumerate(features):
        for word in feat.get("words", []):
            word = word.lower()
            word_to_feature_idxs.setdefault(word, []).append(idx)

    print("Building GPT-2 vocab lookup...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_lookup = build_vocab_lookup(tokenizer)

    # Pre-compute: token_id -> list of feature indices it activates
    # (only for token_ids that match at least one word)
    token_to_active_features: dict[int, list[int]] = {}
    for tok_id, tok_str in vocab_lookup.items():
        if tok_str in word_to_feature_idxs:
            token_to_active_features[tok_id] = word_to_feature_idxs[tok_str]

    print(f"Found {len(token_to_active_features)} token IDs that match at least one feature word")

    tokens = np.load(SAVE_DIR / "tokens.npy")  # (N, SEQ_LEN) int32
    N, SEQ_LEN = tokens.shape
    print(f"Annotating {N} sequences x {SEQ_LEN} tokens = {N * SEQ_LEN:,} positions...")

    labels = np.zeros((N, SEQ_LEN, n_features), dtype=bool)

    for n in range(N):
        for t in range(SEQ_LEN):
            tok_id = int(tokens[n, t])
            if tok_id in token_to_active_features:
                for feat_idx in token_to_active_features[tok_id]:
                    labels[n, t, feat_idx] = True

    # Propagate group labels: group = OR of children
    for idx, feat in enumerate(features):
        if feat["type"] == "group":
            child_idxs = [
                feature_id_to_idx[f["id"]]
                for f in features
                if f.get("parent") == feat["id"]
            ]
            if child_idxs:
                labels[:, :, idx] = labels[:, :, child_idxs].any(axis=-1)

    np.save(SAVE_DIR / "labels.npy", labels)

    print(f"\nLabel statistics ({N * SEQ_LEN:,} total positions):")
    print(f"  {'Feature':<32} {'Positives':>10} {'Rate':>8}")
    print("  " + "-" * 54)
    for idx, feat in enumerate(features):
        count = int(labels[:, :, idx].sum())
        rate = count / (N * SEQ_LEN)
        tag = "  [group]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<32} {count:>10,} {rate:>7.4%}{tag}")

    print(f"\nSaved data/labels.npy  shape={labels.shape}  {labels.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
