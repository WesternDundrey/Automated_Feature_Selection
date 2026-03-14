"""
extract.py — Download wikitext-2, tokenize, extract GPT-2 layer 8 residual stream.

PURPOSE: Data generation for the toy validation pipeline (cheap, CPU-feasible).
Uses GPT-2 small and programmatically-annotated features (colors, days, months)
to validate the supervised training loop before scaling to the primary experiment.
See rabbit_habit_supervised_sae.ipynb for circuit-targeted training on Gemma-2-2B-IT.

Outputs:
  data/tokens.npy       int32  (N_SEQS, SEQ_LEN)
  data/activations.npy  float16 (N_SEQS, SEQ_LEN, D_MODEL)

Requires: transformer_lens, datasets
"""

import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformer_lens import HookedTransformer

LAYER = 8
SEQ_LEN = 256
N_SEQS = 500
BATCH_SIZE = 16
SAVE_DIR = Path("data")


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    print("Downloading wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [row["text"] for row in dataset if len(row["text"].strip()) > 50]

    print(f"Tokenizing {len(texts)} documents and chunking into {SEQ_LEN}-token sequences...")
    tokenizer = model.tokenizer
    all_token_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_token_ids.extend(ids)

    # Non-overlapping chunks of SEQ_LEN
    sequences = []
    for start in range(0, len(all_token_ids) - SEQ_LEN, SEQ_LEN):
        sequences.append(all_token_ids[start : start + SEQ_LEN])
        if len(sequences) >= N_SEQS:
            break

    n = len(sequences)
    print(f"Collected {n} sequences ({n * SEQ_LEN:,} tokens total)")
    if n < N_SEQS:
        print(f"  Note: only got {n} sequences (corpus has {len(all_token_ids):,} tokens)")

    tokens_tensor = torch.tensor(sequences, dtype=torch.long)  # (N, SEQ_LEN)

    hook_name = f"blocks.{LAYER}.hook_resid_post"
    print(f"Extracting residual stream at layer {LAYER}...")

    activations_list = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = tokens_tensor[i : i + BATCH_SIZE].to(device)
            _, cache = model.run_with_cache(batch, names_filter=hook_name)
            act = cache[hook_name]  # (batch, SEQ_LEN, D_MODEL)
            activations_list.append(act.cpu().to(torch.float16).numpy())
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  {min(i + BATCH_SIZE, n)}/{n} sequences")

    activations_np = np.concatenate(activations_list, axis=0)  # (N, SEQ_LEN, D_MODEL)
    tokens_np = tokens_tensor.numpy().astype(np.int32)

    np.save(SAVE_DIR / "tokens.npy", tokens_np)
    np.save(SAVE_DIR / "activations.npy", activations_np)

    print(f"\nSaved:")
    print(f"  tokens:      {tokens_np.shape}  {tokens_np.nbytes / 1e6:.1f} MB")
    print(f"  activations: {activations_np.shape}  {activations_np.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
