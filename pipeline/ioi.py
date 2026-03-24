"""
IOI (Indirect Object Identification) Validation

Validates the supervised SAE pipeline against Makelov et al. 2024's
ground truth. Three stages:

  1. Dictionary math:  Generate IOI data, compute mean dictionary from
     perfect labels, check reconstruction quality.
  2. Annotator quality: Run LLM annotator on IOI data, compare labels
     against ground truth (precision/recall per feature).
  3. End-to-end:  Compare dictionary vectors from LLM labels vs ground
     truth labels (cosine similarity). If they match, the pipeline works.

Usage:
    python -m pipeline.ioi                      # full validation
    python -m pipeline.ioi --skip-annotator     # dictionary math only
"""

import json
import random
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import compute_target_directions

# ── IOI sentence generation ────────────────────────────────────────────────

NAMES = [
    "Mary", "John", "Alice", "Bob", "Emma", "James", "Sarah", "David",
    "Lisa", "Michael", "Anna", "Tom", "Kate", "Mark", "Julia", "Peter",
    "Rachel", "Chris", "Laura", "Daniel",
]

TEMPLATES = [
    "Then, {name1} and {name2} had a long and really crazy argument. Afterwards, {s2} said to",
    "Then, {name1} and {name2} had lots of fun at the park. Afterwards, {s2} gave a present to",
    "Then, {name1} and {name2} were working at the office. {s2} decided to give a letter to",
]


def generate_ioi_sentences(n_sentences: int, seed: int = 42):
    """Generate IOI sentences with ground-truth attribute labels.

    Returns list of dicts:
        {"text": str, "io_name": str, "s_name": str, "pattern": "ABB"|"BAB",
         "io_positions": [int, ...], "s1_positions": [int, ...], "s2_positions": [int, ...]}
    """
    rng = random.Random(seed)
    name_pairs = [(a, b) for a, b in product(NAMES, NAMES) if a != b]
    sentences = []

    for _ in range(n_sentences):
        io_name, s_name = rng.choice(name_pairs)
        template = rng.choice(TEMPLATES)
        pattern = rng.choice(["ABB", "BAB"])

        if pattern == "ABB":
            name1, name2 = io_name, s_name
        else:
            name1, name2 = s_name, io_name

        text = template.format(name1=name1, name2=name2, s2=s_name)
        sentences.append({
            "text": text,
            "io_name": io_name,
            "s_name": s_name,
            "pattern": pattern,
        })

    return sentences


def tokenize_and_label(sentences, tokenizer, seq_len: int = 128):
    """Tokenize IOI sentences and create ground-truth binary labels.

    Returns:
        tokens:      (N, seq_len) int64
        labels:      (N, seq_len, n_features) float32
        features:    list of feature dicts (matching ioi_catalog.json order)
    """
    catalog_path = Path(__file__).parent / "ioi_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    features = catalog["features"]
    leaf_features = [f for f in features if f["type"] == "leaf"]
    n_features = len(features)

    # Feature index mapping
    feat_id_to_idx = {f["id"]: i for i, f in enumerate(features)}
    io_idx = feat_id_to_idx["name_role.indirect_object"]
    s1_idx = feat_id_to_idx["name_role.subject_first"]
    s2_idx = feat_id_to_idx["name_role.subject_second"]
    name_group_idx = feat_id_to_idx["name_role"]
    abb_idx = feat_id_to_idx["position_pattern.abb"]
    bab_idx = feat_id_to_idx["position_pattern.bab"]
    pos_group_idx = feat_id_to_idx["position_pattern"]

    N = len(sentences)
    all_tokens = []
    labels = torch.zeros(N, seq_len, n_features)

    for i, sent in enumerate(sentences):
        ids = tokenizer.encode(sent["text"])[:seq_len]
        # Pad if shorter
        if len(ids) < seq_len:
            ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))
        all_tokens.append(ids)

        # Decode each token to find name positions
        io_name = sent["io_name"]
        s_name = sent["s_name"]

        # Find which token positions correspond to each name
        s_count = 0
        for pos in range(len(ids)):
            tok_str = tokenizer.decode([ids[pos]]).strip()

            if tok_str == io_name:
                labels[i, pos, io_idx] = 1.0
                labels[i, pos, name_group_idx] = 1.0

            if tok_str == s_name:
                s_count += 1
                if s_count == 1:
                    labels[i, pos, s1_idx] = 1.0
                elif s_count == 2:
                    labels[i, pos, s2_idx] = 1.0
                labels[i, pos, name_group_idx] = 1.0

        # Sequence-level: position pattern (all tokens get the same label)
        if sent["pattern"] == "ABB":
            labels[i, :, abb_idx] = 1.0
            labels[i, :, pos_group_idx] = 1.0
        else:
            labels[i, :, bab_idx] = 1.0
            labels[i, :, pos_group_idx] = 1.0

    tokens = torch.tensor(all_tokens, dtype=torch.long)
    return tokens, labels, features


# ── Activation extraction ──────────────────────────────────────────────────

def extract_activations(model, tokens, cfg):
    """Run GPT-2 on IOI tokens, extract residual stream at target layer."""
    N = tokens.shape[0]
    all_resid = []

    with torch.no_grad():
        for i in tqdm(range(0, N, cfg.corpus_batch_size), desc="Extracting activations"):
            batch = tokens[i:i + cfg.corpus_batch_size].to(cfg.device)
            _, cache = model.run_with_cache(
                batch, names_filter=cfg.hook_point, return_type=None
            )
            all_resid.append(cache[cfg.hook_point].float().cpu())

    return torch.cat(all_resid, dim=0)


# ── Validation steps ───────────────────────────────────────────────────────

def validate_dictionary_math(activations, labels, features):
    """Step 1: Compute mean dictionary from ground-truth labels, report quality."""
    print("\n" + "=" * 70)
    print("STEP 1: DICTIONARY MATH VALIDATION")
    print("=" * 70)

    N, T, d_model = activations.shape
    n_features = labels.shape[-1]

    x_flat = activations.reshape(-1, d_model)
    y_flat = labels.reshape(-1, n_features)

    target_dirs, raw_norms, counts = compute_target_directions(
        x_flat, y_flat, n_features,
    )

    print(f"\n  {'Feature':<35} {'Norm':>8} {'Count':>8}")
    print("  " + "-" * 53)
    for i, feat in enumerate(features):
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {raw_norms[i].item():>8.4f} "
              f"{counts[i].item():>8.0f}{tag}")

    # Pairwise cosine similarity between target directions
    valid = raw_norms > 1e-6
    if valid.sum() > 1:
        valid_dirs = target_dirs[valid]
        pairwise = valid_dirs @ valid_dirs.T
        pairwise.fill_diagonal_(0)
        print(f"\n  Max pairwise cosine: {pairwise.max().item():.4f}")
        print(f"  Mean pairwise cosine: {pairwise.sum().item() / (valid.sum() * (valid.sum() - 1)):.4f}")

    # Reconstruction: how much variance do the dictionary vectors explain?
    x_centered = x_flat - x_flat.mean(0)
    total_var = (x_centered ** 2).sum().item()

    # Project onto dictionary
    projections = x_flat @ target_dirs.T  # (N*T, n_features)
    recon = (projections * y_flat) @ target_dirs  # (N*T, d_model)
    recon_err = ((x_centered - recon) ** 2).sum().item()
    r2 = 1 - recon_err / total_var

    print(f"\n  Dictionary R² (ground truth): {r2:.4f}")
    print(f"  (fraction of centered activation variance explained by"
          f" projecting labeled positions onto their target directions)")

    return target_dirs, raw_norms, counts


def validate_annotator(activations, tokens, labels_gt, features, cfg):
    """Step 2: Run LLM annotator, compare against ground truth."""
    print("\n" + "=" * 70)
    print("STEP 2: ANNOTATOR QUALITY VALIDATION")
    print("=" * 70)

    from .annotate import annotate_local, propagate_group_labels

    # Get base model tokenizer for decoding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    leaf_features = [f for f in features if f["type"] == "leaf"]
    leaf_indices = [i for i, f in enumerate(features) if f["type"] == "leaf"]

    # Run LLM annotation
    leaf_annotations = annotate_local(tokens, leaf_features, tokenizer, cfg)

    # Map back to full feature tensor
    N, T = tokens.shape
    labels_llm = torch.zeros(N, T, len(features))
    for li, fi in enumerate(leaf_indices):
        labels_llm[:, :, fi] = leaf_annotations[:, :, li]
    labels_llm = propagate_group_labels(labels_llm, features)

    # Compare against ground truth
    print(f"\n  {'Feature':<35} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'GT+':>7} {'LLM+':>7}")
    print("  " + "-" * 73)

    gt = labels_gt.reshape(-1, len(features)).numpy()
    pred = labels_llm.reshape(-1, len(features)).numpy()

    f1_scores = []
    for k, feat in enumerate(features):
        gt_pos = int(gt[:, k].sum())
        pred_pos = int(pred[:, k].sum())

        tp = int((gt[:, k].astype(bool) & pred[:, k].astype(bool)).sum())
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / gt_pos if gt_pos > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f} "
              f"{gt_pos:>7} {pred_pos:>7}{tag}")

        if feat["type"] == "leaf":
            f1_scores.append(f1)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(f"\n  Mean leaf F1 (annotator vs ground truth): {mean_f1:.4f}")

    return labels_llm, mean_f1


def validate_end_to_end(activations, labels_gt, labels_llm, features):
    """Step 3: Compare dictionary vectors from LLM labels vs ground truth."""
    print("\n" + "=" * 70)
    print("STEP 3: END-TO-END DICTIONARY COMPARISON")
    print("=" * 70)

    N, T, d_model = activations.shape
    n_features = len(features)

    x_flat = activations.reshape(-1, d_model)
    y_gt = labels_gt.reshape(-1, n_features)
    y_llm = labels_llm.reshape(-1, n_features)

    dirs_gt, norms_gt, _ = compute_target_directions(x_flat, y_gt, n_features)
    dirs_llm, norms_llm, _ = compute_target_directions(x_flat, y_llm, n_features)

    print(f"\n  {'Feature':<35} {'Cos(GT,LLM)':>12} {'Norm GT':>9} {'Norm LLM':>9}")
    print("  " + "-" * 67)

    cosines = []
    for k, feat in enumerate(features):
        if norms_gt[k] < 1e-6 or norms_llm[k] < 1e-6:
            cos = 0.0
        else:
            cos = (dirs_gt[k] @ dirs_llm[k]).item()

        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {cos:>12.4f} {norms_gt[k].item():>9.4f} "
              f"{norms_llm[k].item():>9.4f}{tag}")

        if feat["type"] == "leaf":
            cosines.append(cos)

    mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
    high = sum(1 for c in cosines if c > 0.8)
    print(f"\n  Mean cosine (GT vs LLM dictionary): {mean_cos:.4f}")
    print(f"  Features with cosine > 0.8: {high}/{len(cosines)}")

    if mean_cos > 0.8:
        print("\n  PASS: LLM-annotated dictionary closely matches ground truth.")
    elif mean_cos > 0.5:
        print("\n  PARTIAL: Some features match, annotation quality is the bottleneck.")
    else:
        print("\n  FAIL: LLM annotations produce different dictionary vectors.")

    return mean_cos


# ── Main entry point ───────────────────────────────────────────────────────

def run(cfg: Config = None, skip_annotator: bool = False):
    """Run IOI validation."""
    if cfg is None:
        cfg = Config()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    from transformer_lens import HookedTransformer

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(
        cfg.model_name, device=cfg.device, dtype=cfg.model_dtype,
    )
    model.eval()
    tokenizer = model.tokenizer

    # Generate IOI data
    n_ioi = min(cfg.n_sequences, 2000)  # IOI doesn't need huge N
    print(f"\nGenerating {n_ioi} IOI sentences...")
    sentences = generate_ioi_sentences(n_ioi, seed=cfg.seed)
    tokens, labels_gt, features = tokenize_and_label(
        sentences, tokenizer, seq_len=cfg.seq_len,
    )

    # Count ground truth positives
    for i, feat in enumerate(features):
        n_pos = int(labels_gt[:, :, i].sum().item())
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {n_pos:>6} positives{tag}")

    # Extract activations
    print(f"\nExtracting activations at {cfg.hook_point}...")
    activations = extract_activations(model, tokens, cfg)
    print(f"  Activations: {activations.shape}")

    # Free GPU
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 1: Dictionary math
    target_dirs, raw_norms, counts = validate_dictionary_math(
        activations, labels_gt, features,
    )

    # Save for comparison
    torch.save(tokens, cfg.output_dir / "ioi_tokens.pt")
    torch.save(activations, cfg.output_dir / "ioi_activations.pt")
    torch.save(labels_gt, cfg.output_dir / "ioi_labels_groundtruth.pt")

    if skip_annotator:
        print("\n  Skipping annotator validation (--skip-annotator)")
        return

    # Step 2: Annotator quality
    labels_llm, ann_f1 = validate_annotator(
        activations, tokens, labels_gt, features, cfg,
    )
    torch.save(labels_llm, cfg.output_dir / "ioi_labels_llm.pt")

    # Step 3: End-to-end dictionary comparison
    mean_cos = validate_end_to_end(
        activations, labels_gt, labels_llm, features,
    )

    # Save results
    results = {
        "annotator_mean_f1": round(ann_f1, 4),
        "dictionary_cosine": round(mean_cos, 4),
        "n_sentences": n_ioi,
        "model": cfg.model_name,
        "layer": cfg.target_layer,
    }
    results_path = cfg.output_dir / "ioi_validation.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IOI Validation")
    parser.add_argument("--skip-annotator", action="store_true",
                        help="Skip LLM annotator validation (dictionary math only)")
    parser.add_argument("--n_sequences", type=int, default=500)
    args = parser.parse_args()

    cfg = Config(n_sequences=args.n_sequences)
    run(cfg, skip_annotator=args.skip_annotator)
