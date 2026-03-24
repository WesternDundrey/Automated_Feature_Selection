"""
IOI (Indirect Object Identification) Validation

Question 1: Does the SAE architecture and training procedure work?

  Train a supervised SAE directly on IOI with ground-truth labels as
  supervision. Compare SAE decoder columns against Makelov's mean
  dictionary vectors. Run the three-axis evaluation on both.

  If the SAE can't match or beat the mean dictionary, the training
  code is broken — independent of annotation quality.

Question 2: Does the full pipeline work (including LLM annotation)?

  Only after Q1 passes. Run LLM annotator on IOI data, compare
  annotations against ground truth, compare resulting dictionaries.

Usage:
    python -m pipeline.run --step ioi                    # Q1 only
    python -m pipeline.run --step ioi --local-annotator  # Q1 + Q2
"""

import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import (
    SupervisedSAE, compute_target_directions, train_supervised_sae,
    build_hierarchy_map, set_seed,
)

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


def generate_ioi_sentences(n_sentences, seed=42):
    """Generate IOI sentences with ground-truth attribute labels."""
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


def tokenize_and_label(sentences, tokenizer, seq_len=128):
    """Tokenize IOI sentences and create ground-truth binary labels.

    Features: indirect_object, subject_first, subject_second, abb, bab
    """
    catalog_path = Path(__file__).parent / "ioi_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    features = catalog["features"]

    feat_id_to_idx = {f["id"]: i for i, f in enumerate(features)}
    io_idx = feat_id_to_idx["name_role.indirect_object"]
    s1_idx = feat_id_to_idx["name_role.subject_first"]
    s2_idx = feat_id_to_idx["name_role.subject_second"]
    name_group_idx = feat_id_to_idx["name_role"]
    abb_idx = feat_id_to_idx["position_pattern.abb"]
    bab_idx = feat_id_to_idx["position_pattern.bab"]
    pos_group_idx = feat_id_to_idx["position_pattern"]

    N = len(sentences)
    n_features = len(features)
    all_tokens = []
    labels = torch.zeros(N, seq_len, n_features)

    for i, sent in enumerate(sentences):
        ids = tokenizer.encode(sent["text"])[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))
        all_tokens.append(ids)

        io_name = sent["io_name"]
        s_name = sent["s_name"]
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
    """Run model on IOI tokens, extract residual stream at target layer."""
    N = tokens.shape[0]
    all_resid = []
    with torch.no_grad():
        for i in tqdm(range(0, N, cfg.corpus_batch_size), desc="Extracting"):
            batch = tokens[i:i + cfg.corpus_batch_size].to(cfg.device)
            _, cache = model.run_with_cache(
                batch, names_filter=cfg.hook_point, return_type=None
            )
            all_resid.append(cache[cfg.hook_point].float().cpu())
    return torch.cat(all_resid, dim=0)


# ── Question 1: SAE training validation ───────────────────────────────────

def question1_sae_training(model, activations, labels, features, cfg):
    """Train supervised SAE on ground-truth IOI labels, compare to mean dictionary.

    This validates the training procedure independently of annotation quality.
    """
    print("\n" + "=" * 70)
    print("QUESTION 1: DOES THE SAE TRAINING WORK?")
    print("=" * 70)

    N, T, d_model = activations.shape
    n_features = len(features)

    # ── Compute mean dictionary (Makelov's ground truth) ──────────────
    print("\nComputing mean dictionary from ground-truth labels...")
    x_flat = activations.reshape(-1, d_model)
    y_flat = labels.reshape(-1, n_features)

    mean_dirs, mean_norms, counts = compute_target_directions(
        x_flat, y_flat, n_features,
    )

    print(f"\n  {'Feature':<35} {'Norm':>8} {'Count':>8}")
    print("  " + "-" * 53)
    for i, feat in enumerate(features):
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {mean_norms[i].item():>8.4f} "
              f"{counts[i].item():>8.0f}{tag}")

    # Mean dictionary R²
    x_centered = x_flat - x_flat.mean(0)
    total_var = (x_centered ** 2).sum().item()
    projections = x_flat @ mean_dirs.T
    recon = (projections * y_flat) @ mean_dirs
    mean_dict_r2 = 1 - ((x_centered - recon) ** 2).sum().item() / total_var
    print(f"\n  Mean dictionary R²: {mean_dict_r2:.4f}")

    # ── Train supervised SAE on ground-truth labels ──────────────────
    print("\n" + "-" * 50)
    print("Training supervised SAE on IOI ground-truth labels...")
    print("-" * 50)

    sae = train_supervised_sae(
        activations, labels, features, cfg, save_checkpoint=True,
    )

    # ── Compare: SAE decoder columns vs mean dictionary ──────────────
    print("\n" + "-" * 50)
    print("COMPARISON: SAE decoder vs mean dictionary")
    print("-" * 50)

    sae_dec = sae.decoder.weight.cpu()  # (d_model, n_total)
    sae_cols = sae_dec[:, :n_features]  # (d_model, n_features)

    # Cosine similarity: SAE decoder column vs mean dictionary direction
    print(f"\n  {'Feature':<35} {'Cos':>8} {'SAE norm':>9} {'Mean norm':>10}")
    print("  " + "-" * 64)

    cosines = []
    for k, feat in enumerate(features):
        if mean_norms[k] < 1e-6:
            cos = 0.0
        else:
            # Decoder columns are unit-normalized, mean_dirs are unit-normalized
            cos = (sae_cols[:, k] @ mean_dirs[k]).item()
        sae_norm = sae_cols[:, k].norm().item()
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {cos:>8.4f} {sae_norm:>9.4f} "
              f"{mean_norms[k].item():>10.4f}{tag}")
        if feat["type"] == "leaf":
            cosines.append(cos)

    mean_cos = sum(cosines) / len(cosines) if cosines else 0
    high = sum(1 for c in cosines if c > 0.9)
    low = sum(1 for c in cosines if c < 0.5)

    print(f"\n  Mean cosine (SAE vs mean dict): {mean_cos:.4f}")
    print(f"  Leaves with cosine > 0.9: {high}/{len(cosines)}")
    print(f"  Leaves with cosine < 0.5: {low}/{len(cosines)}")

    # SAE reconstruction R²
    sae.eval().to(cfg.device)
    all_recon = []
    with torch.no_grad():
        for i in range(0, x_flat.shape[0], cfg.batch_size):
            xb = x_flat[i:i + cfg.batch_size].to(cfg.device)
            recon_b, _, _, _ = sae(xb)
            all_recon.append(recon_b.cpu())
    sae_recon = torch.cat(all_recon, dim=0)
    sae_r2 = 1 - F.mse_loss(sae_recon, x_flat).item() / (total_var / x_flat.shape[0])
    sae.cpu()

    print(f"\n  Mean dictionary R²: {mean_dict_r2:.4f}")
    print(f"  Supervised SAE R²:  {sae_r2:.4f}")
    if sae_r2 > mean_dict_r2:
        print(f"  SAE beats mean dict by {sae_r2 - mean_dict_r2:.4f} "
              f"(unsupervised latents helping reconstruction)")

    # Verdict
    if mean_cos > 0.9:
        print(f"\n  PASS: SAE decoder columns converge to mean dictionary (cos={mean_cos:.3f})")
    elif mean_cos > 0.5:
        print(f"\n  PARTIAL: Some convergence but not strong (cos={mean_cos:.3f})")
    else:
        print(f"\n  FAIL: SAE decoder columns don't match mean dictionary (cos={mean_cos:.3f})")

    return {
        "mean_dict_r2": round(mean_dict_r2, 4),
        "sae_r2": round(sae_r2, 4),
        "mean_cosine": round(mean_cos, 4),
        "per_feature_cosines": {
            feat["id"]: round(cosines[i], 4) if i < len(cosines) else None
            for i, feat in enumerate(f for f in features if f["type"] == "leaf")
        },
        "high_cosine_count": high,
        "low_cosine_count": low,
    }, sae


# ── Question 2: LLM annotation validation ─────────────────────────────────

def question2_annotation(activations, tokens, labels_gt, features, cfg):
    """Run LLM annotator on IOI data, compare against ground truth."""
    print("\n" + "=" * 70)
    print("QUESTION 2: DOES THE LLM ANNOTATION WORK?")
    print("=" * 70)

    from .annotate import annotate_local, propagate_group_labels
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    leaf_features = [f for f in features if f["type"] == "leaf"]
    leaf_indices = [i for i, f in enumerate(features) if f["type"] == "leaf"]

    leaf_annotations = annotate_local(tokens, leaf_features, tokenizer, cfg)

    N, T = tokens.shape
    labels_llm = torch.zeros(N, T, len(features))
    for li, fi in enumerate(leaf_indices):
        labels_llm[:, :, fi] = leaf_annotations[:, :, li]
    labels_llm = propagate_group_labels(labels_llm, features)

    # Per-feature precision/recall/F1
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

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    print(f"\n  Mean leaf F1 (annotator vs ground truth): {mean_f1:.4f}")

    # Compare dictionaries
    N, T, d_model = activations.shape
    n_features = len(features)
    x_flat = activations.reshape(-1, d_model)
    y_gt = labels_gt.reshape(-1, n_features)
    y_llm = labels_llm.reshape(-1, n_features)

    dirs_gt, _, _ = compute_target_directions(x_flat, y_gt, n_features)
    dirs_llm, _, _ = compute_target_directions(x_flat, y_llm, n_features)

    print(f"\n  {'Feature':<35} {'Cos(GT,LLM)':>12}")
    print("  " + "-" * 49)
    dict_cosines = []
    for k, feat in enumerate(features):
        cos = (dirs_gt[k] @ dirs_llm[k]).item() if dirs_gt[k].norm() > 1e-6 and dirs_llm[k].norm() > 1e-6 else 0
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {cos:>12.4f}{tag}")
        if feat["type"] == "leaf":
            dict_cosines.append(cos)

    mean_dict_cos = sum(dict_cosines) / len(dict_cosines) if dict_cosines else 0
    print(f"\n  Mean dictionary cosine (GT vs LLM): {mean_dict_cos:.4f}")

    return {
        "annotator_mean_f1": round(mean_f1, 4),
        "dictionary_cosine": round(mean_dict_cos, 4),
    }


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

    n_ioi = min(cfg.n_sequences, 2000)
    print(f"\nGenerating {n_ioi} IOI sentences...")
    sentences = generate_ioi_sentences(n_ioi, seed=cfg.seed)
    tokens, labels, features = tokenize_and_label(sentences, tokenizer, cfg.seq_len)

    for i, feat in enumerate(features):
        n_pos = int(labels[:, :, i].sum().item())
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<35} {n_pos:>6} positives{tag}")

    print(f"\nExtracting activations at {cfg.hook_point}...")
    activations = extract_activations(model, tokens, cfg)
    print(f"  Activations: {activations.shape}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Question 1: SAE training validation
    q1_results, sae = question1_sae_training(
        None, activations, labels, features, cfg,
    )

    results = {"question1": q1_results}

    # Question 2: LLM annotation (only if --local-annotator)
    if not skip_annotator:
        q2_results = question2_annotation(
            activations, tokens, labels, features, cfg,
        )
        results["question2"] = q2_results

    results_path = cfg.output_dir / "ioi_validation.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-annotator", action="store_true")
    parser.add_argument("--n_sequences", type=int, default=500)
    args = parser.parse_args()
    cfg = Config(n_sequences=args.n_sequences)
    run(cfg, skip_annotator=args.skip_annotator)
