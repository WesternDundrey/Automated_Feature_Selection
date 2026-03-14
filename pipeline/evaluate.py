"""
Step 3 — Evaluate Supervised SAE

Reports on held-out test data:
  1. Reconstruction quality (MSE, R^2)
  2. Per-feature precision / recall / F1 / AUROC
  3. Supervised vs unsupervised-best-match comparison
  4. Hierarchy consistency

Outputs:
    pipeline_data/evaluation.json

Usage:
    python -m pipeline.evaluate
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE, build_hierarchy_map, set_seed


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn dependency using the trapezoidal rule."""
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    # Sort by score descending
    desc = np.argsort(-scores)
    y_sorted = y_true[desc]

    # Cumulative TP and FP rates
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0, 0) and compute area under curve
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr))


def evaluate(cfg: Config = None):
    """Evaluate the trained supervised SAE on held-out data."""
    if cfg is None:
        cfg = Config()

    # Load model
    model_cfg = torch.load(cfg.checkpoint_config_path, map_location="cpu", weights_only=True)
    sae = SupervisedSAE(
        model_cfg["d_model"],
        model_cfg["n_supervised"],
        model_cfg["n_unsupervised"],
        n_lista_steps=model_cfg.get("n_lista_steps", 0),
    )
    sae.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae.eval().to(cfg.device)

    # Load data
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    N, T, d_model = activations.shape
    n_features = annotations.shape[-1]

    # Flatten and split to test set
    x_flat = activations.reshape(-1, d_model)
    y_flat = annotations.reshape(-1, n_features)
    n_total = x_flat.shape[0]

    # Load split indices from disk (saved by train.py) to avoid RNG coupling
    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        # Fallback: regenerate via RNG (backward compatibility)
        print("WARNING: split_indices.pt not found, regenerating via RNG")
        set_seed(cfg.seed)
        perm = torch.randperm(n_total)

    split_idx = int(cfg.train_fraction * n_total)
    test_idx = perm[split_idx:]
    x_test = x_flat[test_idx]
    y_test = y_flat[test_idx]

    print(f"Evaluating on {x_test.shape[0]:,} test vectors")

    # Forward pass in batches
    all_recon = []
    all_sup_pre = []
    all_sup_acts = []
    all_all_acts = []

    with torch.no_grad():
        for i in range(0, x_test.shape[0], cfg.batch_size):
            x_b = x_test[i : i + cfg.batch_size].to(cfg.device)
            recon, sup_pre, sup_acts, all_acts = sae(x_b)
            all_recon.append(recon.cpu())
            all_sup_pre.append(sup_pre.cpu())
            all_sup_acts.append(sup_acts.cpu())
            all_all_acts.append(all_acts.cpu())

    recon = torch.cat(all_recon)
    sup_pre = torch.cat(all_sup_pre)
    sup_acts = torch.cat(all_sup_acts)
    all_acts = torch.cat(all_all_acts)

    # ── 1. Reconstruction ────────────────────────────────────────────────
    mse = F.mse_loss(recon, x_test).item()
    baseline_mse = F.mse_loss(
        x_test.mean(0, keepdim=True).expand_as(x_test), x_test
    ).item()
    r2 = 1.0 - mse / baseline_mse

    print("\n" + "=" * 70)
    print("RECONSTRUCTION")
    print(f"  SAE MSE:              {mse:.6f}")
    print(f"  Baseline (mean) MSE:  {baseline_mse:.6f}")
    print(f"  R^2:                  {r2:.4f}")

    # ── 2. Per-feature classification ────────────────────────────────────
    gt = y_test.numpy().astype(bool)
    preds = sup_pre.numpy() > 0  # threshold at sigmoid=0.5
    scores = sup_pre.numpy()     # raw logits for AUROC

    print("\n" + "=" * 70)
    print("PER-FEATURE CLASSIFICATION (threshold: pre_act > 0)")
    print(f"  {'Feature':<36} {'P':>6} {'R':>6} {'F1':>6} {'AUROC':>7} {'Pos':>8}")
    print("  " + "-" * 73)

    feature_results = []
    f1_scores = []
    auroc_scores = []

    for k, feat in enumerate(features):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            feature_results.append({
                "id": feat["id"], "type": feat["type"],
                "n_positives": 0, "precision": None, "recall": None,
                "f1": None, "auroc": None,
            })
            print(f"  {feat['id']:<36} {'--':>6} {'--':>6} {'--':>6} {'--':>7} {n_pos:>8}")
            continue

        p, r, f1 = precision_recall_f1(gt[:, k], preds[:, k])
        auc = auroc(gt[:, k], scores[:, k])
        f1_scores.append(f1)
        if not np.isnan(auc):
            auroc_scores.append(auc)
        tag = " [group]" if feat["type"] == "group" else ""
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "--"
        print(f"  {feat['id']:<36} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {auc_str:>7} {n_pos:>8}{tag}")

        feature_results.append({
            "id": feat["id"], "type": feat["type"],
            "n_positives": n_pos, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f1, 4),
            "auroc": round(auc, 4) if not np.isnan(auc) else None,
        })

    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    mean_auroc = float(np.mean(auroc_scores)) if auroc_scores else 0.0
    print(f"\n  Mean F1 (features with positives):    {mean_f1:.3f}")
    print(f"  Mean AUROC (features with positives): {mean_auroc:.3f}")

    # ── 3. Sparsity ─────────────────────────────────────────────────────
    l0_supervised = (sup_acts > 0).float().sum(dim=-1).mean().item()
    l0_total = (all_acts > 0).float().sum(dim=-1).mean().item()
    print(f"\n  L0 (supervised latents): {l0_supervised:.1f}")
    print(f"  L0 (all latents):        {l0_total:.1f}")

    # ── 4. Hierarchy consistency ────────────────────────────────────────
    hier_map = build_hierarchy_map(features)

    print("\n" + "=" * 70)
    print("HIERARCHY CONSISTENCY (parent_act >= child_act when child active)")
    print(f"  {'Parent -> Child':<50} {'Consistent':>12}")
    print("  " + "-" * 64)

    acts_np = sup_acts.numpy()
    hier_results = []

    for parent_idx, child_idxs in hier_map.items():
        parent_id = features[parent_idx]["id"]
        for child_idx in child_idxs:
            child_id = features[child_idx]["id"]
            child_active = acts_np[:, child_idx] > 0
            if child_active.sum() == 0:
                continue
            consistent = float(
                (acts_np[child_active, parent_idx] >= acts_np[child_active, child_idx]).mean()
            )
            pair = f"{parent_id} -> {child_id}"
            print(f"  {pair:<50} {consistent:>12.3f}")
            hier_results.append({
                "parent": parent_id, "child": child_id,
                "consistency": round(consistent, 4),
            })

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "reconstruction": {"mse": mse, "baseline_mse": baseline_mse, "r2": r2},
        "sparsity": {"l0_supervised": l0_supervised, "l0_total": l0_total},
        "features": feature_results,
        "mean_f1": mean_f1,
        "mean_auroc": mean_auroc,
        "hierarchy": hier_results,
        "n_test_vectors": int(x_test.shape[0]),
    }

    cfg.eval_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {cfg.eval_path}")
    return results


if __name__ == "__main__":
    evaluate()
