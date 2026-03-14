"""
evaluate.py — Evaluate a trained SupervisedSAE.

Reports:
  1. Reconstruction quality vs a mean-prediction baseline
  2. Per-feature precision, recall, F1 (threshold: sigmoid(pre_act) > 0.5,
     i.e., pre_act > 0)
  3. Hierarchy consistency: what fraction of (parent, child) pairs satisfy
     act(parent) >= act(child) at positions where the child is active?
"""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from model import SupervisedSAE

DATA_DIR = _DIR / "data"
CKPT_DIR = _DIR / "checkpoints"
BATCH_SIZE = 512


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    cfg = torch.load(CKPT_DIR / "config.pt", map_location="cpu", weights_only=True)
    model = SupervisedSAE(cfg["d_model"], cfg["n_supervised"], cfg["n_unsupervised"])
    model.load_state_dict(torch.load(CKPT_DIR / "model.pt", map_location="cpu", weights_only=True))
    model.eval().to(device)

    # --- Load data ---
    activations = torch.from_numpy(
        np.load(DATA_DIR / "activations.npy").astype(np.float32)
    )
    labels_np = np.load(DATA_DIR / "labels.npy")  # (N, SEQ_LEN, n_features) bool

    N, SEQ_LEN, d_model = activations.shape
    n_features = labels_np.shape[-1]

    acts_flat = activations.reshape(-1, d_model)
    labels_flat = labels_np.reshape(-1, n_features)  # bool

    # --- Forward pass in batches ---
    all_recon = []
    all_sup_pre = []
    all_sup_acts = []

    with torch.no_grad():
        for i in range(0, len(acts_flat), BATCH_SIZE):
            x = acts_flat[i : i + BATCH_SIZE].to(device)
            recon, sup_pre, sup_acts, _ = model(x)
            all_recon.append(recon.cpu())
            all_sup_pre.append(sup_pre.cpu())
            all_sup_acts.append(sup_acts.cpu())

    recon_flat = torch.cat(all_recon, dim=0)
    sup_pre_flat = torch.cat(all_sup_pre, dim=0)
    sup_acts_flat = torch.cat(all_sup_acts, dim=0)

    # --- 1. Reconstruction quality ---
    loss_recon = F.mse_loss(recon_flat, acts_flat).item()
    mean_pred = acts_flat.mean(dim=0, keepdim=True).expand_as(acts_flat)
    baseline_mse = F.mse_loss(mean_pred, acts_flat).item()
    var_explained = 1.0 - loss_recon / baseline_mse

    print("=" * 65)
    print("RECONSTRUCTION")
    print(f"  SAE MSE:              {loss_recon:.6f}")
    print(f"  Baseline (mean) MSE:  {baseline_mse:.6f}")
    print(f"  Variance explained:   {var_explained:.3f}")

    # --- 2. Per-feature precision / recall / F1 ---
    with open(_DIR / "features.json") as f:
        features = json.load(f)["features"]

    # Threshold: pre_act > 0  ⟺  sigmoid(pre_act) > 0.5
    preds = (sup_pre_flat.numpy() > 0)  # (N*SEQ_LEN, n_features) bool

    print()
    print("=" * 65)
    print("PER-FEATURE CLASSIFICATION (threshold: pre_act > 0)")
    print(f"  {'Feature':<32} {'P':>6} {'R':>6} {'F1':>6} {'Pos':>8}")
    print("  " + "-" * 58)

    f1_scores = []
    for idx, feat in enumerate(features):
        y_true = labels_flat[:, idx]
        y_pred = preds[:, idx]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            tag = " [no positives]"
            print(f"  {feat['id']:<32} {'—':>6} {'—':>6} {'—':>6} {n_pos:>8}{tag}")
            continue
        p, r, f1 = precision_recall_f1(y_true, y_pred)
        f1_scores.append(f1)
        tag = "  [group]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<32} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {n_pos:>8}{tag}")

    if f1_scores:
        print(f"\n  Mean F1 (features with positives): {np.mean(f1_scores):.3f}")

    # --- 3. Hierarchy consistency ---
    feature_id_to_idx = {feat["id"]: i for i, feat in enumerate(features)}
    hierarchy = {}
    for feat in features:
        if feat.get("parent"):
            parent_idx = feature_id_to_idx[feat["parent"]]
            child_idx = feature_id_to_idx[feat["id"]]
            hierarchy.setdefault(parent_idx, []).append(child_idx)

    print()
    print("=" * 65)
    print("HIERARCHY CONSISTENCY (parent_act >= child_act when child active)")
    print(f"  {'Parent → Child':<44} {'Consistent':>12}")
    print("  " + "-" * 58)

    acts_np = sup_acts_flat.numpy()
    for parent_idx, child_idxs in hierarchy.items():
        parent_id = features[parent_idx]["id"]
        for child_idx in child_idxs:
            child_id = features[child_idx]["id"]
            # Only evaluate at positions where child is predicted active
            child_active = acts_np[:, child_idx] > 0
            if child_active.sum() == 0:
                continue
            parent_acts = acts_np[child_active, parent_idx]
            child_acts = acts_np[child_active, child_idx]
            consistent = (parent_acts >= child_acts).mean()
            print(f"  {parent_id} → {child_id.split('.')[-1]:<36} {consistent:>12.3f}")


if __name__ == "__main__":
    main()
