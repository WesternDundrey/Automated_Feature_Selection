"""
train.py — Train the SupervisedSAE (toy validation pipeline).

PURPOSE: This script validates that the supervised training loop works correctly
on a cheap, programmatically-annotated dataset (GPT-2 / wikitext-2 / color-day-month
features). It is NOT the primary experiment. See rabbit_habit_supervised_sae.ipynb
for the circuit-targeted experiment (Gemma-2-2B-IT / rabbit→habit / Claude supervision).

Loss = MSE(recon, x)
     + λ_sup   * BCE(sup_pre, labels)      # supervised signal (class-balanced)
     + λ_sparse * L1(all_acts)             # sparsity
     + λ_hier   * hierarchy_loss(sup_acts) # parent >= max(children)

The supervised loss is ramped in linearly over WARMUP_STEPS to let the
reconstruction pathway initialize before the supervised signal dominates.
R² is logged at each epoch end; if R² < 0.5 after epoch 1, reduce λ_sup.

Outputs:
  checkpoints/model.pt   final model state dict
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from model import SupervisedSAE

# --- Hyperparameters ---
N_UNSUPERVISED = 200   # free latents for reconstruction residual
LR = 3e-4
BATCH_SIZE = 256
N_EPOCHS = 10
LAMBDA_SUP = 2.0       # weight for supervised BCE loss
LAMBDA_SPARSE = 1e-3   # weight for L1 sparsity
LAMBDA_HIER = 0.5      # weight for hierarchy consistency loss
WARMUP_STEPS = 500     # steps before supervised loss reaches full weight
                       # NOTE: rare features (e.g. day names) hit pos_weight=100;
                       # a short warmup lets reconstruction initialize before
                       # the class-balanced BCE dominates. Monitor R² epoch-by-epoch.
LOG_EVERY = 50         # print a line every N steps

SAVE_DIR = Path("checkpoints")
DATA_DIR = Path("data")


def hierarchy_loss(sup_acts: torch.Tensor, hierarchy: dict[int, list[int]]) -> torch.Tensor:
    """
    For each (parent, children) pair, penalize cases where a child activates
    more strongly than its parent: loss = mean(relu(max_child - parent)).
    """
    loss = sup_acts.new_zeros(())
    for parent_idx, child_idxs in hierarchy.items():
        parent = sup_acts[..., parent_idx]                        # (...,)
        children = sup_acts[..., child_idxs]                     # (..., n_children)
        max_child = children.max(dim=-1).values                   # (...,)
        loss = loss + F.relu(max_child - parent).mean()
    return loss


def main():
    SAVE_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading activations and labels...")
    activations = torch.from_numpy(
        np.load(DATA_DIR / "activations.npy").astype(np.float32)
    )  # (N, SEQ_LEN, D_MODEL)
    labels = torch.from_numpy(
        np.load(DATA_DIR / "labels.npy").astype(np.float32)
    )  # (N, SEQ_LEN, N_FEATURES)

    N, SEQ_LEN, d_model = activations.shape
    n_supervised = labels.shape[-1]
    print(f"  activations: {activations.shape}  labels: {labels.shape}")

    # Flatten token positions — SAE operates on individual residual stream vectors
    acts_flat = activations.reshape(-1, d_model)      # (N*SEQ_LEN, D_MODEL)
    labels_flat = labels.reshape(-1, n_supervised)    # (N*SEQ_LEN, N_FEATURES)

    # --- Class-balanced pos_weight for BCE ---
    # For each feature, weight positives by (n_neg / n_pos) so the loss is balanced.
    pos_counts = labels_flat.sum(dim=0).clamp(min=1.0)          # (n_supervised,)
    neg_counts = labels_flat.shape[0] - pos_counts
    pos_weight = (neg_counts / pos_counts).clamp(max=100.0)     # cap to avoid instability
    pos_weight = pos_weight.to(device)

    # --- Feature catalog & hierarchy ---
    with open("features.json") as f:
        catalog = json.load(f)
    features = catalog["features"]
    feature_id_to_idx = {feat["id"]: i for i, feat in enumerate(features)}

    hierarchy: dict[int, list[int]] = {}
    for feat in features:
        if feat.get("parent"):
            parent_idx = feature_id_to_idx[feat["parent"]]
            child_idx = feature_id_to_idx[feat["id"]]
            hierarchy.setdefault(parent_idx, []).append(child_idx)

    # --- Model & optimizer ---
    model = SupervisedSAE(d_model, n_supervised, N_UNSUPERVISED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    dataset = TensorDataset(acts_flat, labels_flat)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Precompute mean-prediction baseline MSE (= data variance) for R² tracking.
    # R² = 1 - MSE_SAE / MSE_baseline; if R² < 0.5 after epoch 1, lower λ_sup.
    baseline_mse = float(F.mse_loss(
        acts_flat.mean(dim=0, keepdim=True).expand_as(acts_flat),
        acts_flat,
    ).item())

    print(f"\nTraining: {N_EPOCHS} epochs, {len(loader)} steps/epoch, batch={BATCH_SIZE}")
    print(f"  n_supervised={n_supervised}  n_unsupervised={N_UNSUPERVISED}")
    print(f"  λ_sup={LAMBDA_SUP}  λ_sparse={LAMBDA_SPARSE}  λ_hier={LAMBDA_HIER}")
    print(f"  baseline_mse={baseline_mse:.6f}  (R² target: > 0.5 after epoch 1)")
    print()

    step = 0
    for epoch in range(1, N_EPOCHS + 1):
        epoch_losses = {"total": 0.0, "recon": 0.0, "sup": 0.0, "sparse": 0.0, "hier": 0.0}

        for x, lbl in loader:
            x, lbl = x.to(device), lbl.to(device)

            recon, sup_pre, sup_acts, all_acts = model(x)

            loss_recon = F.mse_loss(recon, x)
            loss_sparse = all_acts.abs().mean()
            loss_sup = F.binary_cross_entropy_with_logits(
                sup_pre, lbl, pos_weight=pos_weight
            )
            loss_hier = hierarchy_loss(sup_acts, hierarchy)

            # Ramp supervised and hierarchy losses in over warmup
            sup_scale = min(1.0, step / max(WARMUP_STEPS, 1))
            loss = (
                loss_recon
                + LAMBDA_SUP * sup_scale * loss_sup
                + LAMBDA_SPARSE * loss_sparse
                + LAMBDA_HIER * sup_scale * loss_hier
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.normalize_decoder()

            for k, v in [
                ("total", loss), ("recon", loss_recon),
                ("sup", loss_sup), ("sparse", loss_sparse), ("hier", loss_hier),
            ]:
                epoch_losses[k] += v.item()

            if step % LOG_EVERY == 0:
                print(
                    f"  step {step:5d}  "
                    f"recon={loss_recon.item():.5f}  "
                    f"sup={loss_sup.item():.5f}  "
                    f"sparse={loss_sparse.item():.5f}  "
                    f"hier={loss_hier.item():.5f}  "
                    f"sup_scale={sup_scale:.2f}"
                )
            step += 1

        n_steps = len(loader)
        avg_recon = epoch_losses["recon"] / n_steps
        r2 = 1.0 - avg_recon / baseline_mse
        print(
            f"Epoch {epoch}/{N_EPOCHS}  "
            + "  ".join(f"{k}={v/n_steps:.5f}" for k, v in epoch_losses.items())
            + f"  R²={r2:.3f}"
        )
        if epoch == 1 and r2 < 0.5:
            print(
                f"  WARNING: R²={r2:.3f} after epoch 1. "
                "Reconstruction may be dominated by supervised loss. "
                "Consider reducing λ_sup or increasing WARMUP_STEPS."
            )

    torch.save(model.state_dict(), SAVE_DIR / "model.pt")
    # Save config alongside for evaluate.py to load
    torch.save(
        {"d_model": d_model, "n_supervised": n_supervised, "n_unsupervised": N_UNSUPERVISED},
        SAVE_DIR / "config.pt",
    )
    print(f"\nSaved model to {SAVE_DIR}/model.pt")


if __name__ == "__main__":
    main()
