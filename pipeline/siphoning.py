"""
FVE Siphoning Sweep

Hypothesis: unsupervised latents absorb variance that supervised latents
could otherwise explain. As n_unsupervised shrinks, supervised-side
contribution to reconstruction (delta_r2_sup) should grow — at the cost
of total R² if unsupervised latents were capturing real residual concepts.

Trains the same supervised SAE (same n_supervised, frozen decoder, same
data, same seed) with n_unsupervised ∈ {0, 64, 128, 256, 512} and reports:

  - total R² on held-out test set
  - delta_r2_sup = R² - R²(no_supervised)   (how much sup slice contributes)
  - delta_r2_unsup = R² - R²(no_unsup)      (how much unsup slice contributes)
  - mean calibrated F1 on supervised features (tuned on val)
  - naive + calibrated L0 (supervised slice)

A clean result is: shrinking n_unsup raises delta_r2_sup (supervised is
forced to pull more weight), with cal_F1 unchanged (classification is
independent of unsupervised capacity). If cal_F1 *drops* without unsup,
the unsup slice was genuinely helping the encoder (shared features).

Outputs:
    pipeline_data/siphoning.json

Usage:
    python -m pipeline.run --step siphoning
"""

import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .evaluate import optimal_threshold_f1, precision_recall_f1
from .train import (
    SupervisedSAE,
    compute_target_directions,
    set_seed,
    train_supervised_sae,
)


SWEEP_N_UNSUPERVISED = [0, 64, 128, 256, 512]


def evaluate_variant(sae: SupervisedSAE, x_val, y_val, x_test, y_test, cfg):
    """Evaluate a trained variant. Returns dict of metrics."""
    sae.eval().to(cfg.device)
    n_sup = sae.n_supervised
    n_features = n_sup  # supervised = labeled features

    # Forward pass: val (for threshold calibration) and test
    def fwd(x):
        recons, sup_pres, sup_actss, all_actss = [], [], [], []
        with torch.no_grad():
            for i in range(0, x.shape[0], cfg.batch_size):
                xb = x[i : i + cfg.batch_size].to(cfg.device)
                r, sp, sa, aa = sae(xb)
                recons.append(r.cpu())
                sup_pres.append(sp.cpu())
                sup_actss.append(sa.cpu())
                all_actss.append(aa.cpu())
        return (
            torch.cat(recons),
            torch.cat(sup_pres),
            torch.cat(sup_actss),
            torch.cat(all_actss),
        )

    _, val_sup_pre, _, _ = fwd(x_val)
    recon, sup_pre, sup_acts, all_acts = fwd(x_test)

    # Calibrated thresholds from val
    val_gt = y_val.numpy().astype(bool)
    val_scores = val_sup_pre.numpy()
    thresholds = np.zeros(n_features)
    for k in range(n_features):
        if val_gt[:, k].sum() == 0:
            continue
        _, _, _, t = optimal_threshold_f1(val_gt[:, k], val_scores[:, k])
        thresholds[k] = t

    # Reconstruction + ablation-based R² decomposition
    mse = F.mse_loss(recon, x_test).item()
    baseline_mse = F.mse_loss(
        x_test.mean(0, keepdim=True).expand_as(x_test), x_test
    ).item()
    r2 = 1.0 - mse / baseline_mse

    def decode_ablated(zero_start, zero_end):
        parts = []
        with torch.no_grad():
            for i in range(0, all_acts.shape[0], cfg.batch_size):
                a = all_acts[i : i + cfg.batch_size].clone().to(cfg.device)
                a[:, zero_start:zero_end] = 0
                parts.append(sae.decoder(a).cpu())
        return torch.cat(parts)

    if n_sup > 0:
        recon_no_sup = decode_ablated(0, n_sup)
        r2_no_sup = 1.0 - F.mse_loss(recon_no_sup, x_test).item() / baseline_mse
    else:
        r2_no_sup = r2  # no supervised slice to ablate

    if all_acts.shape[1] > n_sup:
        recon_no_unsup = decode_ablated(n_sup, all_acts.shape[1])
        r2_no_unsup = 1.0 - F.mse_loss(recon_no_unsup, x_test).item() / baseline_mse
    else:
        r2_no_unsup = r2

    delta_r2_sup = r2 - r2_no_sup
    delta_r2_unsup = r2 - r2_no_unsup

    # Calibrated F1 on test
    gt = y_test.numpy().astype(bool)
    scores = sup_pre.numpy()
    f1_scores = []
    f1_t0_scores = []
    per_feat = []
    for k in range(n_features):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            per_feat.append({"cal_f1": None, "f1_t0": None, "n_pos": 0})
            continue
        _, _, f1_t0 = precision_recall_f1(gt[:, k], scores[:, k] > 0)
        _, _, cal_f1 = precision_recall_f1(gt[:, k], scores[:, k] > thresholds[k])
        f1_scores.append(cal_f1)
        f1_t0_scores.append(f1_t0)
        per_feat.append({
            "cal_f1": round(cal_f1, 4),
            "f1_t0": round(f1_t0, 4),
            "threshold": round(float(thresholds[k]), 4),
            "n_pos": n_pos,
        })
    mean_cal_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    mean_f1_t0 = float(np.mean(f1_t0_scores)) if f1_t0_scores else 0.0

    # L0 — naive and calibrated (supervised slice only)
    l0_naive_sup = (sup_acts > 0).float().sum(dim=-1).mean().item()
    cal_mask = scores > thresholds[None, :]
    l0_cal_sup = float(cal_mask.sum(axis=-1).mean())
    l0_total_naive = (all_acts > 0).float().sum(dim=-1).mean().item()

    sae.cpu()
    return {
        "r2": round(r2, 4),
        "r2_no_sup": round(r2_no_sup, 4),
        "r2_no_unsup": round(r2_no_unsup, 4),
        "delta_r2_sup": round(delta_r2_sup, 4),
        "delta_r2_unsup": round(delta_r2_unsup, 4),
        "mean_cal_f1": round(mean_cal_f1, 4),
        "mean_f1_t0": round(mean_f1_t0, 4),
        "l0_sup_naive": round(l0_naive_sup, 2),
        "l0_sup_calibrated": round(l0_cal_sup, 2),
        "l0_total_naive": round(l0_total_naive, 2),
        "per_feature": per_feat,
    }


def run(cfg: Config = None):
    """Run the n_unsupervised sweep."""
    if cfg is None:
        cfg = Config()

    if cfg.siphoning_path.exists():
        print(f"Siphoning results already exist: {cfg.siphoning_path}")
        return json.loads(cfg.siphoning_path.read_text())

    # Load data
    for path, name in [
        (cfg.activations_path, "activations"),
        (cfg.annotations_path, "annotations"),
        (cfg.catalog_path, "feature catalog"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    N, T, d_model = activations.shape
    n_features = annotations.shape[-1]

    # Prepare val/test splits matching train.py
    x_flat = activations.reshape(-1, d_model)
    y_flat = annotations.reshape(-1, n_features)
    n_total = x_flat.shape[0]

    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        set_seed(cfg.seed)
        perm = torch.randperm(n_total)

    split_idx = int(cfg.train_fraction * n_total)
    remaining = n_total - split_idx
    val_size = remaining // 2
    val_split = split_idx + val_size

    x_val = x_flat[perm[split_idx:val_split]]
    y_val = y_flat[perm[split_idx:val_split]]
    x_test = x_flat[perm[val_split:]]
    y_test = y_flat[perm[val_split:]]

    print("=" * 70)
    print("FVE SIPHONING SWEEP")
    print("=" * 70)
    print(f"  n_supervised (fixed):    {n_features}")
    print(f"  n_unsupervised values:   {SWEEP_N_UNSUPERVISED}")
    print(f"  Epochs per variant:      {cfg.epochs}")
    print(f"  Frozen decoder:          {cfg.freeze_supervised_decoder}")
    print(f"  Val/Test vectors:        {x_val.shape[0]:,} / {x_test.shape[0]:,}")
    print()

    results = {}

    for n_unsup in SWEEP_N_UNSUPERVISED:
        print(f"\n{'─' * 60}")
        print(f"Training variant: n_unsupervised = {n_unsup}")
        print(f"{'─' * 60}")

        variant_cfg = copy.copy(cfg)
        variant_cfg.n_unsupervised = n_unsup

        sae = train_supervised_sae(
            activations, annotations, features, variant_cfg, save_checkpoint=False,
        )
        metrics = evaluate_variant(sae, x_val, y_val, x_test, y_test, variant_cfg)
        metrics["n_unsupervised"] = n_unsup
        results[str(n_unsup)] = metrics

        print(
            f"  R²={metrics['r2']:.3f}  "
            f"ΔR²_sup={metrics['delta_r2_sup']:+.4f}  "
            f"ΔR²_unsup={metrics['delta_r2_unsup']:+.4f}  "
            f"cal_F1={metrics['mean_cal_f1']:.3f}  "
            f"L0_sup_cal={metrics['l0_sup_calibrated']:.1f}"
        )

    # Summary table
    print("\n" + "=" * 70)
    print("SIPHONING TABLE")
    print("=" * 70)
    print(
        f"  {'n_unsup':>7} {'R²':>7} {'ΔR²_sup':>9} {'ΔR²_unsup':>11} "
        f"{'cal_F1':>7} {'F1@t=0':>8} {'L0_cal':>7} {'L0_sup':>7}"
    )
    print("  " + "-" * 72)
    for n_unsup in SWEEP_N_UNSUPERVISED:
        m = results[str(n_unsup)]
        print(
            f"  {n_unsup:>7} "
            f"{m['r2']:>7.3f} "
            f"{m['delta_r2_sup']:>+9.4f} "
            f"{m['delta_r2_unsup']:>+11.4f} "
            f"{m['mean_cal_f1']:>7.3f} "
            f"{m['mean_f1_t0']:>8.3f} "
            f"{m['l0_sup_calibrated']:>7.1f} "
            f"{m['l0_sup_naive']:>7.1f}"
        )

    # Interpretation
    print("\n  Interpretation:")
    r2_by_n = {n: results[str(n)]["r2"] for n in SWEEP_N_UNSUPERVISED}
    dr2_by_n = {n: results[str(n)]["delta_r2_sup"] for n in SWEEP_N_UNSUPERVISED}
    f1_by_n = {n: results[str(n)]["mean_cal_f1"] for n in SWEEP_N_UNSUPERVISED}

    max_f1_n = max(f1_by_n, key=f1_by_n.get)
    min_f1_n = min(f1_by_n, key=f1_by_n.get)
    dr2_spread = max(dr2_by_n.values()) - min(dr2_by_n.values())
    print(
        f"    • cal_F1 range: {f1_by_n[min_f1_n]:.3f} (n={min_f1_n}) to "
        f"{f1_by_n[max_f1_n]:.3f} (n={max_f1_n})"
    )
    print(
        f"    • ΔR²_sup range: {min(dr2_by_n.values()):+.4f} to "
        f"{max(dr2_by_n.values()):+.4f} (spread={dr2_spread:.4f})"
    )
    if dr2_by_n[0] > dr2_by_n[SWEEP_N_UNSUPERVISED[-1]] + 0.05:
        print("    • SIPHONING CONFIRMED: supervised contribution drops as "
              "n_unsup grows.")
    elif abs(f1_by_n[0] - f1_by_n[SWEEP_N_UNSUPERVISED[-1]]) < 0.02:
        print("    • cal_F1 is insensitive to n_unsup — supervision is "
              "self-sufficient for classification.")

    out = {
        "sweep_values": SWEEP_N_UNSUPERVISED,
        "n_supervised": n_features,
        "n_val_vectors": int(x_val.shape[0]),
        "n_test_vectors": int(x_test.shape[0]),
        "variants": results,
    }
    cfg.siphoning_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved: {cfg.siphoning_path}")
    return out


if __name__ == "__main__":
    run()
