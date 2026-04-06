"""
Step 6 — Ablation Study

Train multiple supervised SAE variants with individual components disabled
and compare their performance to produce an ablation table.

Ablations:
  1. baseline          — full model (all components)
  2. no_hierarchy      — lambda_hier = 0
  3. no_warmup         — warmup_steps = 0
  4. no_class_balance  — pos_weight = 1 (uniform)
  5. no_unsupervised   — n_unsupervised = 0
  6. no_lista          — n_lista_steps = 0 (only if baseline uses LISTA)
  7. sup_only          — lambda_sparse = 0 (no L1)

Outputs:
    pipeline_data/ablation.json

Usage:
    python -m pipeline.run --step ablation
"""

import copy
import json

import torch
import torch.nn.functional as F
import numpy as np

from .config import Config
from .train import SupervisedSAE, train_supervised_sae, build_hierarchy_map, set_seed


def evaluate_quick(sae: SupervisedSAE, x_test, y_test, features, cfg):
    """Quick evaluation returning key metrics without full report."""
    sae.eval().to(cfg.device)

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

    # Reconstruction
    mse = F.mse_loss(recon, x_test).item()
    baseline_mse = F.mse_loss(
        x_test.mean(0, keepdim=True).expand_as(x_test), x_test
    ).item()
    r2 = 1.0 - mse / baseline_mse

    # Per-feature F1
    gt = y_test.numpy().astype(bool)
    preds = sup_pre.numpy() > 0
    f1_scores = []
    for k in range(gt.shape[1]):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            continue
        tp = int((gt[:, k] & preds[:, k]).sum())
        fp = int((~gt[:, k] & preds[:, k]).sum())
        fn = int((gt[:, k] & ~preds[:, k]).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)

    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # Sparsity
    l0 = (all_acts > 0).float().sum(dim=-1).mean().item()

    # Hierarchy consistency
    hier_map = build_hierarchy_map(features)
    acts_np = sup_acts.numpy()
    consistencies = []
    for parent_idx, child_idxs in hier_map.items():
        for child_idx in child_idxs:
            child_active = acts_np[:, child_idx] > 0
            if child_active.sum() == 0:
                continue
            c = float(
                (acts_np[child_active, parent_idx] >= acts_np[child_active, child_idx]).mean()
            )
            consistencies.append(c)
    hier_consistency = float(np.mean(consistencies)) if consistencies else 1.0

    sae.cpu()
    return {
        "r2": round(r2, 4),
        "mse": round(mse, 6),
        "mean_f1": round(mean_f1, 4),
        "l0": round(l0, 1),
        "hierarchy_consistency": round(hier_consistency, 4),
    }


def run(cfg: Config = None):
    """Run ablation study comparing model variants."""
    if cfg is None:
        cfg = Config()

    if cfg.ablation_path.exists():
        print(f"Ablation results already exist: {cfg.ablation_path}")
        return json.loads(cfg.ablation_path.read_text())

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

    # Prepare test set (matching train.py's split)
    x_flat = activations.reshape(-1, d_model)
    y_flat = annotations.reshape(-1, n_features)
    n_total = x_flat.shape[0]

    # Load split indices from disk (saved by train.py) to avoid RNG coupling
    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        set_seed(cfg.seed)
        perm = torch.randperm(n_total)

    split_idx = int(cfg.train_fraction * n_total)
    x_test = x_flat[perm[split_idx:]]
    y_test = y_flat[perm[split_idx:]]

    # Define ablation variants
    ablations = {
        "baseline": {},
        "no_hierarchy": {"lambda_hier": 0.0},
        "no_warmup": {"warmup_steps": 0},
        "no_unsupervised": {"n_unsupervised": 0},
        "no_sparsity": {"lambda_sparse": 0.0},
    }

    # Only add LISTA ablation if baseline uses LISTA
    if cfg.n_lista_steps > 0:
        ablations["no_lista"] = {"n_lista_steps": 0}

    # v2: supervision mode ablation
    if cfg.supervision_mode == "hybrid":
        ablations["bce_only"] = {"supervision_mode": "bce"}
        ablations["mse_mode"] = {"supervision_mode": "mse"}
    elif cfg.supervision_mode == "mse":
        ablations["bce_only"] = {"supervision_mode": "bce"}
        ablations["hybrid_mode"] = {"supervision_mode": "hybrid"}
    else:
        ablations["mse_mode"] = {"supervision_mode": "mse"}
        ablations["hybrid_mode"] = {"supervision_mode": "hybrid"}

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print(f"  Variants: {len(ablations)}")
    print(f"  Epochs per variant: {cfg.epochs}")
    print()

    results = {}

    for name, overrides in ablations.items():
        print(f"\n{'─' * 50}")
        print(f"Training variant: {name}")
        if overrides:
            print(f"  Overrides: {overrides}")
        else:
            print("  (no overrides — full model)")
        print(f"{'─' * 50}")

        # Create modified config
        variant_cfg = copy.copy(cfg)
        for k, v in overrides.items():
            setattr(variant_cfg, k, v)

        # Train
        sae = train_supervised_sae(activations, annotations, features, variant_cfg, save_checkpoint=False)

        # Evaluate
        metrics = evaluate_quick(sae, x_test, y_test, features, variant_cfg)
        metrics["overrides"] = overrides
        results[name] = metrics

        print(f"  Result: R2={metrics['r2']:.3f}  F1={metrics['mean_f1']:.3f}  "
              f"L0={metrics['l0']:.1f}  hier={metrics['hierarchy_consistency']:.3f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION TABLE")
    print("=" * 70)
    print(f"  {'Variant':<22} {'R2':>7} {'F1':>7} {'L0':>7} {'Hier':>7}")
    print("  " + "-" * 52)

    baseline = results.get("baseline", {})
    for name, metrics in results.items():
        tag = " *" if name == "baseline" else ""
        print(f"  {name:<22} {metrics['r2']:>7.3f} {metrics['mean_f1']:>7.3f} "
              f"{metrics['l0']:>7.1f} {metrics['hierarchy_consistency']:>7.3f}{tag}")

    # Compute deltas from baseline
    if baseline:
        print(f"\n  {'Delta vs baseline':<22} {'dR2':>7} {'dF1':>7}")
        print("  " + "-" * 38)
        for name, metrics in results.items():
            if name == "baseline":
                continue
            dr2 = metrics["r2"] - baseline["r2"]
            df1 = metrics["mean_f1"] - baseline["mean_f1"]
            print(f"  {name:<22} {dr2:>+7.3f} {df1:>+7.3f}")

    cfg.ablation_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {cfg.ablation_path}")
    return results


if __name__ == "__main__":
    run()
