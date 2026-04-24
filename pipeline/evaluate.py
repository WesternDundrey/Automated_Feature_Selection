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
from .train import SupervisedSAE, build_hierarchy_map, compute_target_directions, set_seed


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def optimal_threshold_f1(y_true: np.ndarray, scores: np.ndarray,
                          n_thresholds: int = 200) -> tuple[float, float, float, float]:
    """Find the threshold maximizing F1. Returns (best_f1, best_p, best_r, best_thresh)."""
    lo, hi = float(scores.min()), float(scores.max())
    best_f1, best_p, best_r, best_t = 0.0, 0.0, 0.0, 0.0
    for t in np.linspace(lo, hi, n_thresholds):
        p, r, f1 = precision_recall_f1(y_true, scores > t)
        if f1 > best_f1:
            best_f1, best_p, best_r, best_t = f1, p, r, float(t)
    return best_f1, best_p, best_r, best_t


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
    remaining = n_total - split_idx
    val_size = remaining // 2
    val_split = split_idx + val_size

    val_idx = perm[split_idx:val_split]
    test_idx = perm[val_split:]

    x_val = x_flat[val_idx]
    y_val = y_flat[val_idx]
    x_test = x_flat[test_idx]
    y_test = y_flat[test_idx]

    print(f"Val: {x_val.shape[0]:,} vectors  |  Test: {x_test.shape[0]:,} vectors")

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

    # ── Val forward pass (threshold calibration) ─────────────────────────
    val_sup_pre_list = []
    with torch.no_grad():
        for i in range(0, x_val.shape[0], cfg.batch_size):
            x_b = x_val[i : i + cfg.batch_size].to(cfg.device)
            _, vsp, _, _ = sae(x_b)
            val_sup_pre_list.append(vsp.cpu())
    val_sup_pre = torch.cat(val_sup_pre_list)

    val_gt = y_val.numpy().astype(bool)
    val_scores = val_sup_pre.numpy()
    calibrated_thresholds = np.zeros(n_features)
    # Per-feature F1 on val AT the calibrated threshold. This is val-only —
    # no test contamination — so it's the correct signal for gating promotion
    # decisions across multiple rounds of the promote loop. Do NOT use
    # `cal_f1` (val-calibrated threshold applied on test) for that purpose:
    # repeating promote-loop rounds that filter on cal_f1 leaks test labels
    # into the pruned catalog.
    val_f1_cal = np.zeros(n_features)
    for k in range(n_features):
        n_pos = int(val_gt[:, k].sum())
        if n_pos == 0:
            continue
        best_f1, _, _, best_t = optimal_threshold_f1(val_gt[:, k], val_scores[:, k])
        calibrated_thresholds[k] = best_t
        val_f1_cal[k] = best_f1

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

    # ── 1b. Supervised vs unsupervised R² decomposition ──────────────────
    # Ablation-based: zero one slice, measure how much R² drops.
    # all_acts is on CPU (collected with .cpu()); decode in batches on GPU.
    with torch.no_grad():
        def _decode_ablated(acts_tensor, zero_start, zero_end):
            """Decode acts with a zeroed slice, batched on GPU."""
            recon_parts = []
            for i in range(0, acts_tensor.shape[0], cfg.batch_size):
                a = acts_tensor[i : i + cfg.batch_size].clone().to(cfg.device)
                a[:, zero_start:zero_end] = 0
                recon_parts.append(sae.decoder(a).cpu())
            return torch.cat(recon_parts)

        # Zero supervised → unsupervised-only reconstruction
        recon_no_sup = _decode_ablated(all_acts, 0, sae.n_supervised)
        mse_no_sup = F.mse_loss(recon_no_sup, x_test).item()
        r2_no_sup = 1.0 - mse_no_sup / baseline_mse

        # Zero unsupervised → supervised-only reconstruction
        recon_no_unsup = _decode_ablated(all_acts, sae.n_supervised, all_acts.shape[1])
        mse_no_unsup = F.mse_loss(recon_no_unsup, x_test).item()
        r2_no_unsup = 1.0 - mse_no_unsup / baseline_mse

    delta_r2_sup = r2 - r2_no_sup      # removing sup hurts by this much
    delta_r2_unsup = r2 - r2_no_unsup  # removing unsup hurts by this much

    print(f"\n  SUPERVISED vs UNSUPERVISED R² DECOMPOSITION (ablation-based)")
    print(f"  Full R²:                {r2:.4f}")
    print(f"  R² without supervised:  {r2_no_sup:.4f}  (delta = {delta_r2_sup:+.4f})")
    print(f"  R² without unsupervised:{r2_no_unsup:.4f}  (delta = {delta_r2_unsup:+.4f})")
    print(f"  Supervised-only R²:     {r2_no_unsup:.4f}")
    print(f"  Unsupervised-only R²:   {r2_no_sup:.4f}")

    # Magnitude correlation: at positive positions, does sup_acts[k] correlate
    # with x @ target_dir[k]? High correlation = reconstruction already
    # supervises activation magnitude without an explicit loss term.
    use_mse = model_cfg.get("use_mse_supervision", False)
    mag_corr = None
    if use_mse and cfg.target_dirs_path.exists():
        target_dirs_q4 = torch.load(cfg.target_dirs_path, weights_only=True)
        projections = x_test @ target_dirs_q4.T  # (n_test, n_features)
        gt_bool = y_test.bool()
        corrs = []
        for k in range(n_features):
            mask = gt_bool[:, k]
            n_pos = int(mask.sum().item())
            if n_pos < 10:
                continue
            proj_k = projections[mask, k].numpy()
            act_k = sup_acts[mask, k].numpy()
            if proj_k.std() < 1e-8 or act_k.std() < 1e-8:
                continue
            corr = float(np.corrcoef(proj_k, act_k)[0, 1])
            if not np.isnan(corr):
                corrs.append(corr)
        if corrs:
            mag_corr = float(np.mean(corrs))
            print(f"\n  MAGNITUDE CORRELATION (sup_acts vs x @ target_dir, at positive positions)")
            print(f"  Mean Pearson r:  {mag_corr:.4f}  (over {len(corrs)} features)")
            print(f"  Interpretation:  {'reconstruction supervises magnitude well' if mag_corr > 0.7 else 'magnitude is weakly supervised by reconstruction' if mag_corr > 0.4 else 'magnitude is NOT supervised by reconstruction'}")

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

    # ── 2a. Calibrated-threshold F1 (tuned on val, applied to test) ──
    print(f"\n  CALIBRATED F1 (thresholds from val set, evaluated on test):")
    print(f"  {'Feature':<36} {'calF1':>6} {'calP':>6} {'calR':>6} {'thresh':>8}")
    print("  " + "-" * 65)

    cal_f1_scores = []
    for k, feat in enumerate(features):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            continue
        cal_pred = scores[:, k] > calibrated_thresholds[k]
        cal_p, cal_r, cal_f1 = precision_recall_f1(gt[:, k], cal_pred)
        cal_f1_scores.append(cal_f1)
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<36} {cal_f1:>6.3f} {cal_p:>6.3f} {cal_r:>6.3f} "
              f"{calibrated_thresholds[k]:>8.3f}{tag}")
        feature_results[k]["cal_f1"] = round(cal_f1, 4)
        feature_results[k]["cal_threshold"] = round(float(calibrated_thresholds[k]), 4)
        # Val-only F1 (threshold optimized on val, evaluated on val). Safe to
        # use for multi-round promotion decisions without test leakage.
        feature_results[k]["val_f1_cal"] = round(float(val_f1_cal[k]), 4)

    cal_mean_f1 = float(np.mean(cal_f1_scores)) if cal_f1_scores else 0.0
    val_mean_f1_cal = float(np.mean(val_f1_cal[val_f1_cal > 0])) if (val_f1_cal > 0).any() else 0.0
    print(f"\n  Mean calibrated F1: {cal_mean_f1:.3f}  (vs t=0: {mean_f1:.3f})")
    print(f"  Mean val-only F1 (for promote-loop gating, no test contamination): {val_mean_f1_cal:.3f}")

    # ── 2b. Oracle-threshold F1 (per-feature optimum on test — reference only)
    print(f"\n  ORACLE F1 (per-feature optimum on test set — NOT honest eval):")
    print(f"  {'Feature':<36} {'optF1':>6} {'optP':>6} {'optR':>6} {'thresh':>8}")
    print("  " + "-" * 65)

    opt_f1_scores = []
    for k, feat in enumerate(features):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            continue
        opt_f1, opt_p, opt_r, opt_t = optimal_threshold_f1(gt[:, k], scores[:, k])
        opt_f1_scores.append(opt_f1)
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<36} {opt_f1:>6.3f} {opt_p:>6.3f} {opt_r:>6.3f} {opt_t:>8.3f}{tag}")
        feature_results[k]["opt_f1"] = round(opt_f1, 4)
        feature_results[k]["opt_threshold"] = round(opt_t, 4)

    opt_mean_f1 = float(np.mean(opt_f1_scores)) if opt_f1_scores else 0.0
    print(f"\n  Mean oracle F1: {opt_mean_f1:.3f}  (calibrated: {cal_mean_f1:.3f}, t=0: {mean_f1:.3f})")

    # ── 3. Sparsity ─────────────────────────────────────────────────────
    l0_supervised = (sup_acts > 0).float().sum(dim=-1).mean().item()
    l0_total = (all_acts > 0).float().sum(dim=-1).mean().item()
    print(f"\n  L0 (supervised latents, naive >0): {l0_supervised:.2f}")
    print(f"  L0 (all latents, naive >0):        {l0_total:.2f}")

    # ── 3b. Calibrated L0 ───────────────────────────────────────────────
    # Naive L0 counts any non-zero activation. But a feature with optimal
    # threshold 2.5 firing at sup_pre=0.1 is noise, not "active". True L0
    # uses each feature's F1-optimal threshold (computed on val) to count
    # only confidently-active latents. Unsupervised latents retain naive >0
    # (no labels to calibrate against).
    test_sup_pre_np = sup_pre.numpy()
    unsup_acts_np = all_acts[:, sae.n_supervised :].numpy()
    cal_mask_sup = test_sup_pre_np > calibrated_thresholds[None, :]
    unsup_active = unsup_acts_np > 0
    l0_sup_calibrated = float(cal_mask_sup.sum(axis=-1).mean())
    l0_unsup = float(unsup_active.sum(axis=-1).mean())
    l0_total_calibrated = l0_sup_calibrated + l0_unsup

    fire_rate_naive = (sup_acts.numpy() > 0).mean(axis=0)
    fire_rate_calibrated = cal_mask_sup.mean(axis=0)
    gt_pos_rate = gt.mean(axis=0)

    print(f"\n  L0 (supervised, calibrated):       {l0_sup_calibrated:.2f}  "
          f"(naive: {l0_supervised:.2f}, ratio: {l0_sup_calibrated / max(l0_supervised, 1e-9):.2f})")
    print(f"  L0 (all, calibrated sup + naive unsup): {l0_total_calibrated:.2f}  "
          f"(naive: {l0_total:.2f})")
    print(f"\n  Per-feature fire rates (test set):")
    print(f"  {'Feature':<36} {'r@0':>7} {'r@cal':>7} {'r@gt':>7} {'thresh':>8} {'n_pos':>6}")
    print("  " + "-" * 78)
    for k, feat in enumerate(features):
        n_pos = int(gt[:, k].sum())
        tag = " [G]" if feat["type"] == "group" else ""
        print(f"  {feat['id']:<36} {fire_rate_naive[k]:>7.4f} "
              f"{fire_rate_calibrated[k]:>7.4f} {gt_pos_rate[k]:>7.4f} "
              f"{calibrated_thresholds[k]:>8.3f} {n_pos:>6}{tag}")
        feature_results[k]["fire_rate_naive"] = round(float(fire_rate_naive[k]), 6)
        feature_results[k]["fire_rate_calibrated"] = round(float(fire_rate_calibrated[k]), 6)
        feature_results[k]["gt_positive_rate"] = round(float(gt_pos_rate[k]), 6)

    # Calibration quality: how close is fire rate to ground-truth positive rate?
    # |fire_rate_cal - gt_pos_rate| < 0.01 = well-calibrated
    cal_error = np.abs(fire_rate_calibrated - gt_pos_rate)
    n_well_cal = int((cal_error < 0.01).sum())
    print(f"\n  Calibration quality: {n_well_cal}/{len(features)} features within "
          f"0.01 of GT positive rate (|r@cal - r@gt| < 0.01)")
    print(f"  Median |r@cal - r@gt|: {float(np.median(cal_error)):.4f}")

    # ── 3c. Ground-truth L0 vs predicted L0 ─────────────────────────────
    # GT L0 = mean # of features marked positive per position by the LLM
    # annotations. Equals sum of per-feature gt_positive_rate (linearity of
    # expectation). This is the "right" number — what calibrated L0 should
    # match. Naive L0 (>0) almost always overshoots because pre-act > 0
    # doesn't mean the feature is confidently present.
    gt_l0_per_pos = gt.sum(axis=-1)
    gt_l0 = float(gt_l0_per_pos.mean())
    pred_l0_per_pos = cal_mask_sup.sum(axis=-1)
    abs_err_per_pos = np.abs(pred_l0_per_pos - gt_l0_per_pos)

    print(f"\n  GROUND-TRUTH L0 vs PREDICTED L0 (supervised slice)")
    print(f"  GT L0 (from annotations):     {gt_l0:.2f}")
    print(f"  Calibrated L0 (predicted):    {l0_sup_calibrated:.2f}  "
          f"(ratio cal/gt = {l0_sup_calibrated / max(gt_l0, 1e-9):.3f})")
    print(f"  Naive L0 (>0, predicted):     {l0_supervised:.2f}  "
          f"(ratio naive/gt = {l0_supervised / max(gt_l0, 1e-9):.3f})")
    print(f"  Per-position |pred - GT|:     "
          f"mean={float(abs_err_per_pos.mean()):.2f}, "
          f"median={float(np.median(abs_err_per_pos)):.2f}")
    if l0_sup_calibrated > 0 and gt_l0 > 0:
        verdict = (
            "calibrated L0 matches GT — SAE fires the right amount"
            if 0.9 <= l0_sup_calibrated / gt_l0 <= 1.1
            else f"calibrated L0 overshoots GT by "
                 f"{(l0_sup_calibrated/gt_l0 - 1)*100:+.0f}%"
            if l0_sup_calibrated > gt_l0
            else f"calibrated L0 undershoots GT by "
                 f"{(1 - l0_sup_calibrated/gt_l0)*100:.0f}%"
        )
        print(f"  Verdict: {verdict}")

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

    # ── 4b. MSE supervision metrics (v2) ─────────────────────────────
    use_mse = model_cfg.get("use_mse_supervision", False)
    cosine_results = []
    fve_results = []
    mean_cosine = None
    mean_fve = None
    mean_fvu = None

    if use_mse:
        print("\n" + "=" * 70)
        print("MSE FEATURE DICTIONARY METRICS (v2)")

        # Load or recompute target directions
        if cfg.target_dirs_path.exists():
            target_dirs = torch.load(cfg.target_dirs_path, weights_only=True)
        else:
            print("  Target directions not found on disk, recomputing from train split...")
            train_idx = perm[:split_idx]
            target_dirs, _, _ = compute_target_directions(
                x_flat[train_idx], y_flat[train_idx], n_features,
            )

        # Per-feature cosine similarity: cos(decoder_col_i, target_dir_i)
        dec_weight = sae.decoder.weight.cpu()  # (d_model, n_total)
        dec_cols = dec_weight[:, :n_features]  # (d_model, n_features)
        cosines_all = (dec_cols * target_dirs.T).sum(dim=0)  # (n_features,)

        # Per-feature FVE: fraction of variance in positive-class activations
        # explained by the decoder column direction
        print(f"\n  {'Feature':<36} {'Cos':>7} {'FVE':>7} {'FVU':>7} {'Pos':>8}")
        print("  " + "-" * 69)

        cosine_vals = []
        fve_vals = []

        for k, feat in enumerate(features):
            pos_mask_np = gt[:, k]
            n_pos = int(pos_mask_np.sum())
            pos_mask = torch.from_numpy(pos_mask_np)  # for torch indexing
            cos_k = cosines_all[k].item()

            if n_pos < 2:
                cosine_results.append({
                    "id": feat["id"], "cosine": round(cos_k, 4),
                    "fve": None, "fvu": None, "n_positives": n_pos,
                })
                fve_results.append({"id": feat["id"], "fve": None, "fvu": None})
                print(f"  {feat['id']:<36} {cos_k:>7.4f} {'--':>7} {'--':>7} {n_pos:>8}")
                continue

            # FVE: fraction of positive-class variance along decoder direction
            pos_acts = x_test[pos_mask]
            pos_centered = pos_acts - pos_acts.mean(dim=0)
            total_var = (pos_centered ** 2).sum().item()

            if total_var < 1e-10:
                fve_k = 0.0
            else:
                dec_dir = dec_cols[:, k]
                dec_dir_unit = dec_dir / dec_dir.norm().clamp(min=1e-8)
                projections = pos_centered @ dec_dir_unit  # (n_pos,)
                explained = (projections ** 2).sum().item()
                fve_k = explained / total_var

            fvu_k = 1.0 - fve_k
            cosine_vals.append(cos_k)
            fve_vals.append(fve_k)
            tag = " [G]" if feat["type"] == "group" else ""
            print(f"  {feat['id']:<36} {cos_k:>7.4f} {fve_k:>7.4f} {fvu_k:>7.4f} {n_pos:>8}{tag}")

            cosine_results.append({
                "id": feat["id"], "cosine": round(cos_k, 4),
                "fve": round(fve_k, 4), "fvu": round(fvu_k, 4),
                "n_positives": n_pos,
            })
            fve_results.append({
                "id": feat["id"], "fve": round(fve_k, 4), "fvu": round(fvu_k, 4),
            })

        if cosine_vals:
            mean_cosine = float(np.mean(cosine_vals))
            mean_fve = float(np.mean(fve_vals))
            mean_fvu = 1.0 - mean_fve
            print(f"\n  Mean cosine to target:          {mean_cosine:.4f}")
            print(f"  Mean FVE (decoder direction):   {mean_fve:.4f}")
            print(f"  Mean FVU (fraction unexplained): {mean_fvu:.4f}")

            high_cos = sum(1 for c in cosine_vals if c > 0.8)
            low_cos = sum(1 for c in cosine_vals if c < 0.5)
            print(f"  Features with cosine > 0.8: {high_cos}/{len(cosine_vals)}")
            print(f"  Features with cosine < 0.5: {low_cos}/{len(cosine_vals)}")

    # ── 5. Linear probe baseline ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("LINEAR PROBE BASELINE (same train/test split)")

    train_idx = perm[:split_idx]
    x_train = x_flat[train_idx]
    y_train = y_flat[train_idx]

    # Class-balanced BCE (same weighting strategy as supervised SAE training)
    n_pos_per_feat = y_train.sum(dim=0)
    pos_weight = ((y_train.shape[0] - n_pos_per_feat) / n_pos_per_feat.clamp(min=1)).clamp(max=100)

    probe = torch.nn.Linear(d_model, n_features)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    probe_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    probe.train()
    for epoch in range(10):
        shuffle_idx = torch.randperm(x_train.shape[0])
        for i in range(0, x_train.shape[0], cfg.batch_size):
            idx = shuffle_idx[i : i + cfg.batch_size]
            logits = probe(x_train[idx])
            loss = probe_loss_fn(logits, y_train[idx])
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()

    probe.eval()
    with torch.no_grad():
        probe_val_logits_list = []
        for i in range(0, x_val.shape[0], cfg.batch_size):
            probe_val_logits_list.append(probe(x_val[i : i + cfg.batch_size]))
        probe_val_logits = torch.cat(probe_val_logits_list).numpy()

        probe_logits_list = []
        for i in range(0, x_test.shape[0], cfg.batch_size):
            probe_logits_list.append(probe(x_test[i : i + cfg.batch_size]))
        probe_logits = torch.cat(probe_logits_list).numpy()

    val_gt_np = y_val.numpy().astype(bool)

    # Per-feature calibrated threshold, optimized on val, applied on test.
    # This matches the protocol used for the supervised SAE at line 161-167;
    # reporting probe F1 at t=0 against the supervised SAE at calibrated
    # thresholds is an apples-to-oranges comparison that inflates the
    # supervised advantage.
    probe_cal_thresholds = np.zeros(n_features)
    for k in range(n_features):
        if int(val_gt_np[:, k].sum()) == 0:
            continue
        _, _, _, best_t = optimal_threshold_f1(val_gt_np[:, k], probe_val_logits[:, k])
        probe_cal_thresholds[k] = best_t
    probe_cal_preds = probe_logits > probe_cal_thresholds[None, :]

    probe_preds = probe_logits > 0
    probe_f1_scores = []
    probe_cal_f1_scores = []
    probe_auroc_scores = []
    probe_results = []

    for k, feat in enumerate(features):
        n_pos = int(gt[:, k].sum())
        if n_pos == 0:
            probe_results.append({
                "id": feat["id"], "f1": None, "f1_cal": None, "auroc": None,
            })
            continue
        pp, pr, pf1 = precision_recall_f1(gt[:, k], probe_preds[:, k])
        _, _, pf1_cal = precision_recall_f1(gt[:, k], probe_cal_preds[:, k])
        pauc = auroc(gt[:, k], probe_logits[:, k])
        probe_f1_scores.append(pf1)
        probe_cal_f1_scores.append(pf1_cal)
        if not np.isnan(pauc):
            probe_auroc_scores.append(pauc)
        probe_results.append({
            "id": feat["id"],
            "f1": round(pf1, 4),
            "f1_cal": round(pf1_cal, 4),
            "auroc": round(pauc, 4) if not np.isnan(pauc) else None,
            "cal_threshold": round(float(probe_cal_thresholds[k]), 4),
        })

    probe_mean_f1 = float(np.mean(probe_f1_scores)) if probe_f1_scores else 0.0
    probe_cal_mean_f1 = (
        float(np.mean(probe_cal_f1_scores)) if probe_cal_f1_scores else 0.0
    )
    probe_mean_auroc = float(np.mean(probe_auroc_scores)) if probe_auroc_scores else 0.0

    print(f"  Probe Mean F1:       {probe_mean_f1:.3f}  "
          f"(supervised SAE t=0: {mean_f1:.3f})")
    print(f"  Probe Calibrated F1: {probe_cal_mean_f1:.3f}  "
          f"(supervised SAE calibrated: {cal_mean_f1:.3f})")
    print(f"  Probe Mean AUROC:    {probe_mean_auroc:.3f}  "
          f"(supervised SAE: {mean_auroc:.3f})")

    del probe  # free memory (keep x_train, y_train for section 7)

    # ── 6 & 7. Pretrained SAE comparisons ─────────────────────────────
    pretrained_mse = None
    pretrained_r2 = None
    posttrain_mean_f1 = 0.0
    posttrain_cal_mean_f1 = 0.0
    posttrain_mean_auroc = 0.0
    posttrain_results = []

    try:
        # Load pretrained SAE via the PretrainedSAE wrapper in inventory.py,
        # which follows the documented GemmaScope JumpReLU convention:
        #     encode: z = JumpReLU_theta(x @ W_enc + b_enc)   (NO b_dec subtraction)
        #     decode: x_hat = z @ W_dec + b_dec
        # The wrapper was previously bypassed in favor of sae_lens's native
        # forward, but that path applied `apply_b_dec_to_input` and/or
        # activation normalization inconsistent with GemmaScope, producing
        # R²=-6.85 on Gemma-2-2B residual-stream activations. The wrapper is
        # the same one the inventory step uses (and is known-correct).
        from .inventory import load_sae

        print("\n" + "=" * 70)
        print("PRETRAINED SAE RECONSTRUCTION COMPARISON")

        wrapped_sae, _ = load_sae(cfg)
        wrapped_sae = wrapped_sae.to(cfg.device)
        d_sae = wrapped_sae.d_sae

        # Sanity check: compute R² on a small batch up front so we fail fast
        # if the pretrained SAE is mis-loaded (e.g., sae_lens weight-convention
        # mismatch that produced R²=-6.85 in summary4).
        with torch.no_grad():
            _probe = x_test[: min(256, x_test.shape[0])].to(cfg.device)
            _probe_recon = wrapped_sae.decode(wrapped_sae.encode(_probe))
            _probe_mse = F.mse_loss(_probe_recon, _probe).item()
            _probe_base = F.mse_loss(
                _probe.mean(0, keepdim=True).expand_as(_probe), _probe,
            ).item()
            _probe_r2 = 1.0 - _probe_mse / max(_probe_base, 1e-9)
            print(f"  [sanity] 256-vec R²={_probe_r2:+.4f}  "
                  f"(MSE={_probe_mse:.3f} vs baseline {_probe_base:.3f})")
            if _probe_r2 < 0.5:
                print("  [sanity] WARNING: pretrained SAE reconstruction is "
                      "broken. Downstream baseline will be unreliable.")
            del _probe, _probe_recon

        # ── 6. Reconstruction comparison ──────────────────────────────
        pre_recon_list = []
        with torch.no_grad():
            for i in range(0, x_test.shape[0], cfg.batch_size):
                x_b = x_test[i : i + cfg.batch_size].to(cfg.device)
                z = wrapped_sae.encode(x_b)
                r = wrapped_sae.decode(z)
                pre_recon_list.append(r.cpu())

        pre_recon = torch.cat(pre_recon_list)
        pretrained_mse = F.mse_loss(pre_recon, x_test).item()
        pretrained_r2 = 1.0 - pretrained_mse / baseline_mse

        n_sup = model_cfg["n_supervised"]
        n_unsup = model_cfg["n_unsupervised"]
        print(f"  Pretrained SAE ({d_sae} latents):  "
              f"MSE={pretrained_mse:.4f}  R²={pretrained_r2:.4f}")
        print(f"  Supervised SAE ({n_sup}+{n_unsup} latents):  "
              f"MSE={mse:.4f}  R²={r2:.4f}")
        print(f"  Reconstruction cost of supervision: {r2 - pretrained_r2:+.4f} R²")
        del pre_recon

        # ── 7. Post-training baseline (AlignSAE-style) ───────────────
        # Train a linear readout from pretrained SAE's 16k latent space
        # to feature labels. Tests whether the pretrained representation
        # already captures our features without from-scratch training.
        print("\n" + "=" * 70)
        print("POST-TRAINING BASELINE (linear readout from pretrained SAE latents)")

        readout = torch.nn.Linear(d_sae, n_features)
        readout_opt = torch.optim.Adam(
            readout.parameters(), lr=1e-3, weight_decay=1e-4
        )
        readout_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        readout.train()
        for epoch in range(10):
            shuffle_idx = torch.randperm(x_train.shape[0])
            for i in range(0, x_train.shape[0], cfg.batch_size):
                idx = shuffle_idx[i : i + cfg.batch_size]
                with torch.no_grad():
                    z_b = wrapped_sae.encode(
                        x_train[idx].to(cfg.device)
                    ).cpu()
                logits = readout(z_b)
                loss = readout_loss_fn(logits, y_train[idx])
                readout_opt.zero_grad()
                loss.backward()
                readout_opt.step()

        readout.eval()
        with torch.no_grad():
            # Compute readout logits on BOTH val (for calibration) and test.
            rt_val_logits_list = []
            for i in range(0, x_val.shape[0], cfg.batch_size):
                z_b = wrapped_sae.encode(
                    x_val[i : i + cfg.batch_size].to(cfg.device)
                ).cpu()
                rt_val_logits_list.append(readout(z_b))
            rt_val_logits = torch.cat(rt_val_logits_list).numpy()

            rt_logits_list = []
            for i in range(0, x_test.shape[0], cfg.batch_size):
                z_b = wrapped_sae.encode(
                    x_test[i : i + cfg.batch_size].to(cfg.device)
                ).cpu()
                rt_logits_list.append(readout(z_b))
            rt_logits = torch.cat(rt_logits_list).numpy()

        # Calibrated thresholds for the post-training readout, same protocol
        # as the supervised SAE: pick best per-feature t on val, evaluate on
        # test. Without this the readout is under-scored at the t=0 default.
        rt_cal_thresholds = np.zeros(n_features)
        for k in range(n_features):
            if int(val_gt_np[:, k].sum()) == 0:
                continue
            _, _, _, best_t = optimal_threshold_f1(val_gt_np[:, k], rt_val_logits[:, k])
            rt_cal_thresholds[k] = best_t
        rt_cal_preds = rt_logits > rt_cal_thresholds[None, :]

        rt_preds = rt_logits > 0
        pt_f1_scores = []
        pt_cal_f1_scores = []
        pt_auroc_scores = []

        for k, feat in enumerate(features):
            n_pos = int(gt[:, k].sum())
            if n_pos == 0:
                posttrain_results.append({
                    "id": feat["id"], "f1": None, "f1_cal": None, "auroc": None,
                })
                continue
            pp, pr, pf1 = precision_recall_f1(gt[:, k], rt_preds[:, k])
            _, _, pf1_cal = precision_recall_f1(gt[:, k], rt_cal_preds[:, k])
            pauc = auroc(gt[:, k], rt_logits[:, k])
            pt_f1_scores.append(pf1)
            pt_cal_f1_scores.append(pf1_cal)
            if not np.isnan(pauc):
                pt_auroc_scores.append(pauc)
            posttrain_results.append({
                "id": feat["id"],
                "f1": round(pf1, 4),
                "f1_cal": round(pf1_cal, 4),
                "auroc": round(pauc, 4) if not np.isnan(pauc) else None,
                "cal_threshold": round(float(rt_cal_thresholds[k]), 4),
            })

        posttrain_mean_f1 = float(np.mean(pt_f1_scores)) if pt_f1_scores else 0.0
        posttrain_cal_mean_f1 = (
            float(np.mean(pt_cal_f1_scores)) if pt_cal_f1_scores else 0.0
        )
        posttrain_mean_auroc = float(np.mean(pt_auroc_scores)) if pt_auroc_scores else 0.0

        print(f"  Post-train Mean F1:       {posttrain_mean_f1:.3f}  "
              f"(supervised SAE t=0: {mean_f1:.3f}, probe t=0: {probe_mean_f1:.3f})")
        print(f"  Post-train Calibrated F1: {posttrain_cal_mean_f1:.3f}  "
              f"(supervised SAE calibrated: {cal_mean_f1:.3f}, probe cal: {probe_cal_mean_f1:.3f})")
        print(f"  Post-train Mean AUROC:    {posttrain_mean_auroc:.3f}  "
              f"(supervised SAE: {mean_auroc:.3f}, probe: {probe_mean_auroc:.3f})")

        del wrapped_sae, readout
    except Exception as e:
        print(f"\n  Skipping pretrained SAE comparisons: {e}")

    del x_train, y_train

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "reconstruction": {
            "mse": mse, "baseline_mse": baseline_mse, "r2": r2,
            "r2_without_supervised": round(r2_no_sup, 4),
            "r2_without_unsupervised": round(r2_no_unsup, 4),
            "delta_r2_supervised": round(delta_r2_sup, 4),
            "delta_r2_unsupervised": round(delta_r2_unsup, 4),
            "magnitude_correlation": round(mag_corr, 4) if mag_corr is not None else None,
        },
        "sparsity": {
            "l0_supervised": l0_supervised,
            "l0_total": l0_total,
            "l0_supervised_calibrated": round(l0_sup_calibrated, 4),
            "l0_total_calibrated": round(l0_total_calibrated, 4),
            "l0_unsupervised": round(l0_unsup, 4),
            "l0_ground_truth": round(gt_l0, 4),
            "l0_cal_over_gt_ratio": round(l0_sup_calibrated / max(gt_l0, 1e-9), 4),
            "l0_naive_over_gt_ratio": round(l0_supervised / max(gt_l0, 1e-9), 4),
            "l0_pred_minus_gt_mean_abs_err": round(float(abs_err_per_pos.mean()), 4),
            "calibration_median_error": round(float(np.median(cal_error)), 4),
            "n_well_calibrated": n_well_cal,
        },
        "features": feature_results,
        "mean_f1": mean_f1,
        "cal_mean_f1": cal_mean_f1,
        "opt_mean_f1": opt_mean_f1,
        "mean_auroc": mean_auroc,
        "n_val_vectors": int(x_val.shape[0]),
        "mse_supervision_metrics": {
            "mean_cosine_to_target": mean_cosine,
            "mean_fve": mean_fve,
            "mean_fvu": mean_fvu,
            "per_feature": cosine_results,
        } if use_mse else None,
        "probe_baseline": {
            "mean_f1": probe_mean_f1,
            "mean_f1_cal": probe_cal_mean_f1,
            "mean_auroc": probe_mean_auroc,
            "per_feature": probe_results,
        },
        "posttrain_baseline": {
            "mean_f1": posttrain_mean_f1,
            "mean_f1_cal": posttrain_cal_mean_f1,
            "mean_auroc": posttrain_mean_auroc,
            "per_feature": posttrain_results,
        },
        "pretrained_reconstruction": {
            "mse": pretrained_mse,
            "r2": pretrained_r2,
        } if pretrained_mse is not None else None,
        "hierarchy": hier_results,
        "n_test_vectors": int(x_test.shape[0]),
    }

    cfg.eval_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {cfg.eval_path}")
    return results


if __name__ == "__main__":
    evaluate()
