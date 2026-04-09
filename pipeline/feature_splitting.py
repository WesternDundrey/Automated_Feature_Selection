"""
Experiment A — Feature Splitting Quantification

For each of N target features, compute a 3-way comparison of how many latents
are needed to cover the concept:

    (S) supervised latents in our SAE (index k = feature index)
    (U) unsupervised latents in our SAE (indices n_supervised..n_total)
    (P) pretrained GemmaScope SAE latents (16K)

Metrics per pool:
    top1_coverage:    fraction of positive positions covered by the best single latent
    top1_specificity: fraction of that latent's fires landing on positive positions
    n_at_80:          greedy set-cover — min #latents to cover 80% of positives
    cov_at_n80:       actual coverage reached when n_at_80 stops

Output:
    pipeline_data/feature_splitting.json

Usage:
    python -m pipeline.run --step splitting
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import Config
from .train import SupervisedSAE


# Target features to test. Matched against catalog feature IDs by substring
# (the flattened catalog uses IDs like "code_and_technical.programming_keyword").
TARGET_FEATURE_HINTS = [
    "programming_keyword",
    "digit_or_number",
    "database_schema",
    "place_name",
    "food_and_cooking",
    "section_header",
    "markup_tag",
    "article_or_determiner",
    "capitalized_proper_noun",
    "bracket_or_paren",
]


def _find_target_indices(
    features: list[dict],
    annotations: torch.Tensor | None = None,
    min_positives: int = 30,
    target_count: int = 10,
) -> list[tuple[int, str]]:
    """Return [(feature_idx, feature_id), ...] for leaf features to analyze.

    Priority:
      1. Features matching the hardcoded hints in TARGET_FEATURE_HINTS
      2. If fewer than min(target_count//2, 5) hints matched, fall back to
         top-`target_count` leaf features by positive count in the annotation
         tensor (if provided), excluding any already matched.
    """
    matched: list[tuple[int, str]] = []
    used: set[int] = set()
    for hint in TARGET_FEATURE_HINTS:
        for idx, feat in enumerate(features):
            if feat.get("type") == "group":
                continue
            if idx in used:
                continue
            if hint in feat["id"]:
                matched.append((idx, feat["id"]))
                used.add(idx)
                break

    min_required = max(3, target_count // 2)
    if len(matched) < min_required and annotations is not None:
        # Fall back: rank leaf features by positive count in full annotations
        pos_counts = annotations.reshape(-1, annotations.shape[-1]).sum(dim=0)
        leaf_flags = [
            i for i, f in enumerate(features)
            if f.get("type") != "group" and i < len(pos_counts)
        ]
        leaf_flags.sort(key=lambda i: -float(pos_counts[i].item()))
        for idx in leaf_flags:
            if idx in used:
                continue
            if float(pos_counts[idx].item()) < min_positives:
                break
            matched.append((idx, features[idx]["id"]))
            used.add(idx)
            if len(matched) >= target_count:
                break

    return matched[:target_count]


def _greedy_set_cover(
    pos_fires: torch.Tensor, target: float = 0.80, max_n: int | None = None,
) -> tuple[int, float, list[int]]:
    """Greedy min-set-cover over latents covering positive positions.

    Args:
        pos_fires: (n_pos, n_latents) bool tensor — does latent i fire at
                   positive position j?
        target: stop once this fraction of positives are covered
        max_n: cap on number of latents to select. If None, defaults to
               min(n_latents, 200) so that the cap scales with pool size
               but remains tractable for the 16K-latent pretrained pool.

    Returns:
        n_used: number of latents selected
        coverage: final fraction of positives covered
        used_latents: ordered list of latent indices selected
    """
    n_pos, n_latents = pos_fires.shape
    if max_n is None:
        max_n = min(n_latents, 200)
    covered = torch.zeros(n_pos, dtype=torch.bool)
    used: list[int] = []

    while covered.float().mean().item() < target and len(used) < max_n:
        remaining = ~covered
        if not remaining.any():
            break
        # How many new positives would each latent cover?
        new_cov = (pos_fires[remaining]).int().sum(dim=0)  # (n_latents,)
        if used:
            new_cov[used] = -1
        best = int(new_cov.argmax().item())
        if new_cov[best].item() <= 0:
            break
        used.append(best)
        covered = covered | pos_fires[:, best]

    return len(used), float(covered.float().mean().item()), used


def _coverage_stats(
    all_fires: torch.Tensor, pos_mask: torch.Tensor, latent_idx: int,
) -> tuple[float, float, int, int]:
    """For one latent, compute (coverage, specificity, n_fires_on_pos, n_fires_total).

    all_fires: (n_test, n_latents) bool
    pos_mask: (n_test,) bool
    latent_idx: which latent to inspect
    """
    col = all_fires[:, latent_idx]
    n_pos = int(pos_mask.sum().item())
    n_fires_on_pos = int((col & pos_mask).sum().item())
    n_fires_total = int(col.sum().item())
    coverage = n_fires_on_pos / n_pos if n_pos > 0 else 0.0
    specificity = n_fires_on_pos / n_fires_total if n_fires_total > 0 else 0.0
    return coverage, specificity, n_fires_on_pos, n_fires_total


def _best_latent_by_coverage(
    all_fires: torch.Tensor, pos_mask: torch.Tensor,
) -> int:
    """Return the latent index with the highest coverage of pos_mask."""
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        return 0
    pos_fires = all_fires[pos_mask]  # (n_pos, n_latents)
    per_latent_cov = pos_fires.float().mean(dim=0)  # (n_latents,)
    return int(per_latent_cov.argmax().item())


def _analyze_pool(
    name: str, all_fires: torch.Tensor, pos_mask: torch.Tensor,
    fixed_latent_idx: int | None = None,
) -> dict:
    """Run coverage/specificity/N@80% analysis for one latent pool.

    If fixed_latent_idx is provided (supervised case), use that as the top-1.
    Otherwise, search the pool for the best-coverage latent.
    """
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        return {
            "pool": name, "n_positives": 0,
            "top1_latent": None, "top1_coverage": None,
            "top1_specificity": None, "n_fires_on_pos": None,
            "n_fires_total": None, "n_at_80": None,
            "cov_at_n80": None, "cover_trajectory": [],
        }

    if fixed_latent_idx is not None:
        top1 = fixed_latent_idx
    else:
        top1 = _best_latent_by_coverage(all_fires, pos_mask)

    cov, spec, n_fp, n_ft = _coverage_stats(all_fires, pos_mask, top1)

    pos_fires = all_fires[pos_mask]  # (n_pos, n_latents)
    n_at_80, cov_at_n80, used_latents = _greedy_set_cover(pos_fires)

    # Cover trajectory: cumulative coverage as each greedy latent is added
    cumulative = torch.zeros(n_pos, dtype=torch.bool)
    trajectory = []
    for lat in used_latents:
        cumulative = cumulative | pos_fires[:, lat]
        trajectory.append(float(cumulative.float().mean().item()))

    return {
        "pool": name,
        "n_positives": n_pos,
        "top1_latent": int(top1),
        "top1_coverage": round(cov, 4),
        "top1_specificity": round(spec, 4),
        "n_fires_on_pos": n_fp,
        "n_fires_total": n_ft,
        "n_at_80": n_at_80,
        "cov_at_n80": round(cov_at_n80, 4),
        "cover_trajectory": [round(v, 4) for v in trajectory],
        "used_latents": [int(l) for l in used_latents],
    }


def run(cfg: Config = None):
    """Main entry for the feature splitting experiment."""
    if cfg is None:
        cfg = Config()

    # ── Load supervised SAE ──────────────────────────────────────────────
    print("Loading trained supervised SAE...")
    model_cfg = torch.load(
        cfg.checkpoint_config_path, map_location="cpu", weights_only=True,
    )
    sae_sup = SupervisedSAE(
        model_cfg["d_model"],
        model_cfg["n_supervised"],
        model_cfg["n_unsupervised"],
        n_lista_steps=model_cfg.get("n_lista_steps", 0),
    )
    sae_sup.load_state_dict(
        torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    )
    sae_sup.eval().to(cfg.device)
    n_supervised = sae_sup.n_supervised
    n_total = sae_sup.n_total
    n_unsup = n_total - n_supervised
    print(f"  Supervised SAE: {n_supervised} supervised + {n_unsup} unsupervised = "
          f"{n_total} latents")

    # ── Load pretrained SAE (via the fixed inventory wrapper) ────────────
    print("Loading pretrained SAE...")
    from .inventory import load_sae
    sae_pre, _ = load_sae(cfg)
    sae_pre = sae_pre.to(cfg.device)
    d_sae_pre = sae_pre.d_sae
    print(f"  Pretrained SAE: {d_sae_pre} latents")

    # ── Load data + test split ───────────────────────────────────────────
    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    N, T, d_model = activations.shape
    n_features = annotations.shape[-1]
    x_flat = activations.reshape(-1, d_model)
    y_flat = annotations.reshape(-1, n_features)
    n_total_vecs = x_flat.shape[0]

    if cfg.split_path.exists():
        perm = torch.load(cfg.split_path, weights_only=True)
    else:
        print("WARNING: split_indices.pt not found, using identity permutation")
        perm = torch.arange(n_total_vecs)

    split_idx = int(cfg.train_fraction * n_total_vecs)
    remaining = n_total_vecs - split_idx
    val_size = remaining // 2
    val_split = split_idx + val_size
    test_idx = perm[val_split:]
    x_test = x_flat[test_idx]
    y_test = y_flat[test_idx]
    print(f"Test set: {x_test.shape[0]:,} vectors")

    # ── Encode the full test set through both SAEs ───────────────────────
    # We store FIRES (bool) rather than full activations to save memory.
    # For the supervised SAE, we keep supervised and unsupervised slices separately.
    print("Encoding test set through supervised SAE...")
    fires_S_list = []   # (n_test, n_supervised) bool
    fires_U_list = []   # (n_test, n_unsupervised) bool
    bs = cfg.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, x_test.shape[0], bs)):
            x_b = x_test[i : i + bs].to(cfg.device)
            _, _, sup_acts, all_acts = sae_sup(x_b)
            fires_S_list.append((sup_acts > 0).cpu())
            fires_U_list.append((all_acts[..., n_supervised:] > 0).cpu())
    fires_S = torch.cat(fires_S_list)  # (n_test, n_supervised)
    fires_U = torch.cat(fires_U_list)  # (n_test, n_unsupervised)
    del fires_S_list, fires_U_list

    print("Encoding test set through pretrained SAE...")
    fires_P_list = []
    with torch.no_grad():
        for i in tqdm(range(0, x_test.shape[0], bs)):
            x_b = x_test[i : i + bs].to(cfg.device)
            z = sae_pre.encode(x_b)
            fires_P_list.append((z > 0).cpu())
    fires_P = torch.cat(fires_P_list)  # (n_test, d_sae_pre)
    del fires_P_list

    print(f"  fires_S: {tuple(fires_S.shape)}  "
          f"fires_U: {tuple(fires_U.shape)}  "
          f"fires_P: {tuple(fires_P.shape)}")

    # ── Run per-feature analysis ─────────────────────────────────────────
    targets = _find_target_indices(features, annotations=annotations)
    if not targets:
        print("ERROR: no target features found. Check catalog and annotations.")
        return
    print(f"\nAnalyzing {len(targets)} target features:")
    for idx, fid in targets:
        print(f"  [{idx}] {fid}")

    y_test_bool = y_test.bool()
    per_feature = []

    print("\n" + "=" * 78)
    print(f"{'Feature':<44} {'Pool':<4} "
          f"{'Top1Cov':>8} {'Top1Spec':>9} {'N@80%':>6}")
    print("-" * 78)

    for feat_idx, feat_id in targets:
        if feat_idx >= n_features or feat_idx >= fires_S.shape[1]:
            print(f"  {feat_id:<42}  SKIP (feat_idx={feat_idx} out of range)")
            continue
        pos_mask = y_test_bool[:, feat_idx]
        n_pos = int(pos_mask.sum().item())

        if n_pos < 5:
            print(f"  {feat_id:<42}  SKIP (n_pos={n_pos})")
            continue

        result_S = _analyze_pool(
            "S", fires_S, pos_mask, fixed_latent_idx=feat_idx,
        )
        result_U = _analyze_pool("U", fires_U, pos_mask)
        result_P = _analyze_pool("P", fires_P, pos_mask)

        for pool_result in (result_S, result_U, result_P):
            cov_str = f"{pool_result['top1_coverage']:.3f}"
            spec_str = f"{pool_result['top1_specificity']:.3f}"
            n80_str = str(pool_result["n_at_80"])
            line = (
                f"  {feat_id:<42} {pool_result['pool']:<4} "
                f"{cov_str:>8} {spec_str:>9} {n80_str:>6}"
            )
            print(line)

        per_feature.append({
            "feature_idx": feat_idx,
            "feature_id": feat_id,
            "n_positives": n_pos,
            "supervised": result_S,
            "unsupervised": result_U,
            "pretrained": result_P,
        })
        print()

    # ── Aggregate metrics ────────────────────────────────────────────────
    def _agg(key: str, pool_key: str) -> float:
        vals = [
            pf[pool_key][key] for pf in per_feature
            if pf[pool_key][key] is not None
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _median_n80(pool_key: str) -> int:
        vals = [
            pf[pool_key]["n_at_80"] for pf in per_feature
            if pf[pool_key]["n_at_80"] is not None
        ]
        return int(np.median(vals)) if vals else 0

    if not per_feature:
        print("\nWARNING: no features were successfully analyzed.")
        agg = {
            "supervised": {"mean_top1_coverage": 0.0, "mean_top1_specificity": 0.0, "median_n_at_80": 0},
            "unsupervised": {"mean_top1_coverage": 0.0, "mean_top1_specificity": 0.0, "median_n_at_80": 0},
            "pretrained": {"mean_top1_coverage": 0.0, "mean_top1_specificity": 0.0, "median_n_at_80": 0},
        }
    else:
        agg = {
            "supervised": {
                "mean_top1_coverage": round(_agg("top1_coverage", "supervised"), 4),
                "mean_top1_specificity": round(_agg("top1_specificity", "supervised"), 4),
                "median_n_at_80": _median_n80("supervised"),
            },
            "unsupervised": {
                "mean_top1_coverage": round(_agg("top1_coverage", "unsupervised"), 4),
                "mean_top1_specificity": round(_agg("top1_specificity", "unsupervised"), 4),
                "median_n_at_80": _median_n80("unsupervised"),
            },
            "pretrained": {
                "mean_top1_coverage": round(_agg("top1_coverage", "pretrained"), 4),
                "mean_top1_specificity": round(_agg("top1_specificity", "pretrained"), 4),
                "median_n_at_80": _median_n80("pretrained"),
            },
        }

    print("\n" + "=" * 78)
    print("AGGREGATE (mean over analyzed features)")
    print("-" * 78)
    print(f"  {'Pool':<18} {'Top1Cov':>10} {'Top1Spec':>10} {'Median N@80%':>14}")
    for pool in ("supervised", "unsupervised", "pretrained"):
        row = agg[pool]
        print(
            f"  {pool:<18} "
            f"{row['mean_top1_coverage']:>10.3f} "
            f"{row['mean_top1_specificity']:>10.3f} "
            f"{row['median_n_at_80']:>14d}"
        )

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "A_feature_splitting",
        "n_supervised": n_supervised,
        "n_unsupervised": n_unsup,
        "n_pretrained": d_sae_pre,
        "aggregate": agg,
        "per_feature": per_feature,
    }
    out_path = cfg.output_dir / "feature_splitting.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
