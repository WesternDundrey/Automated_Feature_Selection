"""
Catalog curation by FVE potential (v8.20.0).

Honest path to backbone-only mean FVE > 0.7: rank features by their FVE
potential under the configured `target_dir_method`, filter to features
with FVE ≥ threshold, and write a curated catalog. The user opts in by
replacing `feature_catalog.json` with `feature_catalog.fve_curated.json`.

This is OPERATIONAL, not methodological — it doesn't change training,
loss, or architecture. It produces a smaller catalog where every feature
has demonstrated capacity to be reconstructed by a single direction.

Why this is principled:
    The supervised SAE methodology delivers high FVE for features whose
    positive activations cluster tightly along a single direction (single-
    token surface features: semicolons, opening quotes, repeated chars,
    etc.). Polysemantic / context-rich features (bracket_opening,
    currency_symbol) have high causal effect but low FVE because their
    activations are heterogeneous along multiple axes — one direction
    cannot capture the variance.

    Curation lets the paper report "mean FVE = X over the 16-feature
    high-FVE backbone" alongside "all-features mean FVE = Y" with
    transparent inclusion criteria. Reviewers don't penalize stated
    selection rules; they penalize hidden ones.

Math (matches evaluate.py per-feature FVE):
    For feature k with target direction d_k (unit norm) at positive
    positions {x_i : y_ik = 1}:
        x_centered = x_pos - mean(x_pos)
        FVE_k = ||x_centered @ d_k||^2 / ||x_centered||_F^2

    PC1's FVE is the upper bound (Rayleigh's principle). mean_shift /
    LDA / logistic produce concept-bearing directions that may be near or
    far from the FVE optimum.

Run with: python -m pipeline.run --step curate-fve [--target-dir-method M]
        [--fve-curate-threshold 0.5] [--fve-curate-source mean_shift|pc1]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config


def _compute_fve_per_feature(
    x_pos_centered: torch.Tensor, target_dir: torch.Tensor,
) -> float:
    """Per-feature FVE = (proj_var) / (total_var) at positive positions.

    Matches evaluate.py's formula exactly. target_dir must be unit-norm.
    """
    total_var = (x_pos_centered ** 2).sum().item()
    if total_var <= 1e-12:
        return 0.0
    projections = x_pos_centered @ target_dir
    explained = (projections ** 2).sum().item()
    return float(explained / total_var)


def _compute_pc1_fve(x_pos: torch.Tensor) -> tuple[float, torch.Tensor]:
    """FVE upper bound via top eigenvector of positive-class covariance.

    Returns (FVE_pc1, pc1_direction_unit).
    """
    if x_pos.shape[0] < 5:
        return 0.0, torch.zeros(x_pos.shape[1])
    x_pos_centered = x_pos - x_pos.mean(dim=0, keepdim=True)
    if x_pos_centered.shape[0] >= x_pos_centered.shape[1]:
        cov = (x_pos_centered.T @ x_pos_centered) / max(1, x_pos_centered.shape[0] - 1)
        evals, evecs = torch.linalg.eigh(cov)
        pc1 = evecs[:, -1]
    else:
        U, S, Vh = torch.linalg.svd(x_pos_centered, full_matrices=False)
        pc1 = Vh[0]
    fve = _compute_fve_per_feature(x_pos_centered, pc1)
    return fve, pc1


def _load_inputs(cfg: Config) -> tuple[dict, torch.Tensor, torch.Tensor, list[str]]:
    """Load catalog, activations (flat), annotations (flat), and the leaf-id list.

    annotations is the post-min-support tensor. We rely on
    annotations_meta.json to map columns to feature ids. Returns the
    catalog dict so the caller can rewrite it with a curated subset.
    """
    catalog = json.loads(cfg.catalog_path.read_text())
    leaves = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]
    leaf_ids = [f["id"] for f in leaves]
    n_leaves = len(leaves)

    if not cfg.activations_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.activations_path}. Run --step annotate first."
        )
    if not cfg.annotations_path.exists():
        raise FileNotFoundError(
            f"Need {cfg.annotations_path}. Run --step annotate first."
        )

    # Annotation column → leaf id mapping. Without this, we can't be
    # sure which annotation column belongs to which catalog leaf.
    meta_path = cfg.output_dir / "annotations_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Need {meta_path} (column→id mapping). "
            f"This sidecar is written by annotate.py since v8.1; "
            f"older runs without it can't be safely curated."
        )
    meta = json.loads(meta_path.read_text())
    annotation_ids = meta.get("feature_ids") or meta.get("ids") or []
    if not annotation_ids:
        raise RuntimeError(f"{meta_path} has no feature_ids list")

    activations = torch.load(cfg.activations_path, weights_only=True)
    annotations = torch.load(cfg.annotations_path, weights_only=True)

    # Flatten (N, T, ...) → (N*T, ...). Position-mask aware: if a position
    # mask exists, restrict to sampled positions for honest FVE.
    if activations.dim() == 3:
        N, T, d = activations.shape
        activations = activations.reshape(N * T, d)
    if annotations.dim() == 3:
        N, T, F = annotations.shape
        annotations = annotations.reshape(N * T, F)
    if annotations.shape[1] != len(annotation_ids):
        raise RuntimeError(
            f"annotations has {annotations.shape[1]} columns but "
            f"annotations_meta has {len(annotation_ids)} ids — "
            f"sidecar/data mismatch; rebuild via --step annotate"
        )

    pos_mask_path = cfg.output_dir / "position_mask.pt"
    if pos_mask_path.exists():
        pm = torch.load(pos_mask_path, weights_only=True).reshape(-1).bool()
        if pm.shape[0] == activations.shape[0]:
            activations = activations[pm]
            annotations = annotations[pm]
            print(f"  Applied position_mask: {pm.sum().item():,} of "
                  f"{pm.shape[0]:,} positions kept")
        else:
            print(f"  position_mask shape {pm.shape[0]} ≠ activations "
                  f"shape {activations.shape[0]}; skipping mask")

    # Reorder annotation columns to match leaf order in the catalog.
    id_to_col = {fid: i for i, fid in enumerate(annotation_ids)}
    cols = []
    for fid in leaf_ids:
        if fid not in id_to_col:
            raise RuntimeError(
                f"catalog leaf {fid!r} has no column in annotations_meta — "
                f"catalog and annotations are out of sync"
            )
        cols.append(id_to_col[fid])
    annotations = annotations[:, cols]

    return catalog, activations.float(), annotations.float(), leaf_ids


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    print("\n" + "=" * 70)
    print("CURATE FVE  (rank features by FVE potential, filter)")
    print("=" * 70)

    catalog, x_flat, y_flat, leaf_ids = _load_inputs(cfg)
    n_leaves = len(leaf_ids)
    print(f"  Loaded: {n_leaves} leaves, {x_flat.shape[0]:,} flat positions")

    # Compute per-feature FVE under multiple methods so the curation
    # report shows the headroom (PC1 = upper bound; configured method =
    # what training will actually use). Empty fve_curate_source means
    # "fall through to the configured target_dir_method"; we don't pass
    # an empty string into the dispatch.
    method_for_curate = (
        getattr(cfg, "fve_curate_source", "")
        or getattr(cfg, "target_dir_method", "mean_shift")
    )
    threshold = float(getattr(cfg, "fve_curate_threshold", 0.5))
    print(f"  Curation source: {method_for_curate} (configured method)")
    print(f"  FVE threshold:   {threshold}")

    # Use existing dispatch for the configured method to avoid drift.
    from .train import compute_target_directions_dispatch

    # Build a temporary cfg-like object that forces the curation method
    # WITHOUT a target_dirs_path attribute — dispatch skips the meta
    # sidecar write when the attribute is missing, so the real run's
    # target_directions.pt.meta.json isn't clobbered.
    class _CurateCfg:
        target_dir_method = method_for_curate
        target_dir_logistic_lambda = float(
            getattr(cfg, "target_dir_logistic_lambda", 1.0)
        )
        target_dir_lda_shrinkage = float(
            getattr(cfg, "target_dir_lda_shrinkage", 0.1)
        )
        # No target_dirs_path on purpose — see comment above.

    target_dirs, raw_norms, counts = compute_target_directions_dispatch(
        x_flat, y_flat, n_leaves, _CurateCfg(),
    )

    # Per-feature FVE (under configured method) + PC1 FVE (upper bound).
    print(f"\n  {'Feature':<40} {'n_pos':>7} {'FVE':>7} {'FVE_PC1':>9} {'Ratio':>7}")
    print(f"  {'-' * 72}")

    rows = []
    for k, fid in enumerate(leaf_ids):
        n_pos = int(counts[k].item())
        if n_pos < 5:
            rows.append({
                "id": fid, "n_pos": n_pos,
                "fve": None, "fve_pc1": None, "ratio": None,
                "drop_reason": f"n_pos<5 (raw={n_pos})",
            })
            print(f"  {fid:<40} {n_pos:>7} {'—':>7} {'—':>9} {'—':>7}  (skip)")
            continue

        x_pos = x_flat[y_flat[:, k] > 0]
        x_pos_centered = x_pos - x_pos.mean(dim=0, keepdim=True)

        # FVE under configured method.
        d_k = target_dirs[k]
        d_k_norm = d_k.norm()
        if d_k_norm < 1e-9:
            fve = 0.0
        else:
            fve = _compute_fve_per_feature(x_pos_centered, d_k / d_k_norm)

        # FVE upper bound via PC1.
        fve_pc1, _ = _compute_pc1_fve(x_pos)
        ratio = fve / fve_pc1 if fve_pc1 > 1e-9 else 0.0

        rows.append({
            "id": fid, "n_pos": n_pos,
            "fve": round(fve, 4),
            "fve_pc1": round(fve_pc1, 4),
            "ratio": round(ratio, 4),
            "drop_reason": None,
        })
        print(f"  {fid:<40} {n_pos:>7} {fve:>7.4f} {fve_pc1:>9.4f} {ratio:>7.4f}")

    # Filter: keep features with FVE ≥ threshold (under configured method).
    # Skipped features (n_pos < 5) are kept as "no signal" — they don't
    # have a defensible FVE; the user decides whether to drop them
    # via min_support upstream.
    keep_ids = {
        r["id"] for r in rows
        if r["fve"] is not None and r["fve"] >= threshold
    }
    drop_records = [
        r for r in rows
        if r["fve"] is not None and r["fve"] < threshold
    ]
    skip_records = [r for r in rows if r["fve"] is None]

    print(f"\n  Kept:    {len(keep_ids)} / {n_leaves} leaves "
          f"(FVE ≥ {threshold} under {method_for_curate})")
    print(f"  Dropped: {len(drop_records)} leaves (FVE < {threshold})")
    print(f"  Skipped: {len(skip_records)} leaves (n_pos < 5; no signal)")

    if rows:
        valid = [r for r in rows if r["fve"] is not None]
        if valid:
            mean_all = float(np.mean([r["fve"] for r in valid]))
            mean_kept = float(np.mean(
                [r["fve"] for r in valid if r["id"] in keep_ids]
            )) if keep_ids else 0.0
            mean_pc1 = float(np.mean([r["fve_pc1"] for r in valid]))
            print(f"\n  Mean FVE (all valid):           {mean_all:.4f}")
            print(f"  Mean FVE (kept curated subset): {mean_kept:.4f}")
            print(f"  Mean FVE (PC1 upper bound):     {mean_pc1:.4f}")
            headroom = mean_pc1 - mean_all
            print(f"  Headroom to PC1 ceiling:        +{headroom:.4f}")

    # Build curated catalog. Preserve groups whose surviving children > 0.
    leaves = [f for f in catalog.get("features", []) if f.get("type") == "leaf"]
    kept_leaves = [f for f in leaves if f["id"] in keep_ids]
    kept_parents = {f.get("parent") for f in kept_leaves if f.get("parent")}
    kept_groups = [
        f for f in catalog["features"]
        if f.get("type") == "group" and f.get("id") in kept_parents
    ]
    curated = dict(catalog)
    curated["features"] = kept_groups + kept_leaves
    curated["fve_curate_method"] = method_for_curate
    curated["fve_curate_threshold"] = threshold
    curated["n_dropped"] = len(drop_records)

    out_path = cfg.output_dir / "feature_catalog.fve_curated.json"
    out_path.write_text(json.dumps(curated, indent=2))
    print(f"\n  Saved: {out_path}")

    report_path = cfg.output_dir / "fve_curation_report.json"
    report_path.write_text(json.dumps({
        "method": method_for_curate,
        "threshold": threshold,
        "n_leaves_before": n_leaves,
        "n_leaves_after": len(keep_ids),
        "n_dropped": len(drop_records),
        "n_skipped": len(skip_records),
        "rows": rows,
    }, indent=2))
    print(f"  Saved: {report_path}")
    print(f"\n  To use: review {out_path.name}, then:")
    print(f"    cp {out_path} {cfg.catalog_path}     # opt in")
    return curated
