"""
Catalog trim by inter-annotator κ.

Reads `agreement.json` (per-feature Cohen's κ and annotator-vs-annotator
F1 ceiling, produced by `--step agreement`) and `evaluation.json`
(per-feature classification metrics), produces:

    1. A trimmed catalog at `pipeline_data/feature_catalog.trimmed.json`
       containing only features whose κ ≥ `kappa_threshold` (default 0.4).
    2. A printed before/after summary showing what mean cal_F1 / val_promo_f1
       would be on the kept set vs the full catalog — using the existing
       per-feature numbers from evaluation.json (so no retrain required).
    3. A list of the worst-dropped features so the user can sanity-check
       the trim isn't removing things they care about.

Optionally, with `--apply-trim`, replaces `feature_catalog.json` with the
trimmed version (after backing the original up to
`feature_catalog.before_trim.json`). After applying, downstream training
artifacts (`supervised_sae.pt`, `target_directions.pt`, etc.) become
stale and the user should re-run train + evaluate to get a *true*
trimmed-catalog SAE — not just a re-aggregation of the full-catalog SAE's
metrics filtered to the kept features. The latter is the headline number
you typically want for the paper ("on the κ ≥ 0.4 subset, cal_F1 = X");
the former is the more rigorous experiment ("an SAE trained only on the
trimmed catalog reaches cal_F1 = Y").

Both numbers are useful, and they often disagree by 0.01–0.03 — the
trimmed-SAE can do slightly better because it's not spending capacity
on the dropped features. We report the cheap re-aggregation here; pass
`--apply-trim` and rerun training for the rigorous version.

Usage:
    # 1. Run agreement first (writes pipeline_data/agreement.json):
    python -m pipeline.run --step agreement \\
        --layer 9 --sae_id blocks.9.hook_resid_pre --local-annotator

    # 2. Trim and report (no catalog change):
    python -m pipeline.run --step trim-by-kappa --kappa-threshold 0.4

    # 3. Optionally apply, then retrain:
    python -m pipeline.run --step trim-by-kappa --kappa-threshold 0.4 --apply-trim
    python -m pipeline.run --step train --layer 9 --sae_id blocks.9.hook_resid_pre
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def _build_kappa_lookup(agreement: dict) -> dict[str, float | None]:
    """{feature_id: kappa}. None when no_data / undefined."""
    out: dict[str, float | None] = {}
    for f in agreement.get("features", []):
        out[f["id"]] = f.get("kappa")
    return out


def _build_f1_ceiling_lookup(agreement: dict) -> dict[str, float | None]:
    """{feature_id: annotator-vs-annotator f1_ceiling}. v8.1+ field."""
    out: dict[str, float | None] = {}
    for f in agreement.get("features", []):
        out[f["id"]] = f.get("f1_ceiling")
    return out


def _filter_metric(
    per_feature_results: list[dict],
    kept_ids: set[str],
    field: str,
    require_n_pos_field: str | None = None,
) -> list[float]:
    """Pull a per-feature metric from the eval.json features list, filtered
    to kept_ids. require_n_pos_field, if set, requires that feature have
    a positive value in that field (e.g., val_promo_n_pos > 0)."""
    out = []
    for r in per_feature_results:
        if r.get("id") not in kept_ids:
            continue
        if require_n_pos_field is not None:
            n_pos = r.get(require_n_pos_field, 0)
            if not n_pos or n_pos <= 0:
                continue
        v = r.get(field)
        if v is None:
            continue
        out.append(float(v))
    return out


def run(
    cfg: Config = None,
    kappa_threshold: float = 0.4,
    apply_to_disk: bool = False,
) -> dict:
    if cfg is None:
        cfg = Config()

    if not cfg.agreement_path.exists():
        raise FileNotFoundError(
            f"Agreement data not found at {cfg.agreement_path}. Run "
            f"`python -m pipeline.run --step agreement` first to generate "
            f"per-feature κ + F1-ceiling stats; trim-by-kappa reads them "
            f"from there."
        )

    if not cfg.catalog_path.exists():
        raise FileNotFoundError(
            f"Feature catalog not found at {cfg.catalog_path}."
        )

    agreement = json.loads(cfg.agreement_path.read_text())
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]

    kappa_by_id = _build_kappa_lookup(agreement)
    f1_ceil_by_id = _build_f1_ceiling_lookup(agreement)

    # Partition features by κ.
    kept: list[dict] = []
    dropped: list[dict] = []
    no_data: list[dict] = []
    for f in features:
        kappa = kappa_by_id.get(f["id"])
        if kappa is None:
            # Either no data (rare features the agreement run didn't see
            # enough positives for) or the agreement step skipped them.
            # Default behavior: KEEP (better to be inclusive than to drop
            # for missing data; user can pass a stricter threshold to
            # trim them too).
            no_data.append(f)
            kept.append(f)
        elif kappa >= kappa_threshold:
            kept.append(f)
        else:
            dropped.append(f)

    kept_ids = {f["id"] for f in kept}
    dropped_ids = {f["id"] for f in dropped}

    print("=" * 70)
    print(f"CATALOG TRIM BY κ  (threshold ≥ {kappa_threshold})")
    print("=" * 70)
    print(f"  total features:        {len(features)}")
    print(f"  kept (κ ≥ {kappa_threshold} or no κ data): {len(kept)}")
    print(f"  dropped (κ < {kappa_threshold}):    {len(dropped)}")
    print(f"  no κ data (kept):      {len(no_data)}")

    # ── List of worst-dropped features so the user can audit ──
    if dropped:
        dropped_sorted = sorted(
            dropped,
            key=lambda f: kappa_by_id.get(f["id"], 0.0) or 0.0,
        )
        print("\n  Worst-κ features being dropped:")
        for f in dropped_sorted[:15]:
            kappa = kappa_by_id.get(f["id"])
            ceil = f1_ceil_by_id.get(f["id"])
            ceil_str = f"f1_ceil={ceil:.3f}" if ceil is not None else "f1_ceil=?"
            print(f"    κ={kappa:.3f}  {ceil_str}  {f['id']}")

    # ── Re-aggregate eval metrics over the kept set ──
    if not cfg.eval_path.exists():
        print(f"\n  WARNING: {cfg.eval_path} not found — can't re-aggregate "
              f"cal_F1 / val_promo_f1 over the kept set. Run `--step evaluate` "
              f"first to get the projected post-trim numbers.")
        projected = None
    else:
        eval_data = json.loads(cfg.eval_path.read_text())
        per_feat = eval_data.get("features") or []

        full_cal_f1 = _filter_metric(per_feat, set(f["id"] for f in features), "cal_f1")
        kept_cal_f1 = _filter_metric(per_feat, kept_ids, "cal_f1")
        dropped_cal_f1 = _filter_metric(per_feat, dropped_ids, "cal_f1")

        full_val_promo = _filter_metric(
            per_feat, set(f["id"] for f in features),
            "val_promo_f1", require_n_pos_field="val_promo_n_pos",
        )
        kept_val_promo = _filter_metric(
            per_feat, kept_ids,
            "val_promo_f1", require_n_pos_field="val_promo_n_pos",
        )

        full_auroc = _filter_metric(per_feat, set(f["id"] for f in features), "auroc")
        kept_auroc = _filter_metric(per_feat, kept_ids, "auroc")

        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        m_full_cal = _mean(full_cal_f1)
        m_kept_cal = _mean(kept_cal_f1)
        m_drop_cal = _mean(dropped_cal_f1)
        m_full_val = _mean(full_val_promo)
        m_kept_val = _mean(kept_val_promo)
        m_full_auroc = _mean(full_auroc)
        m_kept_auroc = _mean(kept_auroc)

        print("\n  Re-aggregated metrics (filtered from existing evaluation.json):")
        print(f"    {'metric':<30} {'full':>10} {'trimmed':>10} {'Δ':>10}")
        print(f"    {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

        def _row(name, full, kept):
            if full is None or kept is None:
                return
            delta = kept - full
            print(f"    {name:<30} {full:>10.4f} {kept:>10.4f} {delta:>+10.4f}")

        _row("Mean cal_F1 (test)", m_full_cal, m_kept_cal)
        _row("Mean val_promo_F1 (honest)", m_full_val, m_kept_val)
        _row("Mean AUROC", m_full_auroc, m_kept_auroc)

        print(f"\n    Mean cal_F1 of dropped features: "
              f"{m_drop_cal:.4f}" if m_drop_cal is not None else
              f"\n    (no dropped-feature cal_F1 data)")

        # F1 ceiling over kept set — if we can show the SAE is approaching
        # the annotator's self-consistency on the kept features, that's
        # the strongest possible "we hit the ceiling" claim.
        kept_ceilings = [
            f1_ceil_by_id[fid] for fid in kept_ids
            if fid in f1_ceil_by_id and f1_ceil_by_id[fid] is not None
        ]
        if kept_ceilings:
            mean_ceiling = sum(kept_ceilings) / len(kept_ceilings)
            print(f"    Mean annotator-self F1 ceiling (kept): {mean_ceiling:.4f}")
            if m_kept_cal is not None:
                gap = mean_ceiling - m_kept_cal
                print(f"    Gap to ceiling (kept):                {gap:+.4f}  "
                      f"({'SAE within annotator noise' if gap < 0.05 else 'room above noise'})")

        projected = {
            "full_mean_cal_f1": m_full_cal,
            "kept_mean_cal_f1": m_kept_cal,
            "delta_mean_cal_f1": (m_kept_cal - m_full_cal) if (m_kept_cal is not None and m_full_cal is not None) else None,
            "full_mean_val_promo_f1": m_full_val,
            "kept_mean_val_promo_f1": m_kept_val,
            "full_mean_auroc": m_full_auroc,
            "kept_mean_auroc": m_kept_auroc,
            "kept_mean_f1_ceiling": (
                sum(kept_ceilings) / len(kept_ceilings) if kept_ceilings else None
            ),
        }

    # ── Write trimmed catalog ──
    trimmed_catalog = {"features": kept}
    trimmed_path = cfg.output_dir / "feature_catalog.trimmed.json"
    trimmed_path.write_text(json.dumps(trimmed_catalog, indent=2))
    print(f"\n  Trimmed catalog written: {trimmed_path}  "
          f"({len(kept)} features)")

    # Audit log: who got dropped, who's missing data
    audit_path = cfg.output_dir / "trim_by_kappa_audit.json"
    audit_path.write_text(json.dumps({
        "kappa_threshold": kappa_threshold,
        "n_total": len(features),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "n_no_kappa_data": len(no_data),
        "kept_ids": [f["id"] for f in kept],
        "dropped": [
            {"id": f["id"], "kappa": kappa_by_id.get(f["id"]),
             "f1_ceiling": f1_ceil_by_id.get(f["id"])}
            for f in dropped
        ],
        "no_kappa_data_kept": [f["id"] for f in no_data],
        "projected_metrics": projected if cfg.eval_path.exists() else None,
    }, indent=2))
    print(f"  Audit log written:        {audit_path}")

    # ── Optionally swap the catalog on disk (with backup) ──
    if apply_to_disk:
        backup_path = cfg.output_dir / "feature_catalog.before_trim.json"
        if not backup_path.exists():
            backup_path.write_text(json.dumps(catalog, indent=2))
            print(f"  Backup of pre-trim catalog: {backup_path}")
        cfg.catalog_path.write_text(json.dumps(trimmed_catalog, indent=2))
        print(f"\n  Catalog REPLACED with trimmed version: {cfg.catalog_path}")
        print(f"  IMPORTANT: existing supervised_sae.pt, target_directions.pt, "
              f"split_indices.pt, evaluation.json are now stale relative to "
              f"this catalog. Delete and re-run train + evaluate for the "
              f"rigorous trimmed-SAE numbers.")

    return {
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "trimmed_catalog_path": str(trimmed_path),
        "applied": apply_to_disk,
    }
