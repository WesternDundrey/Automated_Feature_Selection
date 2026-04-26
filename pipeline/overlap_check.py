"""
Post-annotation pairwise overlap + subset analysis.

Per the user's design: IoU alone is unstable for rare features. For
rare positives, sampling noise dominates and IoU can be high even
when the underlying labels mean different things. Add P(A|B) and
P(B|A) — if one feature is 95% contained in the other, that's a
subset relation we should flag separately from straight redundancy.

Output:
  - per-pair table of {iou, p_a_given_b, p_b_given_a, support_a,
    support_b, overlap_count}
  - flagged_redundant: pairs with iou > iou_threshold (default 0.8)
    AND min support ≥ min_support (rare features can't be reliably
    flagged as redundant)
  - flagged_subset: pairs where max(P(A|B), P(B|A)) ≥ subset_threshold
    (default 0.95) AND the smaller-support side has support ≥
    min_support — one feature is contained in the other

The `apply_overlap_gates` step suggests drops based on AUROC tiebreaker
(if available from a previous evaluation.json) but defaults to "report
only, don't drop" — the user can review and apply manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def compute_pairwise_overlap(
    annotations: torch.Tensor,
    feature_ids: list[str],
    min_support: int = 30,
) -> dict:
    """Compute pairwise overlap statistics for all feature pairs.

    `annotations` shape: (N, T, n_features), float or bool.
    `feature_ids`: list aligned with last axis.
    `min_support`: features with fewer than this many positives across
        the whole annotated set are excluded from analysis (their
        statistics would be dominated by sampling noise).

    Returns:
        {
            "feature_ids": list[str],
            "support":     list[int],          # positives per feature
            "pairs": [
                {"a": id_a, "b": id_b,
                 "support_a": int, "support_b": int,
                 "overlap": int,
                 "iou": float,
                 "p_a_given_b": float,
                 "p_b_given_a": float},
                ...
            ],
            "min_support": min_support,
        }
    """
    bool_anns = (annotations > 0).reshape(-1, annotations.shape[-1]).numpy()
    n_features = bool_anns.shape[-1]
    support = bool_anns.sum(axis=0).astype(int).tolist()

    pairs: list[dict] = []
    for i in range(n_features):
        if support[i] < min_support:
            continue
        a = bool_anns[:, i]
        for j in range(i + 1, n_features):
            if support[j] < min_support:
                continue
            b = bool_anns[:, j]
            overlap = int((a & b).sum())
            union = int((a | b).sum())
            if union == 0:
                continue
            iou = overlap / union
            p_a_given_b = overlap / support[j] if support[j] > 0 else 0.0
            p_b_given_a = overlap / support[i] if support[i] > 0 else 0.0
            pairs.append({
                "a": feature_ids[i],
                "b": feature_ids[j],
                "support_a": support[i],
                "support_b": support[j],
                "overlap": overlap,
                "iou": round(iou, 4),
                "p_a_given_b": round(p_a_given_b, 4),
                "p_b_given_a": round(p_b_given_a, 4),
            })

    return {
        "feature_ids": feature_ids,
        "support": support,
        "pairs": pairs,
        "min_support": min_support,
    }


def find_redundant_and_subset_pairs(
    overlap: dict,
    iou_threshold: float = 0.8,
    subset_threshold: float = 0.95,
) -> tuple[list[dict], list[dict]]:
    """Partition pairs into redundant (high IoU, both ways high overlap)
    vs subset (one direction much higher than the other).

    Returns (redundant_pairs, subset_pairs).
    """
    redundant: list[dict] = []
    subset: list[dict] = []
    for pair in overlap["pairs"]:
        if pair["iou"] >= iou_threshold:
            redundant.append({**pair, "flag": "redundant"})
            continue
        max_p = max(pair["p_a_given_b"], pair["p_b_given_a"])
        min_p = min(pair["p_a_given_b"], pair["p_b_given_a"])
        if max_p >= subset_threshold and min_p < subset_threshold:
            # one direction is much higher → subset relation
            if pair["p_a_given_b"] >= subset_threshold:
                # B is mostly contained in A. A is the broader one.
                subset.append({
                    **pair,
                    "flag": "subset",
                    "broader": pair["a"],
                    "narrower": pair["b"],
                })
            else:
                subset.append({
                    **pair,
                    "flag": "subset",
                    "broader": pair["b"],
                    "narrower": pair["a"],
                })
    return redundant, subset


def write_overlap_report(
    overlap: dict,
    redundant: list[dict],
    subset: list[dict],
    out_path: Path,
) -> dict:
    """Persist the analysis to disk and print a human-readable summary."""
    summary = {
        "n_features_analyzed": len([
            s for s in overlap["support"] if s >= overlap["min_support"]
        ]),
        "n_features_total":    len(overlap["feature_ids"]),
        "min_support":         overlap["min_support"],
        "n_pairs_redundant":   len(redundant),
        "n_pairs_subset":      len(subset),
        "redundant_pairs":     redundant,
        "subset_pairs":        subset,
        "all_pairs_top_50":    sorted(
            overlap["pairs"],
            key=lambda p: -max(p["iou"], p["p_a_given_b"], p["p_b_given_a"]),
        )[:50],
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n  [overlap-check] STATUS=scored")
    print(f"    n features analyzed (support >= {overlap['min_support']}): "
          f"{summary['n_features_analyzed']}/{summary['n_features_total']}")
    print(f"    redundant pairs (IoU >= 0.8):       {len(redundant)}")
    print(f"    subset    pairs (P >= 0.95 oneway): {len(subset)}")
    print(f"  Report: {out_path}")

    if redundant:
        print(f"\n  Top redundant pairs:")
        for r in sorted(redundant, key=lambda p: -p["iou"])[:10]:
            print(f"    IoU={r['iou']:.3f}  "
                  f"|A|={r['support_a']:>5}  "
                  f"|B|={r['support_b']:>5}  "
                  f"{r['a']:<35} ↔ {r['b']:<35}")
    if subset:
        print(f"\n  Top subset pairs (broader → narrower):")
        for s in sorted(
            subset, key=lambda p: -max(p["p_a_given_b"], p["p_b_given_a"]),
        )[:10]:
            max_p = max(s["p_a_given_b"], s["p_b_given_a"])
            print(f"    P={max_p:.3f}  "
                  f"|broader|={s['support_a']:>5}  "
                  f"|narrower|={s['support_b']:>5}  "
                  f"{s['broader']:<35} ⊇ {s['narrower']:<35}")

    return summary


def run_post_annotation_overlap_check(
    cfg,
    iou_threshold: float = 0.8,
    subset_threshold: float = 0.95,
    min_support: int = 30,
) -> Optional[dict]:
    """Run the overlap analysis on cached annotations.pt + catalog.
    Writes the report to cfg.output_dir/overlap_check.json. Returns
    the summary dict (or None if data is missing)."""
    if not cfg.annotations_path.exists() or not cfg.catalog_path.exists():
        print(f"  [overlap-check] skipped: annotations.pt or catalog missing")
        return None

    annotations = torch.load(cfg.annotations_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    feature_ids = [f["id"] for f in catalog["features"]]

    if annotations.shape[-1] != len(feature_ids):
        print(f"  [overlap-check] skipped: annotations have "
              f"{annotations.shape[-1]} features, catalog has "
              f"{len(feature_ids)}; ID alignment broken.")
        return None

    print(f"\n  [overlap-check] computing pairwise overlap on "
          f"{annotations.shape[0]} sequences × "
          f"{annotations.shape[1]} positions × "
          f"{len(feature_ids)} features (min_support={min_support})...")
    overlap = compute_pairwise_overlap(annotations, feature_ids, min_support)
    redundant, subset = find_redundant_and_subset_pairs(
        overlap, iou_threshold=iou_threshold, subset_threshold=subset_threshold,
    )
    out_path = cfg.output_dir / "overlap_check.json"
    return write_overlap_report(overlap, redundant, subset, out_path)
