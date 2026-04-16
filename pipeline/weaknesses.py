"""
Weakness Spotlight — Per-Feature Diagnostic

Reads pipeline_data/evaluation.json (+ optional causal.json) and produces
a sorted list of the most-broken features with failure reasons.

Failure modes detected:
  - classifier_broken   : calibrated F1 below threshold
  - direction_useless   : FVE below threshold (decoder direction no variance)
  - direction_misaligned: cosine to target below threshold (learned decoder only)
  - causally_dead       : mean_kl below threshold (from causal.json)
  - data_starved        : n_positives below threshold
  - never_fires         : fire_rate_calibrated = 0
  - over_fires          : fire_rate_calibrated >> gt_pos_rate (false positives)

Outputs:
    pipeline_data/weaknesses.json

Usage:
    python -m pipeline.run --step weaknesses
"""

import json
from pathlib import Path

from .config import Config


THRESHOLDS = {
    "classifier_broken": 0.30,   # cal F1 below this
    "direction_useless": 0.05,   # FVE below this
    "direction_misaligned": 0.50,  # cosine below this
    "causally_dead": 0.05,       # mean_kl below this
    "data_starved": 10,          # n_positives below this
    "over_fire_ratio": 3.0,      # fire_rate_cal / gt_pos_rate above this
}

SCORES = {
    "classifier_broken": 2,
    "direction_useless": 1,
    "direction_misaligned": 1,
    "causally_dead": 2,
    "data_starved": 3,
    "never_fires": 3,
    "over_fires": 1,
}


def diagnose(feature: dict, mse_feat: dict | None, kl_val: float | None) -> tuple[int, list[str]]:
    """Return (brokenness_score, reasons) for a feature."""
    score = 0
    reasons = []

    n_pos = feature.get("n_positives", 0) or 0
    cal_f1 = feature.get("cal_f1")
    fire_cal = feature.get("fire_rate_calibrated")
    gt_rate = feature.get("gt_positive_rate")
    fve = mse_feat.get("fve") if mse_feat else None
    cosine = mse_feat.get("cosine") if mse_feat else None

    if n_pos < THRESHOLDS["data_starved"]:
        score += SCORES["data_starved"]
        reasons.append(f"data_starved(n_pos={n_pos})")

    if cal_f1 is not None and cal_f1 < THRESHOLDS["classifier_broken"]:
        score += SCORES["classifier_broken"]
        reasons.append(f"classifier_broken(cal_f1={cal_f1:.2f})")

    if fire_cal is not None and fire_cal == 0.0 and n_pos > 0:
        score += SCORES["never_fires"]
        reasons.append("never_fires")
    elif (
        fire_cal is not None and gt_rate is not None and gt_rate > 0
        and fire_cal / gt_rate > THRESHOLDS["over_fire_ratio"]
    ):
        score += SCORES["over_fires"]
        reasons.append(
            f"over_fires(r@cal={fire_cal:.4f}, r@gt={gt_rate:.4f}, "
            f"ratio={fire_cal/gt_rate:.1f}x)"
        )

    if fve is not None and fve < THRESHOLDS["direction_useless"]:
        score += SCORES["direction_useless"]
        reasons.append(f"direction_useless(fve={fve:.3f})")

    if cosine is not None and cosine < THRESHOLDS["direction_misaligned"]:
        score += SCORES["direction_misaligned"]
        reasons.append(f"direction_misaligned(cos={cosine:.2f})")

    if kl_val is not None and kl_val < THRESHOLDS["causally_dead"]:
        score += SCORES["causally_dead"]
        reasons.append(f"causally_dead(mean_kl={kl_val:.4f})")

    return score, reasons


def run(cfg: Config = None):
    """Produce weakness report from evaluation.json + causal.json."""
    if cfg is None:
        cfg = Config()

    if not cfg.eval_path.exists():
        raise FileNotFoundError(
            f"evaluation.json not found at {cfg.eval_path} — run --step evaluate first"
        )

    evaluation = json.loads(cfg.eval_path.read_text())
    features = evaluation["features"]

    # MSE metrics (cosine, FVE) may be missing under pure BCE supervision
    mse_block = evaluation.get("mse_supervision_metrics")
    mse_by_id = {}
    if mse_block and mse_block.get("per_feature"):
        mse_by_id = {f["id"]: f for f in mse_block["per_feature"]}

    # Causal KL per feature (optional)
    causal_by_id = {}
    if cfg.causal_path.exists():
        try:
            causal = json.loads(cfg.causal_path.read_text())
            for entry in causal.get("features", []):
                causal_by_id[entry["id"]] = entry.get("mean_kl")
        except (json.JSONDecodeError, KeyError):
            pass

    # Score every feature
    reports = []
    for feat in features:
        mse_feat = mse_by_id.get(feat["id"])
        kl_val = causal_by_id.get(feat["id"])
        score, reasons = diagnose(feat, mse_feat, kl_val)

        reports.append({
            "id": feat["id"],
            "type": feat.get("type", "leaf"),
            "brokenness": score,
            "reasons": reasons,
            "cal_f1": feat.get("cal_f1"),
            "n_positives": feat.get("n_positives"),
            "fire_rate_calibrated": feat.get("fire_rate_calibrated"),
            "gt_positive_rate": feat.get("gt_positive_rate"),
            "fve": mse_feat.get("fve") if mse_feat else None,
            "cosine": mse_feat.get("cosine") if mse_feat else None,
            "mean_kl": kl_val,
        })

    # Sort worst-first
    reports.sort(key=lambda r: (-r["brokenness"], -(r.get("n_positives") or 0)))

    # Summary by category
    category_counts = {}
    for r in reports:
        for reason in r["reasons"]:
            cat = reason.split("(")[0]
            category_counts[cat] = category_counts.get(cat, 0) + 1

    print("=" * 70)
    print("WEAKNESS SPOTLIGHT")
    print("=" * 70)
    print(f"  Total features: {len(reports)}")
    broken = sum(1 for r in reports if r["brokenness"] > 0)
    print(f"  Features with any weakness: {broken}/{len(reports)}")
    print(f"  Causal data: {'yes' if causal_by_id else 'no'}  |  "
          f"MSE metrics: {'yes' if mse_by_id else 'no'}")

    print("\n  Weakness categories (feature count):")
    for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<24} {count}")

    print("\n  Most-broken features (sorted by brokenness score):")
    print(f"  {'Rank':>4} {'Score':>5} {'Feature':<36} {'Reasons'}")
    print("  " + "-" * 90)
    for i, r in enumerate(reports[:30], 1):
        if r["brokenness"] == 0:
            break
        reasons_str = ", ".join(r["reasons"]) if r["reasons"] else "—"
        print(f"  {i:>4} {r['brokenness']:>5} {r['id']:<36} {reasons_str}")

    # Healthy features (for contrast)
    healthy = [r for r in reports if r["brokenness"] == 0]
    if healthy:
        print(f"\n  Fully healthy features: {len(healthy)}")
        for r in healthy[:10]:
            parts = [f"cal_f1={r['cal_f1']:.2f}"] if r["cal_f1"] is not None else []
            if r["fve"] is not None:
                parts.append(f"fve={r['fve']:.2f}")
            if r["mean_kl"] is not None:
                parts.append(f"kl={r['mean_kl']:.3f}")
            print(f"    {r['id']:<36} {'  '.join(parts)}")

    # Save JSON
    out = {
        "thresholds": THRESHOLDS,
        "scoring": SCORES,
        "summary": {
            "total": len(reports),
            "n_broken": broken,
            "n_healthy": len(healthy),
            "categories": category_counts,
            "has_causal": bool(causal_by_id),
            "has_mse_metrics": bool(mse_by_id),
        },
        "features": reports,
    }
    cfg.weaknesses_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Report saved: {cfg.weaknesses_path}")
    return out


if __name__ == "__main__":
    run()
