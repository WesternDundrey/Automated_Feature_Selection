"""
Compare sup vs unsup arm — the headline table (v8.19.2).

Reads from:
  - cfg.output_dir/evaluation.json       — sup arm (Opus-trained supSAE)
  - cfg.output_dir/oracle_unsup.json     — Type-2 appendix (optional)
  - cfg.unsup_output_dir/unsup_f1.json   — unsup arm (Delphi auto-interp)
  - cfg.unsup_output_dir/irr_report.json — IRR ceiling (optional)
  - cfg.output_dir/irr_report.json       — IRR ceiling for sup arm (optional)

The "unsup output dir" is by convention `pipeline_data_unsup/` — the
arm-local sandbox where --step delphi-run + --step annotate +
--step unsup-f1 ran. Set explicitly via cfg.unsup_output_dir or
falls back to `pipeline_data_unsup`.

Prints + saves comparison.json with:
  - sup arm headline F1 (median + mean across Opus features)
  - unsup arm headline F1 (median + mean across Delphi features)
  - oracle-1 unsup F1 (Type 2: same Opus catalog, best unsup latent)
  - per-arm IRR ceilings if available
  - Δ headlines: "sup beats unsup by X (median, on respective catalogs)"

Honest framing reminder: sup F1 and unsup F1 are computed on DIFFERENT
catalogs. The Type-1 native-pipeline comparison is the headline; the
Type-2 oracle_unsup appendix is the same-catalog reviewer-bulletproof.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_unsup_dir(cfg: Config) -> Path:
    explicit = getattr(cfg, "unsup_output_dir", None)
    if explicit:
        return Path(explicit)
    sup_dir = Path(cfg.output_dir)
    candidate = sup_dir.parent / f"{sup_dir.name}_unsup"
    if candidate.exists():
        return candidate
    return Path("pipeline_data_unsup")


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    sup_dir = Path(cfg.output_dir)
    unsup_dir = _resolve_unsup_dir(cfg)

    print("\n" + "=" * 70)
    print("COMPARE  (sup vs unsup arm)")
    print("=" * 70)
    print(f"  sup arm:    {sup_dir}")
    print(f"  unsup arm:  {unsup_dir}")

    sup_eval = _load_json(sup_dir / "evaluation.json")
    unsup_f1 = _load_json(unsup_dir / "unsup_f1.json")
    oracle = _load_json(sup_dir / "oracle_unsup.json")
    sup_irr = _load_json(sup_dir / "irr_report.json")
    unsup_irr = _load_json(unsup_dir / "irr_report.json")

    summary: dict = {
        "sup_dir": str(sup_dir),
        "unsup_dir": str(unsup_dir),
    }

    # Sup arm headline. evaluate.py writes mean_f1 / cal_mean_f1 /
    # opt_mean_f1 + a `features` list of per-feature records each with
    # an "f1" key. Compute median directly from those records.
    if sup_eval is not None:
        per_feat = sup_eval.get("features") or []
        sup_f1s = [
            r.get("f1") for r in per_feat
            if r.get("f1") is not None
        ]
        sup_cal_f1s = [
            r.get("f1_cal") for r in per_feat
            if r.get("f1_cal") is not None
        ]
        import numpy as np
        summary["sup"] = {
            "f1_mean": sup_eval.get("mean_f1"),
            "f1_median": float(np.median(sup_f1s)) if sup_f1s else None,
            "cal_f1_mean": sup_eval.get("cal_mean_f1"),
            "cal_f1_median": (
                float(np.median(sup_cal_f1s)) if sup_cal_f1s else None
            ),
            "discovery_f1_mean": sup_eval.get("mean_f1_discovery"),
            "n_features": sup_eval.get("n_total_features"),
            "n_with_f1": len(sup_f1s),
        }
    else:
        summary["sup"] = {"missing": str(sup_dir / "evaluation.json")}

    # Unsup arm headline
    if unsup_f1 is not None:
        summary["unsup"] = {
            "f1_mean": unsup_f1.get("f1_mean"),
            "f1_median": unsup_f1.get("f1_median"),
            "n_features": unsup_f1.get("n_evaluated"),
        }
    else:
        summary["unsup"] = {"missing": str(unsup_dir / "unsup_f1.json")}

    # Type 2 oracle (same Opus labels)
    if oracle is not None:
        summary["oracle_type2"] = {
            "test_f1_mean": oracle.get("test_f1_mean"),
            "test_f1_median": oracle.get("test_f1_median"),
            "selection_bias_val_minus_test": oracle.get(
                "selection_bias_val_minus_test"
            ),
            "n_features": oracle.get("n_evaluated"),
        }

    # IRR ceilings
    if sup_irr is not None:
        arms = sup_irr.get("arms", {})
        if "opus" in arms and arms["opus"].get("agreement_f1_mean") is not None:
            summary.setdefault("irr_ceiling", {})["sup_opus"] = (
                arms["opus"]["agreement_f1_mean"]
            )
    if unsup_irr is not None:
        arms = unsup_irr.get("arms", {})
        if "delphi" in arms and arms["delphi"].get("agreement_f1_mean") is not None:
            summary.setdefault("irr_ceiling", {})["unsup_delphi"] = (
                arms["delphi"]["agreement_f1_mean"]
            )

    # Δ headlines
    s = summary.get("sup", {})
    u = summary.get("unsup", {})
    if isinstance(s.get("f1_median"), (int, float)) and \
       isinstance(u.get("f1_median"), (int, float)):
        summary["delta_median_sup_minus_unsup"] = (
            float(s["f1_median"]) - float(u["f1_median"])
        )

    # Pretty print
    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    print()
    print(f"  Sup arm     (Opus catalog):  median F1 = "
          f"{fmt(s.get('f1_median'))}  mean = {fmt(s.get('f1_mean'))}  "
          f"cal mean = {fmt(s.get('cal_f1_mean'))}  "
          f"n = {fmt(s.get('n_features'))}")
    print(f"  Unsup arm   (Delphi catalog):median F1 = "
          f"{fmt(u.get('f1_median'))}  mean = {fmt(u.get('f1_mean'))}  "
          f"n = {fmt(u.get('n_features'))}")
    if "oracle_type2" in summary:
        o = summary["oracle_type2"]
        print(f"  Oracle-1 (Type 2, Opus labels):  test F1 = "
              f"{fmt(o.get('test_f1_median'))} median  "
              f"{fmt(o.get('test_f1_mean'))} mean   "
              f"selection-bias (val−test) = "
              f"{fmt(o.get('selection_bias_val_minus_test'))}")
    if "irr_ceiling" in summary:
        irr = summary["irr_ceiling"]
        print(f"  IRR ceilings:  sup={fmt(irr.get('sup_opus'))}  "
              f"unsup={fmt(irr.get('unsup_delphi'))}")
    if "delta_median_sup_minus_unsup" in summary:
        print(f"  Δ median (sup − unsup): "
              f"{summary['delta_median_sup_minus_unsup']:+.3f}")

    out_path = sup_dir / "comparison.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
    return summary
