"""
Hinge-family ablation runner.

Per reviewer's ordering (paraphrased from the methodology review):

  1. Diagnose AUROC vs F1 gap  — done in evaluate, just aggregate here.
  2. Frozen decoder + hinge selectivity.     # isolate hinge from free-decoder
  3. Margin hinge for the free-decoder path. # score-shape, not just sign
  4. Squared hinge.                          # more gradient on big violations
  5. Lambda_sup sweep.                       # is supervision under-weighted?
  6. Longer training.                        # does cal_F1 climb past 0.6?
  7. Gated + BCE.                            # full decoupling, reference point

This module runs each variant as its own train + evaluate pair under
`pipeline_data/hinge_ablation/<variant_name>/`, with shared artifacts
(tokens, activations, annotations, catalog) symlinked from the main
output dir so no variant pays for re-annotation. After all variants
finish, it aggregates the headline metrics into a single comparison
table focused on the **AUROC vs F1 gap** — the reviewer's primary
signal for whether the objective is score-shaping-limited (AUROC OK,
F1 behind) or capacity-limited (both behind).

Usage:
    python -m pipeline.run --step hinge-ablation
    python -m pipeline.run --step hinge-ablation \\
        --hinge-ablation-variants hybrid_bce,hinge_margin1 \\
        --layer 9 --sae_id blocks.9.hook_resid_pre --n_sequences 1000

Requires a completed main-run pipeline (tokens, activations, annotations,
feature_catalog on disk) so the ablations can skip inventory + annotate.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path

from .config import Config


# Files symlinked from the main output dir into each variant's subdir.
# These are all run-identical — they depend on corpus/model/layer/catalog
# but NOT on the supervision objective.
SHARED_ARTIFACTS = (
    "tokens.pt",
    "tokens.pt.meta.json",
    "activations.pt",
    "activations.pt.meta.json",
    "annotations.pt",
    "annotations_meta.json",
    "feature_catalog.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# Variant definitions.
# Each dict is applied as **overrides on top of the base Config (which
# carries the user's --layer / --sae_id / --n_sequences / --epochs etc).
# Variants are ordered to match the reviewer's diagnostic ordering.
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = {
    # ───────────── legacy / baseline ─────────────
    "hybrid_bce": {
        "supervision_mode": "hybrid",
        "selectivity_loss": "bce",
        "freeze_supervised_decoder": True,
        "hinge_margin": 1.0,
    },
    # ───────────── reviewer #2: frozen decoder + hinge selectivity ─
    "hybrid_hinge": {
        "supervision_mode": "hybrid",
        "selectivity_loss": "hinge",
        "freeze_supervised_decoder": True,
        "hinge_margin": 1.0,
    },
    # ───────────── current default: free decoder, zero-margin hinge ─
    "hinge_free_zero": {
        "supervision_mode": "hinge",
        "hinge_margin": 0.0,
        "hinge_squared": False,
    },
    # ───────────── reviewer #3: margin hinge for free-decoder path ─
    "hinge_free_margin1": {
        "supervision_mode": "hinge",
        "hinge_margin": 1.0,
        "hinge_squared": False,
    },
    # ───────────── reviewer #4: squared hinge, free decoder ────────
    "hinge_free_margin1_sq": {
        "supervision_mode": "hinge",
        "hinge_margin": 1.0,
        "hinge_squared": True,
    },
    # ───────────── reviewer #5: bump lambda_sup with margin hinge ──
    "hinge_free_margin1_lam10": {
        "supervision_mode": "hinge",
        "hinge_margin": 1.0,
        "hinge_squared": False,
        "lambda_sup": 10.0,
    },
    # ───────────── reviewer #6: longer training ────────────────────
    "hinge_free_margin1_30ep": {
        "supervision_mode": "hinge",
        "hinge_margin": 1.0,
        "hinge_squared": False,
        "epochs": 30,
    },
    # ───────────── reference point: Gated + BCE ────────────────────
    "gated_bce": {
        "supervision_mode": "gated_bce",
    },
}


def _link_shared(src_dir: Path, dst_dir: Path) -> None:
    """Symlink (or copy as fallback) shared artifacts into a variant's subdir."""
    for name in SHARED_ARTIFACTS:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def _derive_cfg(base_cfg: Config, overrides: dict, out_dir: Path) -> Config:
    """Clone base_cfg with variant-specific overrides + new output_dir."""
    kwargs = {f.name: getattr(base_cfg, f.name) for f in fields(base_cfg)}
    kwargs.update(overrides)
    kwargs["output_dir"] = str(out_dir)
    kwargs["hook_point"] = ""  # re-derive in __post_init__
    cfg = Config(**kwargs)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _read_variant_metrics(variant_dir: Path, variant_name: str) -> dict:
    """Extract the reviewer-relevant headline metrics from a variant's
    evaluation.json. Focuses on the AUROC-vs-F1 gap because that's the
    diagnostic ordering the reviewer specified: if AUROC ≈ probe but F1 is
    worse, the issue is score-shaping; if AUROC is also worse, the issue
    is capacity/objective/data."""
    out = {"variant": variant_name, "output_dir": str(variant_dir)}
    eval_path = variant_dir / "evaluation.json"
    if not eval_path.exists():
        out["status"] = "no_eval"
        return out
    try:
        ev = json.loads(eval_path.read_text())
    except json.JSONDecodeError:
        out["status"] = "unparseable_eval"
        return out

    recon = ev.get("reconstruction") or {}
    out["r2"] = recon.get("r2")

    out["mean_f1"] = ev.get("mean_f1")
    out["cal_mean_f1"] = ev.get("cal_mean_f1")
    out["mean_auroc"] = ev.get("mean_auroc")

    # v8.11.1 discovery-only fields (scaffold-filtered) if present
    out["cal_mean_f1_discovery"] = ev.get("cal_mean_f1_discovery")
    out["val_promo_f1_discovery"] = ev.get("val_promo_f1_discovery")

    probe = ev.get("probe_baseline") or {}
    out["probe_cal_f1"] = probe.get("mean_f1_cal")
    out["probe_auroc"] = probe.get("mean_auroc")

    posttrain = ev.get("posttrain_baseline") or {}
    out["pretrained_readout_cal_f1"] = posttrain.get("mean_f1_cal")
    out["pretrained_readout_auroc"] = posttrain.get("mean_auroc")

    pre_recon = ev.get("pretrained_reconstruction") or {}
    out["pretrained_sae_r2"] = pre_recon.get("r2") if pre_recon else None

    # The reviewer's central diagnostic: gaps.
    if out["cal_mean_f1"] is not None and out["probe_cal_f1"] is not None:
        out["gap_cal_f1_vs_probe"] = round(
            float(out["cal_mean_f1"]) - float(out["probe_cal_f1"]), 4,
        )
    if out["mean_auroc"] is not None and out["probe_auroc"] is not None:
        out["gap_auroc_vs_probe"] = round(
            float(out["mean_auroc"]) - float(out["probe_auroc"]), 4,
        )

    dtmetrics = ev.get("decoder_target_dir_metrics") or {}
    out["mean_cos_to_target"] = dtmetrics.get("mean_cosine_to_target")
    out["mean_fve"] = dtmetrics.get("mean_fve")

    out["status"] = "ok"
    return out


def run(
    cfg: Config = None,
    variant_names: tuple[str, ...] | None = None,
) -> list[dict]:
    """Run the listed variants, collect metrics, print comparison.

    If `variant_names` is None, runs all VARIANTS (ordered). The ordering
    matters — later variants may want to skip if earlier ones already
    ruled out a hypothesis.
    """
    if cfg is None:
        cfg = Config()

    if variant_names is None:
        variant_names = tuple(VARIANTS.keys())
    else:
        unknown = [v for v in variant_names if v not in VARIANTS]
        if unknown:
            raise ValueError(
                f"Unknown hinge-ablation variant(s): {unknown}. "
                f"Available: {sorted(VARIANTS.keys())}"
            )

    main_output = cfg.output_dir
    ab_root = main_output / "hinge_ablation"
    ab_root.mkdir(parents=True, exist_ok=True)

    # Pre-flight: shared artifacts must exist in main_output.
    for req in ("tokens.pt", "activations.pt", "annotations.pt", "feature_catalog.json"):
        if not (main_output / req).exists():
            raise FileNotFoundError(
                f"hinge-ablation requires {main_output / req}. Run the main "
                f"pipeline (inventory + annotate) at least once first — the "
                f"ablation reuses those artifacts across every variant."
            )

    from .train import run as run_train
    from .evaluate import evaluate as run_evaluate

    print("=" * 78)
    print(f"HINGE-FAMILY ABLATION  variants={list(variant_names)}")
    print(f"  main_output:   {main_output}")
    print(f"  ablation_root: {ab_root}")
    print("=" * 78)

    results: list[dict] = []
    for name in variant_names:
        overrides = VARIANTS[name]
        print(f"\n{'═' * 78}")
        print(f"  variant: {name}")
        print(f"  overrides: {overrides}")
        print(f"{'═' * 78}")

        var_dir = ab_root / name
        var_dir.mkdir(parents=True, exist_ok=True)
        _link_shared(main_output, var_dir)

        var_cfg = _derive_cfg(cfg, overrides, var_dir)

        # Skip train if checkpoint already there (resumability across partial
        # runs). Skip evaluate if eval already there.
        if not var_cfg.checkpoint_path.exists():
            try:
                run_train(var_cfg)
            except Exception as e:
                print(f"  train FAILED for {name}: {type(e).__name__}: {e}")
                results.append({
                    "variant": name, "status": "train_failed",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
        else:
            print(f"  train: cached at {var_cfg.checkpoint_path}")

        if not var_cfg.eval_path.exists():
            try:
                run_evaluate(var_cfg)
            except Exception as e:
                print(f"  evaluate FAILED for {name}: {type(e).__name__}: {e}")
                results.append({
                    "variant": name, "status": "evaluate_failed",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
        else:
            print(f"  evaluate: cached at {var_cfg.eval_path}")

        results.append(_read_variant_metrics(var_dir, name))

    # ── Aggregate summary, focused on the reviewer's primary diagnostic ──
    summary_path = ab_root / "summary.json"
    summary_path.write_text(json.dumps(
        {"variants": list(variant_names), "per_variant": results}, indent=2,
    ))

    print("\n" + "=" * 78)
    print("HINGE-FAMILY ABLATION SUMMARY")
    print("=" * 78)
    print(
        "  Reviewer primary diagnostic: if AUROC_SAE ≈ AUROC_probe but\n"
        "  cal_F1_SAE < cal_F1_probe, the issue is score-shaping (margin,\n"
        "  λ_sup). If AUROC_SAE also lags, it's capacity/objective/data."
    )
    print()

    cols = [
        ("variant", "variant", 28),
        ("r2", "R²", 7),
        ("pretrained_sae_r2", "R²(P)", 7),
        ("cal_mean_f1", "calF1", 7),
        ("mean_auroc", "AUROC", 7),
        ("probe_cal_f1", "probe_F1", 8),
        ("probe_auroc", "probe_AUROC", 11),
        ("gap_cal_f1_vs_probe", "ΔF1", 7),
        ("gap_auroc_vs_probe", "ΔAUROC", 8),
        ("mean_cos_to_target", "cos", 7),
    ]
    header = "  " + "  ".join(f"{h:>{w}}" for _, h, w in cols)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in results:
        parts = []
        for key, _, w in cols:
            v = row.get(key)
            if v is None:
                parts.append(f"{'-':>{w}}")
            elif isinstance(v, float):
                parts.append(f"{v:>{w}.4f}")
            else:
                parts.append(f"{str(v):>{w}}")
        print("  " + "  ".join(parts))
    print(f"\nSaved: {summary_path}")
    return results
