"""
Unsupervised-width sweep.

Runs train + promote-loop-triage-only across a set of `n_unsupervised`
values and reports how U capacity affects proposal quality:

  - multi_concept_rate: fraction of described candidates rejected as
    multi_concept (= bundle rate). Should FALL as width grows if the
    bottleneck is U capacity; should STAY FLAT if the bottleneck is
    the proposal method.
  - crisp_per_100: count of crisp descriptions per 100 proposals. Should
    RISE with width if wider U produces more atomic latents.
  - nuisance_rate: fraction caught by the token-diversity nuisance gate.
    Should stay roughly flat across widths.
  - R²: reconstruction quality post-retrain.
  - supervised slice cost: R² − R²(pretrained). Expected to shrink as
    width grows (closer to pretrained-SAE reconstruction budget).

Each width runs under `pipeline_data/usweep/u{N}/` with symlinked
tokens/activations/annotations/catalog so expensive shared artifacts
aren't regenerated. Only `supervised_sae.pt`, `target_directions.pt`,
`split_indices.pt`, `evaluation.json`, and the promote-loop round_00
artifacts are written per-width.

Usage:
    python -m pipeline.run --step usweep --widths 256,512,1024 \\
        --layer 9 --sae_id blocks.9.hook_resid_pre \\
        --local-annotator
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

from .config import Config


# Shared artifacts symlinked from the main output dir into each width's
# subdir. These depend on model/layer/corpus but NOT on n_unsupervised,
# so regenerating them per-width would waste hours of annotation time.
SHARED_ARTIFACTS = (
    "tokens.pt",
    "tokens.pt.meta.json",
    "activations.pt",
    "activations.pt.meta.json",
    "annotations.pt",
    "annotations_meta.json",
    "feature_catalog.json",
)


def _link_shared(src_dir: Path, dst_dir: Path) -> None:
    """Symlink (or copy as fallback) the shared artifacts into dst_dir."""
    import shutil
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


def _derive_cfg(base_cfg: Config, n_unsupervised: int, out_dir: Path) -> Config:
    """Clone base_cfg with new n_unsupervised + output_dir."""
    kwargs = {f.name: getattr(base_cfg, f.name) for f in fields(base_cfg)}
    kwargs["n_unsupervised"] = n_unsupervised
    kwargs["output_dir"] = str(out_dir)
    kwargs["hook_point"] = ""  # force re-derivation in __post_init__
    sub = Config(**kwargs)
    sub.output_dir.mkdir(parents=True, exist_ok=True)

    # Carry over all promote_* runtime knobs (they aren't dataclass
    # fields, so the field-based copy above doesn't include them).
    for attr in dir(base_cfg):
        if attr.startswith("promote_") and not attr.startswith("promote__"):
            setattr(sub, attr, getattr(base_cfg, attr))
    return sub


def _read_metrics(sub_dir: Path, n_unsupervised: int) -> dict:
    """Pull the sweep-relevant numbers from this width's artifacts."""
    out = {"n_unsupervised": n_unsupervised, "output_dir": str(sub_dir)}

    eval_path = sub_dir / "evaluation.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        recon = ev.get("reconstruction") or {}
        out["r2"] = recon.get("r2")
        out["delta_r2_supervised"] = recon.get("delta_r2_supervised")
        pre = ev.get("pretrained_reconstruction") or {}
        out["pretrained_r2"] = pre.get("r2") if pre else None
        # Supervised SAE F1 — both t=0 (the headline geometry-honest
        # number) and calibrated (oracle-like, signal-in-representation).
        out["sup_t0_f1"] = ev.get("mean_f1")
        out["sup_cal_f1"] = ev.get("cal_mean_f1")
        out["sup_auroc"] = ev.get("mean_auroc")
        # Baselines at t=0 — what closes-the-gap looks like.
        probe = ev.get("probe_baseline") or {}
        out["probe_t0_f1"] = probe.get("mean_f1")
        out["probe_cal_f1"] = probe.get("mean_f1_cal")
        post = ev.get("posttrain_baseline") or {}
        out["post_t0_f1"] = post.get("mean_f1")
        out["post_cal_f1"] = post.get("mean_f1_cal")
        # Compatibility: keep the val_promo_mean_f1 key the existing
        # promote-loop summary block consumes.
        out["val_promo_mean_f1"] = ev.get("mean_f1")
        mse_metrics = ev.get("mse_supervision_metrics") or {}
        out["mean_fve"] = mse_metrics.get("mean_fve")
        # Prefer val-promo (honest) if present
        per_feature = ev.get("features") or []
        val_promo = [
            f.get("val_promo_f1") for f in per_feature
            if f.get("val_promo_f1") is not None and f.get("val_promo_n_pos", 0) > 0
        ]
        if val_promo:
            out["val_promo_mean_f1"] = sum(val_promo) / len(val_promo)

    hist_path = sub_dir / "promote_loop" / "history.json"
    round_path = sub_dir / "promote_loop" / "round_00" / "summary.json"
    r0 = None
    if round_path.exists():
        try:
            r0 = json.loads(round_path.read_text())
        except json.JSONDecodeError:
            r0 = None
    if r0 is None and hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text())
            if hist:
                r0 = hist[0]
        except json.JSONDecodeError:
            pass

    if r0:
        spent = r0.get("spent") or 0
        out["spent"] = spent
        out["n_crisp"] = r0.get("n_crisp", 0)
        out["n_nuisance"] = r0.get("n_nuisance_dropped", 0)
        out["crispness_breakdown"] = r0.get("crispness_breakdown", {})
        out["multi_concept_rate"] = r0.get("multi_concept_rate")
        # Derived rates for the comparison table.
        if spent:
            out["nuisance_rate"] = out["n_nuisance"] / spent
            out["crisp_per_100"] = round(100 * out["n_crisp"] / spent, 3)
    return out


def run(
    cfg: Config = None,
    widths: tuple[int, ...] = (256, 512, 1024),
    skip_promote_loop: bool = False,
) -> list[dict]:
    if cfg is None:
        cfg = Config()

    main_output = cfg.output_dir
    sweep_root = main_output / "usweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    # Sanity: the shared artifacts must already exist in main_output.
    required = ["tokens.pt", "activations.pt", "annotations.pt",
                "feature_catalog.json"]
    for name in required:
        if not (main_output / name).exists():
            raise FileNotFoundError(
                f"usweep requires {main_output / name}. Run the main "
                f"pipeline (inventory + annotate) first so the sweep has "
                f"something to train against."
            )

    # Deferred imports so the module is cheap to import.
    from .train import run as run_train
    from .evaluate import evaluate as run_evaluate
    from .promote_loop import run as run_promote_loop

    print("=" * 70)
    print(f"U-WIDTH SWEEP  widths={list(widths)}")
    print(f"  main_output:   {main_output}")
    print(f"  sweep_root:    {sweep_root}")
    print("=" * 70)

    results: list[dict] = []
    for n_unsup in widths:
        print(f"\n{'═' * 70}")
        print(f"  sweep: n_unsupervised = {n_unsup}")
        print(f"{'═' * 70}")

        sub_dir = sweep_root / f"u{n_unsup}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        _link_shared(main_output, sub_dir)

        sub_cfg = _derive_cfg(cfg, n_unsup, sub_dir)

        # Skip train if a checkpoint is already present (resumability).
        if not sub_cfg.checkpoint_path.exists():
            print(f"\n── train @ n_unsup={n_unsup} ──")
            run_train(sub_cfg)
        else:
            print(f"  train: already trained ({sub_cfg.checkpoint_path})")

        # Evaluate so we get R² + val_promo_f1 per-width for the summary.
        if not sub_cfg.eval_path.exists():
            print(f"\n── evaluate @ n_unsup={n_unsup} ──")
            run_evaluate(sub_cfg)
        else:
            print(f"  evaluate: already have {sub_cfg.eval_path}")

        # Promote-loop in triage-only mode (no decomposition, no merge).
        # Just measure how many multi_concept vs crisp we get at this width.
        # Skipped for fixed catalogs (test_catalog etc.) where there's no
        # U→S discovery question — the user just wants train+eval metrics
        # as a function of n_unsupervised.
        if skip_promote_loop:
            print(f"\n── promote-loop SKIPPED @ n_unsup={n_unsup} "
                  f"(--usweep-skip-promote) ──")
        else:
            sub_cfg.promote_triage_only = True
            sub_cfg.promote_decompose_multi_concept = False
            sub_cfg.promote_max_iters = 1
            sub_cfg.promote_min_kept = 10**6  # force triage-only termination
            # Keep budget + batch size from base_cfg (default 100 / 20).
            print(f"\n── promote-loop triage-only @ n_unsup={n_unsup} ──")
            try:
                run_promote_loop(sub_cfg)
            except Exception as e:
                print(f"  promote-loop error at n_unsup={n_unsup}: "
                      f"{type(e).__name__}: {e}")

        m = _read_metrics(sub_dir, n_unsup)
        results.append(m)

    # ── Summary ────────────────────────────────────────────────────────
    summary_path = sweep_root / "summary.json"
    summary_path.write_text(json.dumps(
        {"widths": list(widths), "per_width": results}, indent=2,
    ))

    print("\n" + "=" * 70)
    print("U-WIDTH SWEEP SUMMARY")
    print("=" * 70)
    if skip_promote_loop:
        # Train+eval-only mode: focus on reconstruction parity + the
        # three F1 numbers at t=0 (the headline geometry-honest read).
        cols = [
            ("n_unsupervised", "n_unsup"),
            ("r2", "R²"),
            ("pretrained_r2", "R²(P)"),
            ("delta_r2_supervised", "ΔR²(S)"),
            ("mean_fve", "FVE"),
            ("sup_t0_f1", "supT0"),
            ("probe_t0_f1", "prbT0"),
            ("post_t0_f1", "ptT0"),
            ("sup_cal_f1", "supCal"),
        ]
    else:
        cols = [
            ("n_unsupervised", "n_unsup"),
            ("r2", "R²"),
            ("pretrained_r2", "R²(P)"),
            ("mean_fve", "FVE"),
            ("val_promo_mean_f1", "val_F1"),
            ("spent", "spent"),
            ("n_crisp", "crisp"),
            ("crisp_per_100", "crisp%"),
            ("multi_concept_rate", "mcRate"),
            ("nuisance_rate", "nuisRate"),
        ]
    header = "  " + "  ".join(f"{h:>9}" for _, h in cols)
    print(header)
    print("-" * len(header))
    for row in results:
        parts = []
        for key, _ in cols:
            v = row.get(key)
            if v is None:
                parts.append(f"{'-':>9}")
            elif isinstance(v, float):
                parts.append(f"{v:>9.4f}")
            else:
                parts.append(f"{v:>9}")
        print("  " + "  ".join(parts))
    print(f"\nSaved: {summary_path}")

    return results
