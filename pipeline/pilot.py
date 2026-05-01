"""
Pilot gate (v8.19.2) — go/no-go before the full Delphi-vs-Opus run.

Two-arm sequential pilot:

  SUP arm  (output_dir = pipeline_data_pilot/):
    1. shortlist_latents → latent_shortlist.json
    2. opus_catalog      → feature_catalog.json (Opus, ~50 features)
    3. annotate          → annotations.pt against Opus labels
    4. train             → supervised_sae.pt
    5. evaluate          → evaluation.json (sup F1)

  UNSUP arm (output_dir = pipeline_data_pilot_unsup/):
    Symlinks shortlist + tokens.pt + activations.pt from sup arm
    (same corpus tokens; recomputing them is wasteful).
    1. delphi_runner     → feature_catalog.json (Delphi, ~30 features)
    2. annotate          → annotations.pt against Delphi labels
    3. unsup_f1          → unsup_f1.json

  COMPARE: prints the headline table.

Aborts the full-run commitment if:
  - any stage fails (raises propagate)
  - sup arm annotation throughput < 200 dec/sec/GPU
  - sup median F1 ≤ unsup median F1 (sign-of-result wrong)
  - training diverges

Wall-clock target: ~2 hr on 4 GPUs.

Run with: python -m pipeline.run --step pilot
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

from .config import Config


def _arm_cfg(parent_cfg: Config, output_dir: Path, n_features_for_arm: int) -> Config:
    """Sandbox a Config for one arm. Inherits scalars from parent, swaps
    output_dir, scales pilot scalars."""
    arm = copy.copy(parent_cfg)
    arm.output_dir = output_dir
    arm.n_sequences = parent_cfg.pilot_n_sequences
    arm.warmup_steps = max(50, parent_cfg.warmup_steps // 10)
    arm.shortlist_calibration_tokens = max(
        500_000, parent_cfg.shortlist_calibration_tokens // 5
    )
    arm.shortlist_size = max(
        100, n_features_for_arm * 3
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return arm


def _symlink(src: Path, dst: Path):
    """Replace dst with a symlink to src (creating parent dirs)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    sup_dir = Path(str(cfg.output_dir) + "_pilot")
    unsup_dir = Path(str(cfg.output_dir) + "_pilot_unsup")
    print("\n" + "=" * 70)
    print("PILOT GATE (two-arm)")
    print("=" * 70)
    print(f"  pilot_n_sequences:  {cfg.pilot_n_sequences}")
    print(f"  pilot_opus_n:       {cfg.pilot_opus_n_features}")
    print(f"  pilot_delphi_n:     {cfg.pilot_delphi_n_features}")
    print(f"  sup_dir:   {sup_dir}")
    print(f"  unsup_dir: {unsup_dir}")

    findings: dict[str, dict] = {}
    abort_reasons: list[str] = []

    # ── SUP ARM ─────────────────────────────────────────────────────
    sup_cfg = _arm_cfg(cfg, sup_dir, cfg.pilot_opus_n_features)
    sup_cfg.opus_n_features = cfg.pilot_opus_n_features
    sup_cfg.delphi_n_features = cfg.pilot_delphi_n_features  # for shortlist sizing

    print("\n" + "─" * 70)
    print(" SUP ARM ")
    print("─" * 70)

    # 1. shortlist
    print("\n[sup 1/5] shortlist")
    t0 = time.time()
    from .shortlist_latents import run as run_shortlist
    run_shortlist(sup_cfg)
    findings["sup_shortlist"] = {"seconds": time.time() - t0}

    # 2. opus-catalog (writes feature_catalog.json directly)
    print("\n[sup 2/5] opus_catalog")
    t0 = time.time()
    try:
        from .opus_catalog import run as run_opus
        opus_catalog = run_opus(sup_cfg)
        n_leaves = sum(
            1 for f in opus_catalog.get("features", [])
            if f.get("type") == "leaf"
        )
        findings["sup_catalog"] = {"seconds": time.time() - t0, "n_features": n_leaves}
    except Exception as e:
        abort_reasons.append(f"opus_catalog failed: {e}")
        findings["sup_catalog"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # 3. annotate
    print("\n[sup 3/5] annotate")
    t0 = time.time()
    try:
        from .annotate import run as run_annotate
        run_annotate(sup_cfg)
        sec = time.time() - t0
        n_decisions = (
            sup_cfg.n_sequences
            * getattr(sup_cfg, "seq_len", 128)
            * n_leaves
        )
        per_gpu = max(1, getattr(sup_cfg, "n_annotation_gpus", 4) or 4)
        dec_per_sec_per_gpu = n_decisions / max(sec, 1e-3) / per_gpu
        findings["sup_annotate"] = {
            "seconds": sec,
            "n_decisions": n_decisions,
            "dec_per_sec_per_gpu": dec_per_sec_per_gpu,
        }
        if dec_per_sec_per_gpu < 200:
            abort_reasons.append(
                f"sup-arm annotation throughput {dec_per_sec_per_gpu:.0f} "
                f"dec/sec/GPU < 200 floor"
            )
    except Exception as e:
        abort_reasons.append(f"sup annotate failed: {e}")
        findings["sup_annotate"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # 4. train
    print("\n[sup 4/5] train")
    t0 = time.time()
    try:
        from .train import run as run_train
        run_train(sup_cfg)
        findings["sup_train"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"sup train failed: {e}")
        findings["sup_train"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # 5. evaluate
    print("\n[sup 5/5] evaluate")
    t0 = time.time()
    try:
        from .evaluate import evaluate as run_evaluate
        run_evaluate(sup_cfg)
        findings["sup_evaluate"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"sup evaluate failed: {e}")
        findings["sup_evaluate"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # ── UNSUP ARM ──────────────────────────────────────────────────
    unsup_cfg = _arm_cfg(cfg, unsup_dir, cfg.pilot_delphi_n_features)
    unsup_cfg.opus_n_features = cfg.pilot_opus_n_features
    unsup_cfg.delphi_n_features = cfg.pilot_delphi_n_features
    unsup_cfg.delphi_run_dir = str(unsup_dir / "delphi_run")

    print("\n" + "─" * 70)
    print(" UNSUP ARM ")
    print("─" * 70)

    # Reuse sup arm's shortlist + tokens + activations (same corpus,
    # avoid re-tokenization + re-activation collection). Also share
    # split_indices.pt so the unsup arm's F1 is on IDENTICAL held-out
    # flat positions to the sup arm's evaluate test set.
    print(f"\n  Symlinking shared artifacts from {sup_dir} → {unsup_dir}")
    for fname in (
        "latent_shortlist.json",
        "tokens.pt",
        "activations.pt",
        "split_indices.pt",
    ):
        src = sup_dir / fname
        dst = unsup_dir / fname
        if src.exists():
            _symlink(src, dst)
            print(f"    symlink: {dst.name} → {src}")

    # 1. delphi-run
    print("\n[unsup 1/3] delphi_runner")
    t0 = time.time()
    try:
        from .delphi_runner import run as run_delphi
        delphi_catalog = run_delphi(unsup_cfg)
        findings["unsup_delphi"] = {
            "seconds": time.time() - t0,
            "n_features": delphi_catalog.get("n_latents_described", 0),
        }
    except Exception as e:
        abort_reasons.append(f"delphi_runner failed: {e}")
        findings["unsup_delphi"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # 2. annotate (Delphi catalog now lives at unsup_dir/feature_catalog.json)
    print("\n[unsup 2/3] annotate")
    t0 = time.time()
    try:
        from .annotate import run as run_annotate
        run_annotate(unsup_cfg)
        findings["unsup_annotate"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"unsup annotate failed: {e}")
        findings["unsup_annotate"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # 3. unsup-f1
    print("\n[unsup 3/3] unsup_f1")
    t0 = time.time()
    try:
        from .unsup_f1 import run as run_unsup_f1
        run_unsup_f1(unsup_cfg)
        findings["unsup_f1"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"unsup_f1 failed: {e}")
        findings["unsup_f1"] = {"failed": str(e)}
        return _finalize(sup_dir, findings, abort_reasons)

    # ── COMPARE ────────────────────────────────────────────────────
    compare_cfg = copy.copy(sup_cfg)
    compare_cfg.unsup_output_dir = str(unsup_dir)
    print("\n" + "─" * 70)
    print(" COMPARE ")
    print("─" * 70)
    try:
        from .compare import run as run_compare
        comp = run_compare(compare_cfg)
        findings["compare"] = comp
        sup = comp.get("sup", {}) or {}
        unsup = comp.get("unsup", {}) or {}
        if isinstance(sup.get("f1_median"), (int, float)) and \
           isinstance(unsup.get("f1_median"), (int, float)):
            if sup["f1_median"] <= unsup["f1_median"]:
                abort_reasons.append(
                    f"F1 directionality wrong: sup median "
                    f"{sup['f1_median']:.3f} ≤ unsup median "
                    f"{unsup['f1_median']:.3f}. Pilot says do not "
                    f"commit to full run; investigate first."
                )
    except Exception as e:
        # Compare is informational; don't abort on its failure.
        findings["compare"] = {"failed": str(e)}

    return _finalize(sup_dir, findings, abort_reasons)


def _finalize(sup_dir: Path, findings: dict, abort_reasons: list[str]) -> dict:
    out = {
        "findings": findings,
        "abort_reasons": abort_reasons,
        "passed": len(abort_reasons) == 0,
    }
    out_path = sup_dir / "pilot_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 70)
    if out["passed"]:
        print("PILOT PASSED — proceed to full run")
    else:
        print("PILOT FAILED — DO NOT proceed; abort reasons:")
        for r in abort_reasons:
            print(f"  - {r}")
    print(f"Report: {out_path}")
    print("=" * 70)
    return out
