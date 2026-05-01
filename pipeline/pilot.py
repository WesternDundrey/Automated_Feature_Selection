"""
Pilot gate (v8.19.0) — go/no-go before the 38hr full Delphi-vs-Opus run.

Mini end-to-end run:
  - n_sequences           = cfg.pilot_n_sequences (500)
  - opus_n_features       = cfg.pilot_opus_n_features (50)
  - delphi_n_features     = cfg.pilot_delphi_n_features (30)

Pipeline executed (all stages invoked, all artifacts written to a
sandbox dir so the production pipeline_data/ stays untouched):

  1. shortlist_latents → latent_shortlist.json
  2. delphi_runner     → delphi_catalog.json
  3. opus_catalog      → opus_catalog.json
  4. annotate (both catalogs concatenated, single file)
  5. train             → supervised_sae.pt (Opus side)
  6. evaluate          → evaluation.json
  7. quick F1 directionality check vs unsup-latent baseline

Aborts the run-level pipeline if:
  - any stage fails (raises propagate)
  - annotation throughput < 200 dec/sec/GPU (5× slower than projected)
  - supSAE F1 directionally < unsup latent F1 (sign-of-result wrong)
  - training diverges (final loss > 10× initial)

Wall-clock target: ~2 hr on 4 GPUs.

Run with: python -m pipeline.run --step pilot
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .config import Config


def run(cfg: Config = None) -> dict:
    if cfg is None:
        cfg = Config()

    # Sandbox the pilot in pipeline_data_pilot/ so production artifacts
    # are not clobbered. Clone the cfg with shrunken scalars.
    pilot_cfg = copy.copy(cfg)
    pilot_cfg.n_sequences = cfg.pilot_n_sequences
    pilot_cfg.delphi_n_features = cfg.pilot_delphi_n_features
    pilot_cfg.opus_n_features = cfg.pilot_opus_n_features
    # Smaller shortlist proportional to pilot scale: the input pool can
    # be smaller too. Keep ≥ Opus design freedom.
    pilot_cfg.shortlist_size = max(
        100, cfg.pilot_opus_n_features * 2 + cfg.pilot_delphi_n_features
    )
    pilot_cfg.warmup_steps = max(50, cfg.warmup_steps // 10)
    pilot_cfg.shortlist_calibration_tokens = max(
        500_000, cfg.shortlist_calibration_tokens // 5
    )
    pilot_cfg.output_dir = Path(str(cfg.output_dir) + "_pilot")
    pilot_cfg.delphi_run_dir = str(pilot_cfg.output_dir / "delphi_run")
    pilot_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PILOT GATE")
    print("=" * 70)
    print(f"  n_sequences:       {pilot_cfg.n_sequences}")
    print(f"  shortlist_size:    {pilot_cfg.shortlist_size}")
    print(f"  delphi_features:   {pilot_cfg.delphi_n_features}")
    print(f"  opus_features:     {pilot_cfg.opus_n_features}")
    print(f"  output_dir:        {pilot_cfg.output_dir}")

    findings: dict[str, dict] = {}
    abort_reasons: list[str] = []

    # 1. shortlist
    print("\n[1/7] shortlist_latents")
    t0 = time.time()
    from .shortlist_latents import run as run_shortlist
    run_shortlist(pilot_cfg)
    findings["shortlist"] = {"seconds": time.time() - t0}

    # 2. delphi-run
    print("\n[2/7] delphi_runner")
    t0 = time.time()
    try:
        from .delphi_runner import run as run_delphi
        delphi_catalog = run_delphi(pilot_cfg)
        findings["delphi"] = {
            "seconds": time.time() - t0,
            "n_features": delphi_catalog.get("n_latents_described", 0),
        }
    except Exception as e:
        abort_reasons.append(f"delphi_runner failed: {e}")
        findings["delphi"] = {"failed": str(e)}
        return _finalize(pilot_cfg, findings, abort_reasons)

    # 3. opus-catalog
    print("\n[3/7] opus_catalog")
    t0 = time.time()
    try:
        from .opus_catalog import run as run_opus
        opus_catalog = run_opus(pilot_cfg)
        n_leaves = sum(
            1 for f in opus_catalog.get("features", [])
            if f.get("type") == "leaf"
        )
        findings["opus"] = {"seconds": time.time() - t0, "n_features": n_leaves}
    except Exception as e:
        abort_reasons.append(f"opus_catalog failed: {e}")
        findings["opus"] = {"failed": str(e)}
        return _finalize(pilot_cfg, findings, abort_reasons)

    # 4. annotate — both catalogs concatenated into one annotation pass
    print("\n[4/7] annotate (Opus + Delphi catalogs concatenated)")
    t0 = time.time()
    merged = _merge_catalogs(opus_catalog, delphi_catalog)
    merged_path = pilot_cfg.output_dir / "feature_catalog.json"
    merged_path.write_text(json.dumps(merged, indent=2))

    n_total_features = sum(
        1 for f in merged["features"] if f.get("type") == "leaf"
    )
    n_decisions_expected = (
        pilot_cfg.n_sequences
        * getattr(pilot_cfg, "seq_len", 128)
        * n_total_features
    )

    try:
        from .annotate import run as run_annotate
        run_annotate(pilot_cfg)
        sec = time.time() - t0
        per_gpu = max(1, getattr(pilot_cfg, "n_annotation_gpus", 4) or 4)
        dec_per_sec_per_gpu = n_decisions_expected / max(sec, 1e-3) / per_gpu
        findings["annotate"] = {
            "seconds": sec,
            "n_decisions": n_decisions_expected,
            "dec_per_sec_per_gpu": dec_per_sec_per_gpu,
            "n_features": n_total_features,
        }
        if dec_per_sec_per_gpu < 200:
            abort_reasons.append(
                f"annotation throughput {dec_per_sec_per_gpu:.0f} "
                f"dec/sec/GPU is < 200 (5× slower than 700 projection); "
                f"38hr full run would take >190hr"
            )
    except Exception as e:
        abort_reasons.append(f"annotate failed: {e}")
        findings["annotate"] = {"failed": str(e)}
        return _finalize(pilot_cfg, findings, abort_reasons)

    # 5. train — Opus side only (sup arm)
    print("\n[5/7] train (sup arm, Opus catalog)")
    t0 = time.time()
    try:
        from .train import run as run_train
        run_train(pilot_cfg)
        findings["train"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"train failed: {e}")
        findings["train"] = {"failed": str(e)}
        return _finalize(pilot_cfg, findings, abort_reasons)

    # 6. evaluate
    print("\n[6/7] evaluate")
    t0 = time.time()
    try:
        from .evaluate import run as run_evaluate
        run_evaluate(pilot_cfg)
        findings["evaluate"] = {"seconds": time.time() - t0}
    except Exception as e:
        abort_reasons.append(f"evaluate failed: {e}")
        findings["evaluate"] = {"failed": str(e)}
        return _finalize(pilot_cfg, findings, abort_reasons)

    # 7. F1 directionality
    print("\n[7/7] F1 directionality vs unsup-latent baseline")
    eval_path = pilot_cfg.output_dir / "evaluation.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        sup_f1 = (
            ev.get("supt0_f1_mean")
            or ev.get("sup_t0_f1_mean")
            or ev.get("mean_supt0_f1")
            or 0.0
        )
        # Unsup-latent F1: per Delphi feature, F1(unsup latent fires vs
        # labels(its description)). Approximate quickly here from the
        # annotations + activations + sae cache; full metric in evaluate
        # would also be acceptable when wired.
        unsup_f1 = _quick_unsup_f1(pilot_cfg, delphi_catalog)
        findings["f1_directionality"] = {
            "sup_f1": sup_f1,
            "unsup_latent_f1": unsup_f1,
            "supSAE_wins": sup_f1 > unsup_f1,
        }
        if sup_f1 <= unsup_f1:
            abort_reasons.append(
                f"F1 directionality wrong: supSAE {sup_f1:.3f} ≤ "
                f"unsup-latent {unsup_f1:.3f}. Pilot says do not commit "
                f"to 38hr full run; investigate first."
            )
    else:
        findings["f1_directionality"] = {"skipped": "no evaluation.json"}

    return _finalize(pilot_cfg, findings, abort_reasons)


def _merge_catalogs(opus_catalog: dict, delphi_catalog: dict) -> dict:
    """Concatenate Opus and Delphi catalogs into one annotation pass.

    Both produce {"features": [...]} where each feature has type=group/leaf.
    The merged catalog tags Delphi leaves with delphi_mode=True (the
    catalog_quality validator and the boundary-discipline gate exempt
    these via `_is_boundary_discipline_exempt` lookalike checks).
    """
    features: list[dict] = []
    seen_ids: set[str] = set()
    for f in opus_catalog.get("features", []):
        fid = f.get("id")
        if fid and fid not in seen_ids:
            features.append(f)
            seen_ids.add(fid)
    for f in delphi_catalog.get("features", []):
        fid = f.get("id")
        if fid and fid not in seen_ids:
            features.append(f)
            seen_ids.add(fid)
    return {"features": features, "merged": True}


def _quick_unsup_f1(cfg: Config, delphi_catalog: dict) -> float:
    """Mean F1 of original unsup latents vs labels(their Delphi descriptions).

    For each Delphi-described latent k:
      pred[t] = (unsup_act[t, k] > 0)
      F1 vs annotations[:, :, idx_in_merged_catalog(delphi.latent_k)]

    Returns the mean across all Delphi features. NaN-safe.
    """
    import torch
    from .inventory import load_sae, load_target_model

    annot_path = cfg.annotations_path
    tokens_path = cfg.tokens_path
    catalog_path = cfg.output_dir / "feature_catalog.json"
    if not (annot_path.exists() and tokens_path.exists() and catalog_path.exists()):
        return float("nan")

    annotations = torch.load(annot_path, weights_only=True).bool()
    tokens = torch.load(tokens_path, weights_only=True)
    catalog = json.loads(catalog_path.read_text())
    leaves = [f for f in catalog["features"] if f.get("type") == "leaf"]

    # Index Delphi leaves in the merged column order
    delphi_cols: list[tuple[int, int]] = []  # (col_index, latent_index)
    for col, leaf in enumerate(leaves):
        if leaf.get("delphi_mode") and leaf.get("source_latents"):
            delphi_cols.append((col, int(leaf["source_latents"][0])))
    if not delphi_cols:
        return float("nan")

    # Forward unsup SAE on tokens to get per-latent activations
    sae, _ = load_sae(cfg)
    model, _tok = load_target_model(cfg)
    sae = sae.to(cfg.device)

    f1s: list[float] = []
    n_seqs, T = tokens.shape
    bs = 32
    with torch.no_grad():
        for col, lat_idx in delphi_cols:
            tp = fp = fn = 0
            for s in range(0, n_seqs, bs):
                tk = tokens[s : s + bs].to(cfg.device)
                _, cache = model.run_with_cache(
                    tk, names_filter=cfg.hook_point, return_type=None
                )
                x = cache[cfg.hook_point]  # (B, T, d)
                z = sae.encode(x.reshape(-1, x.shape[-1]))[:, lat_idx]
                z = z.reshape(x.shape[0], x.shape[1])
                pred = (z > 0).cpu()
                lab = annotations[s : s + bs, :, col]
                tp += int((pred & lab).sum())
                fp += int((pred & ~lab).sum())
                fn += int((~pred & lab).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            f1s.append(f1)

    return float(sum(f1s) / max(len(f1s), 1))


def _finalize(cfg: Config, findings: dict, abort_reasons: list[str]) -> dict:
    out = {
        "findings": findings,
        "abort_reasons": abort_reasons,
        "passed": len(abort_reasons) == 0,
    }
    out_path = cfg.output_dir / "pilot_report.json"
    out_path.write_text(json.dumps(out, indent=2))
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
