"""
Layer Sweep — Cross-layer supervised-SAE evaluation orchestrator

Runs the full pipeline (inventory → annotate → train → evaluate → causal
→ intervention) at each specified layer of the base model, storing per-layer
artifacts under `pipeline_data/layer_{N}/`, and aggregates the headline
metrics into a single cross-layer summary.

Two distinct questions this sweep can answer — pick deliberately:

(A) "What catalog does each layer naturally yield?" — the DEFAULT path.
    Per-layer `inventory` is re-run, so Sonnet organizes each layer's own
    pretrained-SAE top activations into a layer-specific catalog. Cross-layer
    F1/R²/FVE deltas conflate "which layer represents the same concepts best"
    with "which layer's catalog is more coherent". Useful for scouting which
    layers the pretrained SAE organizes cleanly.

(B) "Where are the SAME concepts most linearly/causally represented?" — the
    FIXED-CATALOG path. Pass `--catalog` so every layer uses the identical
    feature set; only activations and target_dirs differ. This is the fair
    cross-layer comparison and the right design for a "which layer is best"
    claim.

Both are legitimate; they answer different questions. The cross-layer R²
and FVE numbers this sweep prints are meaningful only if you know which
question you're asking. `layer_sweep_summary.json` records
`mode: "per_layer_catalog"` or `"fixed_catalog"` so downstream plots don't
conflate them.

Caching: each expensive artifact (`evaluation.json`, `causal.json`,
`intervention_precision.json`) is skipped if already present for that
layer. Delete those files to re-run.

Usage — question (A):
    python -m pipeline.run --step layer-sweep --layers 4,6,8,9,10,11 \\
        --local-annotator --full-desc \\
        --n_latents 500 --n_sequences 1000 --epochs 15

Usage — question (B):
    python -m pipeline.run --step layer-sweep --layers 4,6,8,9,10,11 \\
        --catalog pipeline/gpt2_catalog.json \\
        --local-annotator --full-desc \\
        --n_sequences 1000 --epochs 15

The SAE release + sae_id are derived from the current cfg: the `{N}` in
`sae_id` is substituted with the layer number. For `gpt2-small-res-jb`,
layers 0-11 have pretrained SAEs under `blocks.{N}.hook_resid_pre`.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import fields
from pathlib import Path

from .config import Config


DEFAULT_LAYERS_GPT2 = (4, 6, 8, 9, 10, 11)


def _layer_cfg(base_cfg: Config, layer: int, sweep_root: Path) -> Config:
    """Build a fresh Config for this layer, routing output to sweep_root/layer_{N}."""
    kwargs = {f.name: getattr(base_cfg, f.name) for f in fields(base_cfg)}
    kwargs["target_layer"] = layer
    kwargs["hook_point"] = ""  # force re-derive in __post_init__

    # Substitute layer number in the sae_id if it looks like a blocks.{N}.* pattern.
    sae_id = str(kwargs.get("sae_id", ""))
    new_sae_id = re.sub(r"blocks\.\d+\.", f"blocks.{layer}.", sae_id)
    if new_sae_id != sae_id:
        kwargs["sae_id"] = new_sae_id

    kwargs["output_dir"] = str(sweep_root / f"layer_{layer:02d}")
    # Config.__post_init__ converts output_dir to Path, but we pass as str to
    # avoid carrying a Path into the dataclass-replay.
    cfg = Config(**kwargs)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _collect_metrics(cfg: Config) -> dict:
    """Pull headline metrics from a layer's output dir."""
    out: dict = {"layer": cfg.target_layer}

    eval_data = _read_json(cfg.eval_path)
    if eval_data:
        recon = eval_data.get("reconstruction") or {}
        out["r2"] = recon.get("r2")
        out["delta_r2_supervised"] = recon.get("delta_r2_supervised")
        out["delta_r2_unsupervised"] = recon.get("delta_r2_unsupervised")

        # evaluate.py reports three F1 variants: naive (t=0), calibrated, oracle.
        out["naive_f1"] = eval_data.get("mean_f1")
        out["calibrated_f1"] = eval_data.get("cal_mean_f1")
        out["oracle_f1"] = eval_data.get("opt_mean_f1")
        out["mean_auroc"] = eval_data.get("mean_auroc")

        probe = eval_data.get("probe_baseline") or {}
        out["linear_probe_f1"] = probe.get("mean_f1")
        out["linear_probe_f1_cal"] = probe.get("mean_f1_cal")

        posttrain = eval_data.get("posttrain_baseline") or {}
        out["pretrained_readout_f1"] = posttrain.get("mean_f1")
        out["pretrained_readout_f1_cal"] = posttrain.get("mean_f1_cal")

        pre_recon = eval_data.get("pretrained_reconstruction") or {}
        out["pretrained_sae_r2"] = pre_recon.get("r2") if pre_recon else None

        mse_metrics = eval_data.get("mse_supervision_metrics") or {}
        out["mean_cosine_to_target"] = mse_metrics.get("mean_cosine_to_target")
        out["mean_fve"] = mse_metrics.get("mean_fve")

        sparsity = eval_data.get("sparsity") or {}
        out["l0_supervised_calibrated"] = sparsity.get("l0_supervised_calibrated")
        out["l0_total_calibrated"] = sparsity.get("l0_total_calibrated")

    causal_data = _read_json(cfg.causal_path)
    if causal_data:
        # causal.py writes {"feature_necessity": {"features": [...]}}; each
        # feature dict has {id, mean_kl, pred_change_rate, ...}. No pre-computed
        # aggregate, so we build the summary ourselves.
        fn = causal_data.get("feature_necessity") or {}
        feats = fn.get("features") or []
        valid = [f for f in feats if f.get("mean_kl") is not None]
        if valid:
            kls = [float(f["mean_kl"]) for f in valid]
            pcrs = [
                float(f["pred_change_rate"])
                for f in valid if f.get("pred_change_rate") is not None
            ]
            out["causal_n_evaluated"] = len(valid)
            out["causal_mean_kl"] = round(sum(kls) / len(kls), 6)
            out["causal_max_kl"] = round(max(kls), 6)
            # "Live" = above the mean_kl=0.05 threshold used in summary7.
            out["causal_n_live_features"] = sum(1 for k in kls if k >= 0.05)
            if pcrs:
                out["causal_mean_pred_change_rate"] = round(
                    sum(pcrs) / len(pcrs), 4,
                )

    intervention_path = cfg.output_dir / "intervention_precision.json"
    intervention_data = _read_json(intervention_path)
    if intervention_data:
        agg = intervention_data.get("aggregate") or {}
        out["intervention_supervised_mean_ratio"] = agg.get("supervised_mean_ratio")
        out["intervention_unsupervised_mean_ratio"] = agg.get("unsupervised_mean_ratio")
        out["intervention_pretrained_mean_ratio"] = agg.get("pretrained_mean_ratio")

    return out


def _run_step_safely(name: str, fn, *args, **kwargs) -> bool:
    """Run a pipeline step; swallow + log exceptions so the sweep continues
    on the next layer instead of aborting the whole run."""
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        print(f"  ✓ {name} ({time.time() - t0:.1f}s)")
        return True
    except Exception as e:
        print(f"  ✗ {name} FAILED after {time.time() - t0:.1f}s: "
              f"{type(e).__name__}: {e}")
        return False


def run(
    cfg: Config = None,
    layers: tuple[int, ...] = DEFAULT_LAYERS_GPT2,
    run_intervention: bool = True,
    run_causal: bool = True,
) -> list[dict]:
    """Execute the pipeline at each layer and aggregate the results.

    Args:
        cfg: base config (layer / hook_point / sae_id overridden per iteration)
        layers: iterable of target layers
        run_intervention: include Experiment C per layer
        run_causal: include the per-feature KL necessity test per layer
    """
    if cfg is None:
        cfg = Config()

    sweep_root = cfg.output_dir / "layer_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    mode = "fixed_catalog" if cfg.manual_catalog else "per_layer_catalog"
    print(f"Layer sweep root: {sweep_root}")
    print(f"Layers: {list(layers)}")
    print(f"Mode:   {mode}")
    if mode == "per_layer_catalog":
        print(
            "  NOTE: no --catalog supplied. Each layer will get its own "
            "Sonnet-organized catalog, so cross-layer F1/R²/FVE numbers "
            "reflect (layer × catalog) jointly, NOT pure layer effect. "
            "For 'which layer represents these concepts best' pass --catalog."
        )

    # Defer imports so we pay the cost once per sweep, not per layer.
    from .inventory import run as run_inventory
    from .annotate import run as run_annotate
    from .train import run as run_train
    from .evaluate import evaluate as run_evaluate
    from .causal import run as run_causal_fn
    from .intervention import run as run_intervention_fn

    per_layer_metrics: list[dict] = []

    for layer in layers:
        print("\n" + "=" * 70)
        print(f"LAYER {layer}")
        print("=" * 70)
        layer_cfg = _layer_cfg(cfg, layer, sweep_root)
        print(f"  output_dir: {layer_cfg.output_dir}")
        print(f"  hook_point: {layer_cfg.hook_point}")
        print(f"  sae_id:     {layer_cfg.sae_id}")

        # Inventory (skipped when --catalog is already in place OR catalog exists)
        if layer_cfg.manual_catalog:
            import shutil
            src = Path(layer_cfg.manual_catalog)
            if src.exists() and not layer_cfg.catalog_path.exists():
                shutil.copy2(src, layer_cfg.catalog_path)
        elif not layer_cfg.catalog_path.exists():
            _run_step_safely("inventory", run_inventory, layer_cfg)

        # Annotate (idempotent via annotations.pt presence)
        if not layer_cfg.annotations_path.exists():
            _run_step_safely("annotate", run_annotate, layer_cfg)

        # Train
        if not layer_cfg.checkpoint_path.exists():
            _run_step_safely("train", run_train, layer_cfg)

        # Evaluate
        if not layer_cfg.eval_path.exists():
            _run_step_safely("evaluate", run_evaluate, layer_cfg)

        # Causal per-feature necessity (optional)
        if run_causal and not layer_cfg.causal_path.exists():
            _run_step_safely("causal", run_causal_fn, layer_cfg)

        # Intervention precision (optional)
        intervention_path = layer_cfg.output_dir / "intervention_precision.json"
        if run_intervention and not intervention_path.exists():
            _run_step_safely("intervention", run_intervention_fn, layer_cfg)

        metrics = _collect_metrics(layer_cfg)
        per_layer_metrics.append(metrics)
        print("\n  metrics so far:")
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # ── Aggregate summary ──────────────────────────────────────────────────
    summary_path = sweep_root / "layer_sweep_summary.json"
    summary_path.write_text(json.dumps(
        {
            "model_name": cfg.model_name,
            "sae_release": cfg.sae_release,
            "mode": mode,
            "catalog": str(cfg.manual_catalog) if cfg.manual_catalog else None,
            "per_layer": per_layer_metrics,
        },
        indent=2,
    ))

    # ── Print headline table ───────────────────────────────────────────────
    cols = [
        ("layer", "L"),
        ("calibrated_f1", "calF1"),
        ("r2", "R²"),
        ("pretrained_sae_r2", "R²(P)"),
        ("mean_cosine_to_target", "cos"),
        ("mean_fve", "FVE"),
        ("causal_mean_kl", "KL"),
        ("causal_n_live_features", "live"),
        ("intervention_supervised_mean_ratio", "TR(S)"),
        ("intervention_pretrained_mean_ratio", "TR(P)"),
    ]
    header = "  " + "  ".join(f"{h:>7}" for _, h in cols)
    print("\n" + "=" * len(header))
    print("CROSS-LAYER SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in per_layer_metrics:
        parts = []
        for key, _ in cols:
            val = row.get(key)
            if val is None:
                parts.append(f"{'-':>7}")
            elif isinstance(val, float):
                parts.append(f"{val:>7.3f}")
            else:
                parts.append(f"{val:>7}")
        print("  " + "  ".join(parts))
    print(f"\nSaved: {summary_path}")

    return per_layer_metrics
