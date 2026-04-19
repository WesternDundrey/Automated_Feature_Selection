"""
Automated Feature Selection — Full Pipeline

Runs all steps end-to-end:
  1. Feature Inventory  — explain pretrained SAE latents, organize into catalog
  2. Annotation         — tokenize corpus, extract activations, LLM-annotate
  3. Training           — train supervised SAE
  4. Evaluation         — held-out metrics

Each step is resumable: if its output files already exist, the step is skipped.
Delete specific output files to re-run individual steps.

Usage:
    python -m pipeline.run
    python -m pipeline.run --layer 16 --n_sequences 10000
"""

import argparse
import time
from pathlib import Path

from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Automated Feature Selection Pipeline")
    parser.add_argument("--model", default=None, help="Base model name")
    parser.add_argument("--layer", type=int, default=None, help="Target layer")
    parser.add_argument("--sae_release", default=None, help="SAE release (sae_lens)")
    parser.add_argument("--sae_id", default=None, help="SAE ID (sae_lens)")
    parser.add_argument("--n_latents", type=int, default=None, help="Latents to explain")
    parser.add_argument("--n_sequences", type=int, default=None, help="Corpus sequences")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model-dtype", default=None,
                        help="Model dtype (float32/bfloat16)")
    parser.add_argument("--lista", type=int, default=None, help="LISTA refinement steps")
    parser.add_argument("--lambda-sup", type=float, default=None,
                        help="Supervision loss weight (default: 2.0)")
    parser.add_argument("--lambda-sparse", type=float, default=None,
                        help="L1 sparsity loss weight (default: 0.05; bump for tighter unsup L0)")
    parser.add_argument("--lambda-hier", type=float, default=None,
                        help="Hierarchy loss weight (default: 0.5)")
    parser.add_argument("--local-annotator", action="store_true",
                        help="Use local model for annotation (vLLM)")
    parser.add_argument("--annotator-model", default=None,
                        help="Local annotator HF model ID (default: Qwen/Qwen3.5-9B)")
    parser.add_argument("--batch-positions", action="store_true",
                        help="Full-sequence JSON output (vs per-token)")
    parser.add_argument("--catalog", default=None,
                        help="Path to manual feature catalog JSON (skips inventory step)")
    parser.add_argument("--no-mse", action="store_true",
                        help="Use legacy BCE supervision instead of MSE")
    parser.add_argument("--full-desc", action="store_true",
                        help="Use full description in suffix instead of F-index (slower but more accurate)")
    parser.add_argument("--flat", action="store_true",
                        help="Strip group features from catalog, keep only leaves (no hierarchy loss)")
    parser.add_argument("--supervision", default=None,
                        choices=["hybrid", "mse", "bce"],
                        help="Supervision mode: hybrid (BCE+direction), mse, bce")
    parser.add_argument("--no-freeze-decoder", action="store_true",
                        help="Train decoder columns (legacy). Default: frozen to target_dirs")
    parser.add_argument("--selectivity", default=None,
                        choices=["bce", "hinge", "none"],
                        help="Selectivity loss type (default: bce)")
    parser.add_argument(
        "--step", default=None,
        choices=["inventory", "annotate", "train", "evaluate",
                 "agreement", "ablation", "residual", "causal", "ioi",
                 "validate-annotator",
                 "splitting", "circuit", "intervention", "amplify",
                 "weaknesses", "siphoning", "discover", "discover-loop"],
        help="Run only this step",
    )
    parser.add_argument("--discover-loop-max-iters", type=int, default=5,
                        help="Max iterations for --step discover-loop")
    parser.add_argument("--discover-loop-min-new", type=int, default=3,
                        help="Terminate discover-loop if a round yields fewer "
                             "than N novel features after merge (default: 3)")
    parser.add_argument("--discover-loop-min-delta-r2", type=float, default=0.005,
                        help="Terminate discover-loop if a round's ΔR² falls "
                             "below this (default: 0.005)")
    parser.add_argument("--discover-loop-cos-threshold", type=float, default=0.6,
                        help="Cosine-dedup threshold for merge (default: 0.6). "
                             "Candidates above this cosine with an existing "
                             "target_dir are rejected as rediscoveries. "
                             "0.6 is calibrated for encoder-row vs target_dir "
                             "comparison in d_model=768 space; raise to 0.7-0.8 "
                             "if the loop rejects too aggressively.")
    parser.add_argument("--discover-loop-no-llm-separability", action="store_true",
                        help="Skip LLM separability gate; rely on cosine only")
    args = parser.parse_args()

    # Build config with overrides
    overrides = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.layer is not None:
        overrides["target_layer"] = args.layer
        overrides["hook_point"] = ""  # force re-derive
    if args.sae_release:
        overrides["sae_release"] = args.sae_release
    if args.sae_id:
        overrides["sae_id"] = args.sae_id
    if args.n_latents:
        overrides["n_latents_to_explain"] = args.n_latents
    if args.n_sequences:
        overrides["n_sequences"] = args.n_sequences
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.model_dtype:
        overrides["model_dtype"] = args.model_dtype
    if args.lista is not None:
        overrides["n_lista_steps"] = args.lista
    if args.lambda_sup is not None:
        overrides["lambda_sup"] = args.lambda_sup
    if args.lambda_sparse is not None:
        overrides["lambda_sparse"] = args.lambda_sparse
    if args.lambda_hier is not None:
        overrides["lambda_hier"] = args.lambda_hier
    if args.local_annotator:
        overrides["use_local_annotator"] = True
    if args.annotator_model:
        overrides["local_annotator_model"] = args.annotator_model
    if args.batch_positions:
        overrides["batch_positions"] = True
    if args.catalog:
        overrides["manual_catalog"] = args.catalog
    if args.supervision:
        overrides["supervision_mode"] = args.supervision
    if args.no_mse:
        overrides["supervision_mode"] = "bce"
    if args.full_desc:
        overrides["use_findex_suffix"] = False
    if args.no_freeze_decoder:
        overrides["freeze_supervised_decoder"] = False
    if args.selectivity:
        overrides["selectivity_loss"] = args.selectivity

    cfg = Config(**overrides)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUTOMATED FEATURE SELECTION PIPELINE")
    print("=" * 70)
    print(f"  Model:      {cfg.model_name}")
    print(f"  Layer:      {cfg.target_layer}")
    print(f"  SAE:        {cfg.sae_release} / {cfg.sae_id}")
    print(f"  Sequences:  {cfg.n_sequences}")
    print(f"  Output:     {cfg.output_dir}")
    print(f"  Device:     {cfg.device}")
    print()

    t_total = time.time()

    # Step 1: Feature Inventory (skipped if manual catalog provided)
    if cfg.manual_catalog:
        import json
        import shutil
        src = Path(cfg.manual_catalog)
        if not src.exists():
            raise FileNotFoundError(f"Manual catalog not found: {src}")
        if not cfg.catalog_path.exists() or src.resolve() != cfg.catalog_path.resolve():
            shutil.copy2(src, cfg.catalog_path)
        catalog = json.loads(cfg.catalog_path.read_text())
        n_g = sum(1 for f in catalog["features"] if f["type"] == "group")
        n_l = sum(1 for f in catalog["features"] if f["type"] == "leaf")
        print(f"\n  Using manual catalog: {src}")
        print(f"  {n_g} groups, {n_l} leaves — skipping inventory step")
    elif args.step is None or args.step == "inventory":
        print("\n" + "=" * 70)
        print("STEP 1: FEATURE INVENTORY")
        print("=" * 70)
        t0 = time.time()
        from .inventory import run as run_inventory
        run_inventory(cfg)
        print(f"Step 1 completed in {time.time() - t0:.1f}s")

    # Flatten catalog if requested (strip groups, keep only leaves)
    if args.flat and cfg.catalog_path.exists():
        import json as _json
        _cat = _json.loads(cfg.catalog_path.read_text())
        _before = len(_cat["features"])
        _cat["features"] = [f for f in _cat["features"] if f["type"] == "leaf"]
        # Clear parent references (no hierarchy)
        for f in _cat["features"]:
            f.pop("parent", None)
        _after = len(_cat["features"])
        if _after < _before:
            cfg.catalog_path.write_text(_json.dumps(_cat, indent=2))
            print(f"\n  --flat: stripped {_before - _after} group features, "
                  f"{_after} leaves remain")

    # Step 2: Annotation
    if args.step is None or args.step == "annotate":
        print("\n" + "=" * 70)
        print("STEP 2: CORPUS PREPARATION & LLM ANNOTATION")
        print("=" * 70)
        t0 = time.time()
        from .annotate import run as run_annotate
        run_annotate(cfg)
        print(f"Step 2 completed in {time.time() - t0:.1f}s")

    # Step 3: Training
    if args.step is None or args.step == "train":
        print("\n" + "=" * 70)
        print("STEP 3: TRAINING")
        print("=" * 70)
        t0 = time.time()
        from .train import run as run_train
        run_train(cfg)
        print(f"Step 3 completed in {time.time() - t0:.1f}s")

    # Step 4: Evaluation
    if args.step is None or args.step == "evaluate":
        print("\n" + "=" * 70)
        print("STEP 4: EVALUATION")
        print("=" * 70)
        t0 = time.time()
        from .evaluate import evaluate
        evaluate(cfg)
        print(f"Step 4 completed in {time.time() - t0:.1f}s")

    # Step 5: Inter-annotator agreement (optional, run with --step agreement)
    if args.step == "agreement":
        print("\n" + "=" * 70)
        print("STEP 5: INTER-ANNOTATOR AGREEMENT")
        print("=" * 70)
        t0 = time.time()
        from .agreement import run as run_agreement
        run_agreement(cfg)
        print(f"Step 5 completed in {time.time() - t0:.1f}s")

    # Step 6: Ablation study (optional, run with --step ablation)
    if args.step == "ablation":
        print("\n" + "=" * 70)
        print("STEP 6: ABLATION STUDY")
        print("=" * 70)
        t0 = time.time()
        from .ablation import run as run_ablation
        run_ablation(cfg)
        print(f"Step 6 completed in {time.time() - t0:.1f}s")

    # Step 7: Explain the residual (optional, run with --step residual)
    if args.step == "residual":
        print("\n" + "=" * 70)
        print("STEP 7: EXPLAIN THE RESIDUAL")
        print("=" * 70)
        t0 = time.time()
        from .residual import run as run_residual
        run_residual(cfg)
        print(f"Step 7 completed in {time.time() - t0:.1f}s")

    # Step 8: Causal validation (optional, run with --step causal)
    if args.step == "causal":
        print("\n" + "=" * 70)
        print("STEP 8: CAUSAL VALIDATION")
        print("=" * 70)
        t0 = time.time()
        from .causal import run as run_causal
        run_causal(cfg)
        print(f"Step 8 completed in {time.time() - t0:.1f}s")

    # IOI validation (standalone, run with --step ioi)
    if args.step == "ioi":
        print("\n" + "=" * 70)
        print("IOI VALIDATION (Makelov et al. 2024)")
        print("=" * 70)
        t0 = time.time()
        from .ioi import run as run_ioi
        skip_ann = not cfg.use_local_annotator
        run_ioi(cfg, skip_annotator=skip_ann)
        print(f"IOI validation completed in {time.time() - t0:.1f}s")

    # Phase 3 — Experiment A: Feature splitting quantification
    if args.step == "splitting":
        print("\n" + "=" * 70)
        print("PHASE 3 — EXPERIMENT A: FEATURE SPLITTING")
        print("=" * 70)
        t0 = time.time()
        from .feature_splitting import run as run_splitting
        run_splitting(cfg)
        print(f"Experiment A completed in {time.time() - t0:.1f}s")

    # Phase 3 — Experiment B: Downstream circuit analysis
    if args.step == "circuit":
        print("\n" + "=" * 70)
        print("PHASE 3 — EXPERIMENT B: CIRCUIT ANALYSIS")
        print("=" * 70)
        t0 = time.time()
        from .circuit import run as run_circuit
        run_circuit(cfg)
        print(f"Experiment B completed in {time.time() - t0:.1f}s")

    # Phase 3 — Experiment C: Intervention precision
    if args.step == "intervention":
        print("\n" + "=" * 70)
        print("PHASE 3 — EXPERIMENT C: INTERVENTION PRECISION")
        print("=" * 70)
        t0 = time.time()
        from .intervention import run as run_intervention
        run_intervention(cfg)
        print(f"Experiment C completed in {time.time() - t0:.1f}s")

    # Experiment D: Activation amplification sweep
    if args.step == "amplify":
        print("\n" + "=" * 70)
        print("EXPERIMENT D: ACTIVATION AMPLIFICATION SWEEP")
        print("=" * 70)
        t0 = time.time()
        from .amplify import run as run_amplify
        run_amplify(cfg)
        print(f"Experiment D completed in {time.time() - t0:.1f}s")

    # Weakness spotlight (reads evaluation.json + causal.json, no retrain)
    if args.step == "weaknesses":
        print("\n" + "=" * 70)
        print("WEAKNESS SPOTLIGHT")
        print("=" * 70)
        t0 = time.time()
        from .weaknesses import run as run_weaknesses
        run_weaknesses(cfg)
        print(f"Weakness spotlight completed in {time.time() - t0:.1f}s")

    # FVE siphoning sweep (retrains across n_unsupervised values)
    if args.step == "siphoning":
        print("\n" + "=" * 70)
        print("FVE SIPHONING SWEEP")
        print("=" * 70)
        t0 = time.time()
        from .siphoning import run as run_siphoning
        run_siphoning(cfg)
        print(f"Siphoning sweep completed in {time.time() - t0:.1f}s")

    # Plan 2 — Discovery pipeline: unsupervised SAE → annotate → catalog
    if args.step == "discover":
        print("\n" + "=" * 70)
        print("DISCOVERY PIPELINE (unsupervised → annotate → catalog)")
        print("=" * 70)
        t0 = time.time()
        from .discover import run as run_discover
        run_discover(cfg)
        print(f"Discovery pipeline completed in {time.time() - t0:.1f}s")

    if args.step == "discover-loop":
        print("\n" + "=" * 70)
        print("DISCOVERY LOOP — iterative supervised-SAE catalog growth")
        print("=" * 70)
        t0 = time.time()
        from .discover_loop import run as run_discover_loop
        run_discover_loop(
            cfg,
            max_iters=args.discover_loop_max_iters,
            min_new_features=args.discover_loop_min_new,
            min_delta_r2=args.discover_loop_min_delta_r2,
            cos_threshold=args.discover_loop_cos_threshold,
            use_llm_separability=not args.discover_loop_no_llm_separability,
        )
        print(f"Discovery loop completed in {time.time() - t0:.1f}s")

    # Annotator validation on trivial features
    if args.step == "validate-annotator":
        print("\n" + "=" * 70)
        print("ANNOTATOR VALIDATION (trivial features)")
        print("=" * 70)
        t0 = time.time()
        from .validate_annotator import run as run_validate
        run_validate(cfg)
        print(f"Validation completed in {time.time() - t0:.1f}s")

    print(f"\nTotal pipeline time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
