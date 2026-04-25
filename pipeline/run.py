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
                        help="Local annotator HF model ID (default: Qwen/Qwen3-4B-Base)")
    parser.add_argument("--batch-positions", action="store_true",
                        help="Full-sequence JSON output (vs per-token)")
    parser.add_argument("--catalog", default=None,
                        help="Path to manual feature catalog JSON (skips inventory step)")
    parser.add_argument("--scaffold-catalog", default=None,
                        help="Optional scaffold catalog merged into the "
                             "main catalog pre-training. Features inherit "
                             "role='control' so downstream eval stats can "
                             "separate discovery-only headlines from scaffold. "
                             "Default pipeline/scaffold_catalog.json ships with "
                             "~20 surface/artifact features.")
    parser.add_argument("--no-mse", action="store_true",
                        help="Use legacy BCE supervision instead of MSE")
    parser.add_argument("--full-desc", action="store_true",
                        help="Use full description in suffix instead of F-index (slower but more accurate)")
    parser.add_argument("--flat", action="store_true",
                        help="Strip group features from catalog, keep only leaves (no hierarchy loss)")
    parser.add_argument("--supervision", default=None,
                        choices=["hinge", "hinge_jumprelu", "gated_bce",
                                 "hybrid", "mse", "bce"],
                        help="Supervision mode. NEW (v8.11, end-to-end): "
                             "hinge (default), hinge_jumprelu, gated_bce. "
                             "LEGACY (frozen-decoder pipeline): hybrid, mse, bce.")
    parser.add_argument("--gated-tie-weights", action="store_true",
                        help="For --supervision gated_bce: tie magnitude encoder "
                             "to gate encoder via per-feature scale (halves params)")
    parser.add_argument("--jumprelu-theta-init", type=float, default=None,
                        help="For --supervision hinge_jumprelu: initial value "
                             "of per-feature θ threshold (default 0.1)")
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
                 "weaknesses", "siphoning", "discover", "discover-loop",
                 "composition", "layer-sweep", "promote-loop", "usweep",
                 "hinge-ablation", "trim-by-kappa"],
        help="Run only this step",
    )
    parser.add_argument(
        "--kappa-threshold", type=float, default=0.4,
        help="κ threshold for --step trim-by-kappa (default 0.4). Features "
             "below this drop from the catalog. 0.4 is the conventional "
             "'fair agreement' boundary in the inter-rater literature.",
    )
    parser.add_argument(
        "--apply-trim", action="store_true",
        help="For --step trim-by-kappa, replace feature_catalog.json with "
             "the trimmed version. Without this flag, the trimmed catalog "
             "is written to feature_catalog.trimmed.json but the active "
             "catalog is left alone (so you can audit before committing).",
    )
    parser.add_argument(
        "--hinge-ablation-variants", default=None,
        help="Comma-separated variants to run under --step hinge-ablation "
             "(default: all). Available: hybrid_bce, hybrid_hinge, "
             "hinge_free_zero, hinge_free_margin1, hinge_free_margin1_sq, "
             "hinge_free_margin1_lam10, hinge_free_margin1_30ep, gated_bce.",
    )
    parser.add_argument(
        "--hinge-margin", type=float, default=None,
        help="Margin for hinge supervision (default 1.0 legacy, 0.0 for "
             "mentor's zero-margin formulation). Applies to --supervision "
             "hinge / hinge_jumprelu and to the legacy --selectivity hinge path.",
    )
    parser.add_argument(
        "--hinge-squared", action="store_true",
        help="Use squared hinge: violation² / 2 instead of violation. "
             "Smoother gradient, more push on large violations.",
    )
    parser.add_argument("--widths", default=None,
                        help="Comma-separated n_unsupervised values for "
                             "--step usweep (default: 256,512,1024)")
    parser.add_argument("--promote-top-k", type=int, default=None,
                        help="K U latents considered per promote-loop round (default 20)")
    parser.add_argument("--promote-max-iters", type=int, default=None,
                        help="Max promote-loop rounds (default 5)")
    parser.add_argument("--promote-min-kept", type=int, default=None,
                        help="Terminate promote-loop if fewer than N survive a round (default 3)")
    parser.add_argument("--promote-post-train-f1-floor", type=float, default=None,
                        help="Post-training F1 floor for new features (default 0.30)")
    parser.add_argument("--promote-cos-threshold", type=float, default=None,
                        help="Cosine-dedup threshold for merge (default 0.6)")
    parser.add_argument("--promote-no-llm-separability", action="store_true",
                        help="Skip LLM separability gate in the merge step")
    parser.add_argument("--promote-proposal-budget", type=int, default=None,
                        help="Max U latents to describe per round (default 100). "
                             "Adaptive batching pulls batches until budget or min_kept crisp.")
    parser.add_argument("--promote-batch-size", type=int, default=None,
                        help="Candidates per adaptive batch (default 20)")
    parser.add_argument("--promote-no-decompose", action="store_true",
                        help="Skip multi_concept decomposition path (default: on)")
    parser.add_argument("--promote-decompose-max-atoms", type=int, default=None,
                        help="Max atoms per multi_concept decomposition (default 5)")
    parser.add_argument("--promote-atom-mini-min-pos", type=int, default=None,
                        help="Min mini-annotation positives required for an "
                             "atom to get a target_dir (default 3)")
    parser.add_argument("--promote-no-mini-prefilter", action="store_true",
                        help="Skip mini-annotation prefilter (always do full annotation)")
    parser.add_argument("--promote-mini-prefilter-n", type=int, default=None,
                        help="Sequences for mini-prefilter annotation (default 50)")
    parser.add_argument("--promote-mini-prefilter-min-auroc", type=float, default=None,
                        help="Mini-prefilter AUROC floor (default 0.70). "
                             "The v8.3 --min-f1 flag is deprecated — the real gate has been AUROC since v8.4.")
    parser.add_argument("--promote-mini-prefilter-min-f1", type=float, default=None,
                        help=argparse.SUPPRESS)  # deprecated; retained for backwards compat only
    parser.add_argument("--use-findex", action="store_true",
                        help="Opt back into F-index annotation suffix (deprecated-by-default)")
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer list for --step layer-sweep "
             "(default: 4,6,8,9,10,11 for GPT-2 Small).",
    )
    parser.add_argument(
        "--sweep-skip-intervention", action="store_true",
        help="For --step layer-sweep, skip the intervention experiment "
             "per layer (faster, no 3-way S/U/P comparison).",
    )
    parser.add_argument(
        "--sweep-skip-causal", action="store_true",
        help="For --step layer-sweep, skip per-feature KL necessity per layer.",
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
    if args.scaffold_catalog:
        overrides["scaffold_catalog"] = args.scaffold_catalog
    if args.supervision:
        overrides["supervision_mode"] = args.supervision
    if args.gated_tie_weights:
        overrides["gated_tie_weights"] = True
    if args.jumprelu_theta_init is not None:
        overrides["jumprelu_theta_init"] = args.jumprelu_theta_init
    if args.hinge_margin is not None:
        overrides["hinge_margin"] = args.hinge_margin
    if args.hinge_squared:
        overrides["hinge_squared"] = True
    if args.no_mse:
        overrides["supervision_mode"] = "bce"
    if args.full_desc:
        overrides["use_findex_suffix"] = False
    if args.use_findex:
        overrides["use_findex_suffix"] = True
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

    # Merge scaffold catalog (if provided) into the main catalog BEFORE
    # annotation. Scaffold entries carry role="control" so eval stats can
    # separate headline/discovery from surface scaffolding.
    if cfg.scaffold_catalog and cfg.catalog_path.exists():
        scaffold_path = Path(cfg.scaffold_catalog)
        if scaffold_path.exists():
            from .catalog_utils import merge_scaffold
            print(f"\n  Merging scaffold catalog: {scaffold_path}")
            merge_scaffold(cfg.catalog_path, scaffold_path)
        else:
            print(f"\n  WARNING: scaffold_catalog not found at {scaffold_path}")

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

    # Composition — K-way joint ablation linearity
    if args.step == "composition":
        print("\n" + "=" * 70)
        print("COMPOSITION — K-WAY JOINT ABLATION LINEARITY")
        print("=" * 70)
        t0 = time.time()
        from .composition import run as run_composition
        run_composition(cfg)
        print(f"Composition completed in {time.time() - t0:.1f}s")

    # Trim catalog by inter-annotator κ — drops features the annotator
    # can't reliably label, projects what mean F1 would be on the kept set.
    if args.step == "trim-by-kappa":
        print("\n" + "=" * 70)
        print("CATALOG TRIM BY ANNOTATOR κ")
        print("=" * 70)
        t0 = time.time()
        from .trim_catalog import run as run_trim
        run_trim(
            cfg,
            kappa_threshold=args.kappa_threshold,
            apply_to_disk=args.apply_trim,
        )
        print(f"Trim completed in {time.time() - t0:.1f}s")

    # Hinge-family ablation — margin / squared / frozen-vs-free / λ / epochs
    if args.step == "hinge-ablation":
        print("\n" + "=" * 70)
        print("HINGE-FAMILY ABLATION — margin/squared/frozen/λ/epochs sweep")
        print("=" * 70)
        t0 = time.time()
        from .hinge_ablation import run as run_hinge_ablation
        if args.hinge_ablation_variants:
            variants = tuple(
                v.strip() for v in args.hinge_ablation_variants.split(",") if v.strip()
            )
        else:
            variants = None
        run_hinge_ablation(cfg, variant_names=variants)
        print(f"Hinge ablation completed in {time.time() - t0:.1f}s")

    # U-width sweep — measure n_unsupervised effect on proposal quality
    if args.step == "usweep":
        print("\n" + "=" * 70)
        print("U-WIDTH SWEEP — n_unsupervised capacity diagnostic")
        print("=" * 70)
        t0 = time.time()
        from .usweep import run as run_usweep
        if args.widths:
            widths = tuple(int(x.strip()) for x in args.widths.split(",") if x.strip())
        else:
            widths = (256, 512, 1024)
        run_usweep(cfg, widths=widths)
        print(f"U-width sweep completed in {time.time() - t0:.1f}s")

    # Promote loop — residual-ranked U→S promotion
    if args.step == "promote-loop":
        print("\n" + "=" * 70)
        print("PROMOTE LOOP — U→S capacity transfer via residual ranking")
        print("=" * 70)
        t0 = time.time()
        from .promote_loop import run as run_promote_loop
        if args.promote_top_k is not None:
            cfg.promote_top_k = args.promote_top_k
        if args.promote_max_iters is not None:
            cfg.promote_max_iters = args.promote_max_iters
        if args.promote_min_kept is not None:
            cfg.promote_min_kept = args.promote_min_kept
        if args.promote_post_train_f1_floor is not None:
            cfg.promote_post_train_f1_floor = args.promote_post_train_f1_floor
        if args.promote_cos_threshold is not None:
            cfg.promote_cos_threshold = args.promote_cos_threshold
        if args.promote_no_llm_separability:
            cfg.promote_use_llm_separability = False
        if args.promote_proposal_budget is not None:
            cfg.promote_proposal_budget = args.promote_proposal_budget
        if args.promote_batch_size is not None:
            cfg.promote_batch_size = args.promote_batch_size
        if args.promote_no_decompose:
            cfg.promote_decompose_multi_concept = False
        if args.promote_decompose_max_atoms is not None:
            cfg.promote_decompose_max_atoms = args.promote_decompose_max_atoms
        if args.promote_atom_mini_min_pos is not None:
            cfg.promote_atom_mini_min_pos = args.promote_atom_mini_min_pos
        if args.promote_no_mini_prefilter:
            cfg.promote_mini_prefilter = False
        if args.promote_mini_prefilter_n is not None:
            cfg.promote_mini_prefilter_n_seqs = args.promote_mini_prefilter_n
        if args.promote_mini_prefilter_min_auroc is not None:
            cfg.promote_mini_prefilter_min_auroc = args.promote_mini_prefilter_min_auroc
        if args.promote_mini_prefilter_min_f1 is not None:
            print(
                "  [deprecated] --promote-mini-prefilter-min-f1 is a no-op since v8.4; "
                "use --promote-mini-prefilter-min-auroc instead."
            )
        run_promote_loop(cfg)
        print(f"Promote loop completed in {time.time() - t0:.1f}s")

    # Layer sweep — cross-layer pipeline orchestrator
    if args.step == "layer-sweep":
        print("\n" + "=" * 70)
        print("LAYER SWEEP — cross-layer pipeline orchestrator")
        print("=" * 70)
        t0 = time.time()
        from .layer_sweep import run as run_layer_sweep, DEFAULT_LAYERS_GPT2
        if args.layers:
            layers = tuple(
                int(x.strip()) for x in args.layers.split(",") if x.strip()
            )
        else:
            layers = DEFAULT_LAYERS_GPT2
        run_layer_sweep(
            cfg, layers=layers,
            run_intervention=not args.sweep_skip_intervention,
            run_causal=not args.sweep_skip_causal,
        )
        print(f"Layer sweep completed in {time.time() - t0:.1f}s")

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
