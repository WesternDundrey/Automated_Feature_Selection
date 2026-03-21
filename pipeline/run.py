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
    parser.add_argument("--lista", type=int, default=None, help="LISTA refinement steps")
    parser.add_argument("--local-annotator", action="store_true",
                        help="Use local model for annotation (vLLM + prefix caching)")
    parser.add_argument("--annotator-model", default=None,
                        help="Local annotator HF model ID (default: Qwen/Qwen3-8B)")
    parser.add_argument("--no-mse", action="store_true",
                        help="Use legacy BCE supervision instead of MSE")
    parser.add_argument(
        "--step", default=None,
        choices=["inventory", "annotate", "train", "evaluate",
                 "agreement", "ablation", "residual", "causal"],
        help="Run only this step",
    )
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
    if args.lista is not None:
        overrides["n_lista_steps"] = args.lista
    if args.local_annotator:
        overrides["use_local_annotator"] = True
    if args.annotator_model:
        overrides["local_annotator_model"] = args.annotator_model
    if args.no_mse:
        overrides["use_mse_supervision"] = False

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

    # Step 1: Feature Inventory
    if args.step is None or args.step == "inventory":
        print("\n" + "=" * 70)
        print("STEP 1: FEATURE INVENTORY")
        print("=" * 70)
        t0 = time.time()
        from .inventory import run as run_inventory
        run_inventory(cfg)
        print(f"Step 1 completed in {time.time() - t0:.1f}s")

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

    print(f"\nTotal pipeline time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
