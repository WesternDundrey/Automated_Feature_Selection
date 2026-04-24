"""
Step 5 — Inter-Annotator Agreement

Re-annotate a subset of sequences multiple times and compute Cohen's kappa
per feature to measure annotation reliability.

Features with high kappa (>0.6) have clean labels; low kappa (<0.3) features
are noisy and may train poorly regardless of architecture.

Outputs:
    pipeline_data/agreement.json

Usage:
    python -m pipeline.run --step agreement
"""

import asyncio
import json

import numpy as np
import torch
from tqdm.auto import tqdm

from .annotate import annotate_corpus_async, propagate_group_labels
from .config import Config


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa between two binary annotation vectors."""
    n = len(y1)
    if n == 0:
        return float('nan')

    # Observed agreement
    agree = (y1 == y2).sum()
    p_o = agree / n

    # Expected agreement by chance
    p1_pos = y1.mean()
    p2_pos = y2.mean()
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0

    return float((p_o - p_e) / (1 - p_e))


def run(cfg: Config = None):
    """Measure inter-annotator agreement on a subset of sequences."""
    if cfg is None:
        cfg = Config()

    if cfg.agreement_path.exists():
        print(f"Agreement results already exist: {cfg.agreement_path}")
        return json.loads(cfg.agreement_path.read_text())

    # Load required data
    for path, name in [
        (cfg.tokens_path, "tokens"),
        (cfg.catalog_path, "feature catalog"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    tokens = torch.load(cfg.tokens_path, weights_only=True)
    catalog = json.loads(cfg.catalog_path.read_text())
    features = catalog["features"]
    leaf_features = [f for f in features if f["type"] == "leaf"]
    leaf_indices = [i for i, f in enumerate(features) if f["type"] == "leaf"]

    # Select subset for agreement testing
    n_agree = min(cfg.agreement_n_sequences, tokens.shape[0])
    subset_tokens = tokens[:n_agree]
    print(f"Measuring inter-annotator agreement on {n_agree} sequences, "
          f"{cfg.agreement_n_reruns} independent runs")

    # Load tokenizer (lightweight — no need for full model)
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Run annotation multiple times independently (leaf features only).
    # IMPORTANT: the agreement run must use the SAME annotator backend that
    # produced the training labels. Previous versions hard-coded the API
    # (annotate_corpus_async) even when the main pipeline ran with
    # --local-annotator, so the reported κ was inter-rater for a DIFFERENT
    # annotator than the one whose noise actually bounds F1 on the trained
    # SAE. Dispatch on cfg.use_local_annotator to match.
    annotation_runs = []
    for run_idx in range(cfg.agreement_n_reruns):
        print(
            f"\nAnnotation run {run_idx + 1}/{cfg.agreement_n_reruns} "
            f"(backend: {'local vLLM' if cfg.use_local_annotator else 'API'})..."
        )
        if cfg.use_local_annotator:
            from .annotate import annotate_local
            leaf_labels = annotate_local(
                subset_tokens, leaf_features, tokenizer, cfg,
            )
        else:
            leaf_labels = asyncio.run(
                annotate_corpus_async(subset_tokens, leaf_features, tokenizer, cfg)
            )
        # Expand leaf annotations to full feature tensor
        full_labels = torch.zeros(subset_tokens.shape[0], subset_tokens.shape[1], len(features))
        for li, fi in enumerate(leaf_indices):
            full_labels[:, :, fi] = leaf_labels[:, :, li]
        full_labels = propagate_group_labels(full_labels, features)
        annotation_runs.append(full_labels)

    # Compute pairwise agreement metrics for each feature.
    #
    # We report two numbers per feature:
    #   - Cohen's κ: agreement beyond chance, the classical inter-rater metric.
    #   - Annotator-vs-annotator F1: a direct ceiling on the F1 a classifier
    #     can achieve against either annotator. Treat run 0 as the "reference"
    #     and run 1 as "predictions" (a symmetric swap gives the same F1 in
    #     expectation); any classifier will have F1 bounded above by this value
    #     because the labels it's trained against disagree at exactly this rate.
    # κ does NOT directly bound F1 — they measure different things. The F1
    # ceiling is the more defensible "how high can our supervised SAE go on
    # this feature" claim.
    n_features = len(features)
    kappa_results = []

    print("\n" + "=" * 70)
    print("INTER-ANNOTATOR AGREEMENT (κ + F1 ceiling)")
    print(f"  {'Feature':<36} {'Kappa':>8} {'F1':>7} {'Agree%':>8} "
          f"{'Pos1':>6} {'Pos2':>6}")
    print("  " + "-" * 75)

    kappa_values = []
    f1_ceiling_values = []

    for k, feat in enumerate(features):
        # Compare first two runs (flatten to 1D)
        y1 = annotation_runs[0][:, :, k].numpy().flatten().astype(bool)
        y2 = annotation_runs[1][:, :, k].numpy().flatten().astype(bool)

        n_pos_1 = int(y1.sum())
        n_pos_2 = int(y2.sum())

        if n_pos_1 == 0 and n_pos_2 == 0:
            kappa_results.append({
                "id": feat["id"], "type": feat["type"],
                "kappa": None, "agreement": None,
                "f1_ceiling": None,
                "n_pos_run1": 0, "n_pos_run2": 0,
                "quality": "no_data",
            })
            print(f"  {feat['id']:<36} {'--':>8} {'--':>7} {'--':>8} "
                  f"{0:>6} {0:>6}")
            continue

        kappa = cohens_kappa(y1, y2)
        agreement = float((y1 == y2).mean())

        # Annotator-vs-annotator F1 (run 0 = truth, run 1 = predictions).
        # tp = both True; precision = tp / pred+; recall = tp / truth+.
        tp = int((y1 & y2).sum())
        prec = tp / n_pos_2 if n_pos_2 > 0 else 0.0
        rec = tp / n_pos_1 if n_pos_1 > 0 else 0.0
        f1_ceiling = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )

        # Classify quality
        if np.isnan(kappa):
            quality = "undefined"
        elif kappa >= 0.6:
            quality = "good"
        elif kappa >= 0.3:
            quality = "moderate"
        else:
            quality = "poor"

        kappa_values.append(kappa)
        f1_ceiling_values.append(f1_ceiling)
        tag = " [group]" if feat["type"] == "group" else ""
        k_str = f"{kappa:.4f}" if not np.isnan(kappa) else "--"
        print(f"  {feat['id']:<36} {k_str:>8} {f1_ceiling:>7.3f} "
              f"{agreement:>7.1%} {n_pos_1:>6} {n_pos_2:>6}{tag}")

        kappa_results.append({
            "id": feat["id"], "type": feat["type"],
            "kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            "agreement": round(agreement, 4),
            "f1_ceiling": round(f1_ceiling, 4),
            "n_pos_run1": n_pos_1, "n_pos_run2": n_pos_2,
            "quality": quality,
        })

    # Summary
    if kappa_values:
        mean_kappa = float(np.mean(kappa_values))
        n_good = sum(1 for k in kappa_values if k >= 0.6)
        n_moderate = sum(1 for k in kappa_values if 0.3 <= k < 0.6)
        n_poor = sum(1 for k in kappa_values if k < 0.3)
    else:
        mean_kappa = 0.0
        n_good = n_moderate = n_poor = 0
    mean_f1_ceiling = float(np.mean(f1_ceiling_values)) if f1_ceiling_values else 0.0

    print(f"\n  Mean κ:           {mean_kappa:.3f}")
    print(f"  Mean F1 ceiling:  {mean_f1_ceiling:.3f}  "
          f"(direct upper bound on any classifier's F1)")
    print(f"  κ bands: good (>=0.6): {n_good}  moderate (0.3-0.6): {n_moderate}  "
          f"poor (<0.3): {n_poor}")

    results = {
        "n_sequences": n_agree,
        "n_reruns": cfg.agreement_n_reruns,
        "annotator_backend": "local_vllm" if cfg.use_local_annotator else "api",
        "mean_kappa": round(mean_kappa, 4),
        "mean_f1_ceiling": round(mean_f1_ceiling, 4),
        "n_good": n_good,
        "n_moderate": n_moderate,
        "n_poor": n_poor,
        "features": kappa_results,
    }

    cfg.agreement_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {cfg.agreement_path}")
    return results


if __name__ == "__main__":
    run()
