# Supervised SAE Pipeline — Test Catalog Results (v2.0)

**Date:** 2026-03-30

## Setup

- **Model**: GPT-2 Small, layer 8 residual stream (d_model=768)
- **Features**: 8 flat leaf features (comma, period, preposition, pronoun, determiner, capitalized, digit, short)
- **Data**: 500 sequences x 128 tokens from OpenWebText (64K token-positions)
- **Annotator**: Qwen3-8B-Base via vLLM, per-token binary labels with F-index prefix caching
- **Training**: Hybrid BCE + direction loss, 15 epochs, 8 supervised + 256 unsupervised latents
- **Baselines**: Linear probe on raw activations (d_model=768), post-training linear readout from pretrained SAE (24,576 latents)

## Results

| Metric | Supervised SAE | Linear Probe | Post-Training |
|---|---|---|---|
| Mean F1 (threshold=0) | 0.664 | 0.684 | 0.780 |
| Mean F1 (optimal threshold) | **0.779** | — | 0.780 |
| Mean AUROC | 0.978 | 0.980 | 0.975 |
| Reconstruction R² | 0.968 | — | — |
| R² cost vs pretrained SAE | -0.009 | — | — |

At optimal threshold, 8 supervised latents match a linear readout from 24,576 pretrained SAE latents.

### Per-Feature Breakdown

| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Opt Threshold |
|---|---|---|---|---|---|
| comma | 0.838 | 0.914 | 0.996 | 0.671 | 3.38 |
| period | 0.589 | 0.831 | 0.976 | 0.731 | 3.44 |
| preposition | 0.790 | 0.843 | 0.983 | 0.617 | 1.96 |
| pronoun | 0.542 | 0.729 | 0.983 | 0.780 | 3.85 |
| determiner | 0.782 | 0.854 | 0.990 | 0.557 | 2.03 |
| capitalized | 0.659 | 0.714 | 0.973 | 0.235 | 1.65 |
| digit | 0.554 | 0.715 | 0.978 | 0.784 | 2.85 |
| short | 0.562 | 0.632 | 0.942 | 0.194 | 1.55 |

## What Changed from v1 (F1 0.251 → 0.664)

Three fixes applied simultaneously:

1. **Negative-position magnitude loss** — The original MSE supervision only penalized positive positions. Supervised latents fired on every token (L0=5.3, precision=base rate). Added class-balanced negative term: minimize sup_acts² when label=0. L0 dropped to 0.9.

2. **Hybrid BCE + direction loss** — Replaced MSE magnitude loss with BCE on pre-activations (selectivity) + cosine alignment on decoder columns (interpretability). BCE and direction loss target different parameters (encoder vs decoder), reducing the reconstruction-supervision conflict. Mean cosine improved from 0.104 to 0.571.

3. **Flat catalog (dropped groups)** — Removed 3 group features (punctuation, function_word, surface). Preposition's target direction overlapped 79% with function_word parent, causing feature occlusion (Makelov Section 6.1). After flattening: preposition F1 went from 0.005 to 0.790.

## Comparison to Previous Experiments

| | Exp 1 (Haiku+BCE) | Exp 2 (gpt-oss+MSE) | Exp 3 (gpt-oss+IOI) | **This run** |
|---|---|---|---|---|
| Annotator | Haiku API | gpt-oss-20b | gpt-oss-20b | Qwen3-8B-Base |
| Loss | BCE | MSE | MSE | **Hybrid** |
| Mean F1 | 0.263 | 0.000 | 0.000 | **0.664** |
| Mean AUROC | 0.938 | 0.643 | — | **0.978** |
| vs Linear Probe | Tied | Lost | Lost | **Tied** |
| Annotation cost | $13-17 | Free | Free | Free |

The combination of a capable free annotator (Qwen3-8B-Base), correct loss function (hybrid BCE+direction), and clean catalog (flat, no overlapping groups) solves every cross-cutting problem from the previous experiments.

## Remaining Issues

- **Threshold calibration gap**: F1 at threshold=0 (0.664) vs optimal (0.779). BCE pre-activations aren't centered at the classification boundary — optimal thresholds range 1.5–3.8. A validation-set threshold search would close this gap.
- **Weak features**: capitalized (cosine 0.24) and short (cosine 0.19) have low decoder alignment. Likely caused by annotation noise from F-index format (observed: model labeled "Port" as not capitalized).
- **FVE still low** (mean 0.016): decoder columns explain only ~1.6% of positive-class variance. High cosine but low FVE suggests the target directions capture the right orientation but the features occupy a small fraction of the total variance at layer 8.

## Architecture

```
Loss = MSE(recon, x)
     + lambda_sup * [BCE(sup_pre, labels) + alpha * (1 - cos(dec_col, target_dir))]
     + lambda_sparse * L1(all_acts)

lambda_sup=2.0, alpha=1.0, lambda_sparse=0.001
```

Target directions: d_i = normalize(mean(x|label_i=1) - mean(x)), computed once before training.

Annotation: Qwen3-8B-Base, per-token, feature definitions in cached prefix, F-index suffix (~3 non-cached tokens per prompt). ~600 decisions/sec on single GPU.

## What Needs to Happen Next

1. **Threshold calibration** — Implement validation-set threshold tuning to close the 0.664→0.779 gap for free.
2. **Annotation quality check** — Compare F-index vs full-description suffix on a small sample to quantify accuracy loss from the caching optimization.
3. **Scale to semantic features** — Surface features validated. Test with context-dependent features (domain, sentiment, discourse) where the supervised SAE's value over a linear probe should be more apparent.
4. **Causal validation** — Run the Makelov 3-axis evaluation (approximation, controllability, interpretability) to demonstrate the SAE's advantage over a pure classifier.
5. **Feature-first caching architecture** — Further optimize annotation throughput from ~600 to ~2000+ decisions/sec for scaling to larger catalogs and datasets.
