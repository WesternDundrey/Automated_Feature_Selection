# Supervised SAE Pipeline — Gemma-2-2B Results (v4.0)

**Date:** 2026-04-07

## Setup

- **Model**: Gemma-2-2B, layer 20 residual stream (d_model=2304)
- **Features**: 60 supervised latents (flat, leaves only) across 12 semantic categories
- **Data**: 1000 sequences x 128 tokens from OpenWebText (128K token-positions, 12,800 val / 12,800 test)
- **Annotator**: Qwen3-4B-Base via vLLM, full-desc suffix, per-token binary labels with prefix caching
- **Feature catalog**: Sonnet-generated from 500 GemmaScope SAE latents, flattened to 60 leaves
- **Training**: Hybrid BCE + direction loss, 60 supervised + 256 unsupervised = 316 total latents
- **Sparsity**: lambda_sparse=0.01 (10x GPT-2 default)
- **Baselines**: Linear probe on raw activations (d_model=2304), post-training readout from GemmaScope SAE (16,384 latents)

## Headline Results

| Metric | Supervised SAE | Linear Probe | Post-Training (16K) |
|---|---|---|---|
| Mean F1 (t=0) | 0.537 | 0.487 | 0.565 |
| **Mean F1 (calibrated)** | **0.601** | — | **0.565** |
| Mean F1 (oracle) | 0.629 | — | — |
| Mean AUROC | 0.967 | 0.967 | 0.959 |
| R² | 0.7361 | — | — |
| Mean cosine (decoder alignment) | 0.600 | — | — |
| L0 (supervised) | 4.5 | — | — |
| L0 (total) | 247.1 | — | — |

**Key result:** 60 supervised latents (calibrated F1 0.601) beat the linear probe (0.487) by +0.114 and the post-training readout from 16K GemmaScope latents (0.565) by +0.036. AUROC is tied at 0.967.

**Caveat:** The GemmaScope pretrained SAE reconstruction shows R²=-6.85, indicating a loading/convention mismatch. The post-training baseline uses latent activations from this (potentially miscalibrated) SAE, so the +0.036 margin may not be reliable. The linear probe comparison is clean — both use identical activations.

## Comparison: GPT-2 vs Gemma-2-2B

| Metric | GPT-2 Small (54 feat) | Gemma-2-2B (60 feat) |
|---|---|---|
| Calibrated F1 | 0.546 | **0.601** (+0.055) |
| Linear probe F1 | 0.451 | 0.487 |
| **SAE vs probe gap** | +0.095 | **+0.114** |
| Mean AUROC | 0.961 | **0.967** |
| Mean cosine | 0.533 | **0.600** |
| R² | 0.966 | 0.736 |
| L0 (supervised) | 7.0 | 4.5 |
| L0 (total) | 254.2 | 247.1 |

Gemma improves on every metric except R² (expected — 316 latents reconstructing 2304-dim is much harder than 768-dim). The supervised SAE's advantage over the probe grows from +0.095 to +0.114.

## Sparsity

- **L0 supervised:** 4.5/60 — only 7.5% of supervised latents fire per token (very selective)
- **L0 total:** 247.1/316 — 78% of all latents active (improved from GPT-2's 79.4% with 10x lambda_sparse)
- **lambda_sparse=0.01** (was 0.001 for GPT-2) — higher penalty but still not enough for sparse unsupervised latents

## Per-Feature Classification (calibrated thresholds)

### Top 10 Features by Calibrated F1

| Feature | calF1 | AUROC | Cosine | Pos |
|---|---|---|---|---|
| code_and_technical.programming_keyword | 0.986 | 0.9997 | 0.431 | 108 |
| token_surface_form.digit_or_number | 0.918 | 0.9984 | 0.671 | 510 |
| code_and_technical.database_schema | 0.881 | 0.9818 | 0.384 | 131 |
| grammar_function.article_or_determiner | 0.826 | 0.9849 | 0.747 | 836 |
| semantic_domain.food_and_cooking | 0.826 | 0.9906 | 0.396 | 236 |
| document_structure.section_header | 0.833 | 0.9891 | 0.252 | 161 |
| token_surface_form.capitalized_proper_noun | 0.823 | 0.9838 | 0.527 | 1570 |
| punctuation.bracket_or_paren | 0.780 | 0.9745 | 0.470 | 1695 |
| named_entity_type.place_name | 0.763 | 0.9933 | 0.706 | 285 |
| document_structure.beginning_of_sequence | 0.762 | 0.8525 | 0.395 | 5955 |

### Bottom 5 Features (data-starved or genuinely hard)

| Feature | calF1 | AUROC | Cosine | Pos |
|---|---|---|---|---|
| narrative_and_rhetorical.comparison_or_analogy | 0.059 | 0.9508 | 0.713 | 13 |
| quantitative_and_measurement.range_expression | 0.294 | 0.9752 | 0.536 | 23 |
| temporal_expression.past_habitual_or_historical | 0.360 | 0.9696 | 0.741 | 35 |
| punctuation.exclamation_mark | 0.366 | 0.9251 | 0.548 | 306 |
| token_surface_form.code_operator_or_symbol | 0.395 | 0.9829 | 0.557 | 84 |

Note: comparison_or_analogy has high AUROC (0.95) and cosine (0.71) despite terrible F1 — it's purely data-starved (13 positives). The model learned the concept but has too few examples for reliable threshold calibration.

## Semantic Features — Gemma's Strength

Gemma-2-2B shows notably better semantic feature detection than GPT-2:

| Semantic Feature | Gemma calF1 | GPT-2 calF1 |
|---|---|---|
| food_and_cooking / food_lifestyle | 0.826 | 0.508 |
| sports_and_athletics / — | 0.753 | — |
| technology_and_computing / technology_business | 0.718 | 0.415 |
| politics_and_government | 0.664 | 0.693 |
| arts_and_entertainment / entertainment_media | 0.694 | 0.432 |
| science_and_medicine / science_academia | 0.668 | 0.587 |

Gemma's richer 2304-dim representations enable cleaner linear features for semantic domains.

## Decoder Alignment

- **Mean cosine to target:** 0.600 (GPT-2: 0.533)
- **Features > 0.8:** date_or_year (0.824), reader_address (0.801)
- **Features > 0.7:** politics_government (0.759), sports (0.771), law_and_crime (0.751), religion_and_culture (0.798), conjunction (0.743), quotation_mark (0.744), article_or_determiner (0.747), auxiliary_verb (0.793), ordinal_or_enumerator (0.778), comma (0.716), place_name (0.706), possibility_or_feasibility (0.724), direct_quote (0.720), first_person_narrative (0.735), comparison_or_analogy (0.713), variable_or_identifier (0.731)

Gemma achieves higher cosine alignment on semantic and grammatical features, suggesting these concepts are more linearly represented in Gemma's residual stream than in GPT-2's.

## Architecture

```
Loss = MSE(recon, x)
     + lambda_sup * [BCE(sup_pre, labels) + alpha * (1 - cos(dec_col, target_dir))]
     + lambda_sparse * L1(all_acts)

lambda_sup=2.0, alpha=1.0, lambda_sparse=0.01
60 supervised + 256 unsupervised = 316 total latents
```

Hierarchy loss removed (flat catalog, no parent-child pairs).

## What Needs to Happen Next

1. **Fix GemmaScope SAE loading** — R²=-6.85 means the pretrained SAE baseline is broken. Likely a b_dec subtraction or hook point convention mismatch in sae_lens. Fix to get a fair post-training comparison.

2. **Increase unsupervised latents** — R²=0.736 is low. With d_model=2304, the SAE needs more capacity. Try 60 supervised + 512 or 1024 unsupervised.

3. **Increase data** — 1000 sequences is adequate but several features have <30 test positives. 2000+ sequences would stabilize rare features.

4. **Causal validation** — Run per-feature necessity test on Gemma. Expect stronger causal signal than GPT-2 given the higher cosine alignment.

5. **Tune sparsity** — L0=247/316 is still too dense. Try lambda_sparse=0.05 or 0.1 to get meaningful sparsity in unsupervised latents.
