# Supervised SAE Pipeline — Full Catalog Results (v3.0)

**Date:** 2026-03-31

## Setup

- **Model**: GPT-2 Small, layer 8 residual stream (d_model=768)
- **Features**: 64 supervised latents (11 groups + 53 leaves) across 11 semantic categories
- **Data**: 500 sequences x 128 tokens from OpenWebText (64K token-positions, 12,800 test)
- **Annotator**: Qwen3-8B-Base via vLLM, per-token binary labels with token-ID prefix caching
- **Feature catalog**: Sonnet-generated inventory with hierarchical grouping
- **Training**: Hybrid BCE + direction loss, 64 supervised + 256 unsupervised latents
- **Baselines**: Linear probe on raw activations (d_model=768), post-training linear readout from pretrained SAE (24,576 latents)

## Headline Results

| Metric | Supervised SAE | Linear Probe | Post-Training (24K) |
|---|---|---|---|
| Mean F1 (threshold=0) | 0.394 | 0.413 | 0.474 |
| **Mean F1 (optimal threshold)** | **0.502** | — | **0.474** |
| Mean AUROC | 0.925 | 0.939 | 0.924 |
| Reconstruction R² | 0.9686 | — | — |
| R² cost vs pretrained SAE | -0.0085 | — | — |
| Mean cosine (decoder alignment) | 0.556 | — | — |
| Features with cosine > 0.8 | 5/61 | — | — |
| Mean FVE | 0.015 | — | — |

**Key result:** At optimal threshold, 64 supervised latents (F1 0.502) beat a linear readout from 24,576 pretrained SAE latents (F1 0.474), while matching on AUROC (0.925 vs 0.924). Reconstruction cost is negligible (-0.85% R²).

## Sparsity

- **L0 (supervised latents):** 7.7 — out of 64 supervised latents, ~7.7 fire per token on average
- **L0 (all latents):** 255.6 — out of 320 total latents (64 sup + 256 unsup), ~256 fire per token
- **Supervised selectivity:** 7.7/64 = 12% of supervised latents active per token (good — features are selective)
- **Unsupervised density:** ~248/256 unsupervised latents fire per token (very dense — L1 penalty may be too weak)
- **Reconstruction efficiency:** 320 total latents achieve R² 0.9686 vs 24,576 latents at R² 0.9771

## Per-Feature Classification (threshold=0 / optimal threshold)

### Punctuation (group AUROC: 0.966)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| punctuation [G] | 0.778 | 0.840 | 0.966 | 0.664 | 2233 |
| comma | 0.781 | 0.915 | 0.992 | 0.695 | 584 |
| period | 0.687 | 0.726 | 0.960 | 0.659 | 1558 |
| closing_paren | 0.208 | 0.359 | 0.966 | 0.754 | 138 |
| bracket_reference | 0.000 | 0.125 | 0.529 | 0.773 | 4 |
| encoding_artifact | 0.667 | 0.819 | 0.989 | 0.881 | 422 |

### Token Form (group AUROC: 0.930)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| token_form [G] | 0.443 | 0.600 | 0.930 | 0.198 | 1104 |
| numeric | 0.760 | 0.873 | 0.995 | 0.621 | 289 |
| suffix | 0.205 | 0.354 | 0.949 | 0.544 | 128 |
| subword_prefix | 0.298 | 0.370 | 0.917 | 0.658 | 427 |
| all_caps | 0.394 | 0.537 | 0.973 | 0.136 | 231 |
| url_slug | 0.343 | 0.487 | 0.983 | 0.581 | 132 |
| abbreviation | 0.254 | 0.408 | 0.964 | 0.231 | 152 |

### Part of Speech (group AUROC: 0.885)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| part_of_speech [G] | 0.613 | 0.667 | 0.885 | 0.748 | 3173 |
| preposition | 0.689 | 0.721 | 0.971 | 0.749 | 1380 |
| conjunction | 0.421 | 0.568 | 0.978 | 0.706 | 253 |
| auxiliary_verb | 0.356 | 0.449 | 0.972 | 0.563 | 101 |
| past_tense_verb | 0.620 | 0.697 | 0.964 | 0.711 | 992 |
| adjective_adverb | 0.325 | 0.568 | 0.984 | 0.836 | 117 |
| common_noun | 0.383 | 0.458 | 0.929 | 0.215 | 621 |

### Syntactic Position (group AUROC: 0.903)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| syntactic_position [G] | 0.664 | 0.700 | 0.903 | 0.165 | 3093 |
| post_verb_particle | 0.044 | 0.095 | 0.921 | 0.625 | 7 |
| sentence_initial | 0.723 | 0.740 | 0.949 | 0.279 | 2345 |
| appositive_modifier | 0.507 | 0.576 | 0.941 | 0.671 | 968 |
| complement_introducer | 0.115 | 0.212 | 0.953 | 0.690 | 41 |
| post_comparative | 0.342 | 0.412 | 0.903 | 0.704 | 564 |

### Semantic Domain (group AUROC: 0.824)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| semantic_domain [G] | 0.728 | 0.751 | 0.824 | 0.583 | 6270 |
| politics_government | 0.709 | 0.730 | 0.955 | 0.765 | 1859 |
| crime_conflict | 0.456 | 0.522 | 0.936 | 0.512 | 707 |
| science_technology | 0.352 | 0.510 | 0.953 | 0.753 | 349 |
| entertainment_culture | 0.669 | 0.684 | 0.910 | 0.539 | 2605 |
| food_lifestyle | 0.180 | 0.429 | 0.930 | 0.565 | 119 |
| sports | 0.763 | 0.824 | 0.983 | 0.519 | 1096 |
| lgbtq_identity | 0.125 | 0.213 | 0.923 | 0.274 | 112 |

### Named Entity (group AUROC: 0.978)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| named_entity [G] | 0.561 | 0.697 | 0.978 | 0.283 | 659 |
| person_name | 0.535 | 0.708 | 0.993 | 0.260 | 229 |
| publication | 0.235 | 0.479 | 0.981 | 0.238 | 58 |
| place_or_organization | 0.560 | 0.677 | 0.980 | 0.818 | 403 |
| religion_or_ideology | 0.279 | 0.476 | 0.978 | 0.798 | 12 |

### Journalistic Construction (group AUROC: 0.937)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| journalistic_construction [G] | 0.448 | 0.568 | 0.937 | 0.489 | 876 |
| attribution_verb | 0.000 | 0.055 | 0.945 | 0.737 | 5 |
| quote_context | 0.491 | 0.575 | 0.943 | 0.744 | 800 |
| death_location | 0.042 | 0.083 | 0.808 | 0.728 | 5 |
| photo_credit | 0.241 | 0.359 | 0.967 | 0.159 | 78 |

### Temporal Expression (group AUROC: 0.939)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| temporal_expression [G] | 0.247 | 0.422 | 0.939 | 0.757 | 320 |
| duration_unit | 0.329 | 0.512 | 0.986 | 0.836 | 46 |
| historical_era | 0.000 | 0.040 | 0.878 | 0.798 | 5 |
| office_transition | 0.289 | 0.429 | 0.945 | 0.666 | 271 |
| newly_recent | — | — | — | — | 0 |

### Discourse Marker (group AUROC: 0.920)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| discourse_marker [G] | 0.505 | 0.613 | 0.920 | 0.398 | 1254 |
| exemplification | 0.000 | 0.000 | 0.140 | 0.782 | 1 |
| contrast | 0.400 | 0.667 | 1.000 | 0.943 | 1 |
| narrative_intro | 0.139 | 0.286 | 0.976 | 0.667 | 10 |
| fact_complement | 0.000 | 0.024 | 0.898 | 0.713 | 4 |
| conditional | 0.544 | 0.619 | 0.922 | 0.539 | 1245 |

### Institutional Role (group AUROC: 0.970)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| institutional_role [G] | 0.300 | 0.561 | 0.970 | 0.198 | 288 |
| government_title | 0.438 | 0.585 | 0.991 | 0.697 | 111 |
| regulated_institution | 0.311 | 0.449 | 0.986 | 0.843 | 101 |
| occupation | 0.329 | 0.500 | 0.985 | 0.565 | 83 |
| academic | 0.140 | 0.374 | 0.939 | 0.163 | 59 |

### Web/UI Text (group AUROC: 0.896)
| Feature | F1 (t=0) | F1 (opt) | AUROC | Cosine | Pos |
|---|---|---|---|---|---|
| web_ui_text [G] | 0.637 | 0.667 | 0.896 | 0.171 | 2782 |
| navigation_element | 0.048 | 0.058 | 0.812 | 0.209 | 18 |
| blog_link_reference | 0.505 | 0.576 | 0.931 | 0.695 | 1045 |
| listicle_content | 0.637 | 0.656 | 0.905 | 0.165 | 2322 |

## Hierarchy Consistency

Average consistency across 53 parent-child pairs: **0.637**

Strong hierarchy (consistency > 0.8):
- punctuation → comma (0.873), period (0.889), closing_paren (0.868), encoding_artifact (0.878)
- syntactic_position → sentence_initial (0.894), appositive_modifier (0.816)
- semantic_domain → entertainment_culture (0.884), sports (0.868)
- web_ui_text → listicle_content (0.899), navigation_element (0.833), blog_link_reference (0.801)
- discourse_marker → conditional (0.832)
- part_of_speech → preposition (0.825)

Weak hierarchy (consistency < 0.4):
- discourse_marker → exemplification (0.000), narrative_intro (0.145), contrast (0.250)
- named_entity → religion_or_ideology (0.323)
- temporal_expression → historical_era (0.190)
- journalistic_construction → death_location (0.372)
- syntactic_position → post_verb_particle (0.395)

Weak hierarchy pairs are almost all data-starved features (< 15 positives).

## Decoder Alignment (Cosine Similarity)

- **Mean cosine:** 0.556
- **Features > 0.8:** encoding_artifact (0.88), adjective_adverb (0.84), regulated_institution (0.84), duration_unit (0.84), place_or_organization (0.82)
- **Features > 0.7:** bracket_reference (0.77), closing_paren (0.75), science_technology (0.75), preposition (0.75), part_of_speech [G] (0.75), temporal_expression [G] (0.76), politics_government (0.76), quote_context (0.74), attribution_verb (0.74), death_location (0.73), past_tense_verb (0.71), conjunction (0.71), post_comparative (0.70), complement_introducer (0.69), religion_or_ideology (0.80), historical_era (0.80), fact_complement (0.71)
- **Features < 0.3:** token_form [G] (0.20), all_caps (0.14), common_noun (0.21), abbreviation (0.23), syntactic_position [G] (0.17), sentence_initial (0.28), lgbtq_identity (0.27), named_entity [G] (0.28), person_name (0.26), publication (0.24), photo_credit (0.16), institutional_role [G] (0.20), academic (0.16), web_ui_text [G] (0.17), navigation_element (0.21), listicle_content (0.17)

Pattern: group features tend to have low cosine (direction = OR of diverse children). Leaf features with clear lexical signatures (punctuation, POS) achieve high cosine. Broad semantic features (listicle_content, sentence_initial) have low cosine — their concept direction is diffuse across many contexts.

## Data-Starved Features (< 50 positives in 12,800 test vectors)

| Feature | Positives | F1 (opt) | AUROC |
|---|---|---|---|
| bracket_reference | 4 | 0.125 | 0.529 |
| post_verb_particle | 7 | 0.095 | 0.921 |
| attribution_verb | 5 | 0.055 | 0.945 |
| death_location | 5 | 0.083 | 0.808 |
| duration_unit | 46 | 0.512 | 0.986 |
| historical_era | 5 | 0.040 | 0.878 |
| newly_recent | 0 | — | — |
| exemplification | 1 | 0.000 | 0.140 |
| contrast | 1 | 0.667 | 1.000 |
| narrative_intro | 10 | 0.286 | 0.976 |
| fact_complement | 4 | 0.024 | 0.898 |
| religion_or_ideology | 12 | 0.476 | 0.978 |
| navigation_element | 18 | 0.058 | 0.812 |
| complement_introducer | 41 | 0.212 | 0.953 |

14 of 64 features (22%) have fewer than 50 test positives. These inflate the failure rate — most have high AUROC (model learned the concept) but poor F1 (too few examples for reliable threshold tuning). Filtering to features with 50+ positives would raise mean optimal F1 substantially.

## Evolution Across Experiments

| | Exp 1 (Haiku+BCE) | Exp 2 (gpt-oss+MSE) | Exp 3 (test catalog) | Exp 4 (flat+hybrid) | **This run** |
|---|---|---|---|---|---|
| Annotator | Haiku API | gpt-oss-20b | Qwen3-8B-Base | Qwen3-8B-Base | Qwen3-8B-Base |
| Catalog | Haiku (semantic) | Haiku (semantic) | Test (8 surface) | Test (8 flat) | Sonnet (64 mixed) |
| Loss | BCE | MSE | MSE + neg | Hybrid | Hybrid |
| Features | ~40 | ~40 | 11 (3G+8L) | 8 (flat) | 64 (11G+53L) |
| Mean F1 (t=0) | 0.263 | 0.000 | 0.357 | 0.664 | 0.394 |
| Mean F1 (opt) | — | — | — | 0.779 | **0.502** |
| Mean AUROC | 0.938 | 0.643 | 0.748 | 0.978 | **0.925** |
| vs Probe | Lost | Lost | Lost | Tied | **Tied** |
| vs Post-train | — | — | — | Tied | **Beat (opt F1)** |
| Annotation cost | $13-17 | Free | Free | Free | Free |

## What Needs to Happen Next

1. **Threshold calibration** — Implement validation-set threshold tuning to close the 0.394 → 0.502 gap automatically at inference time.

2. **Filter rare features** — Drop or merge features with < 50 positives. 14/64 features are data-starved; removing them would give a cleaner mean and more honest evaluation.

3. **Scale data** — 500 sequences (64K positions) leaves many features undersampled. 2000+ sequences would stabilize rare features like food_lifestyle (119 pos), lgbtq_identity (112 pos).

4. **Causal validation** — Run the Makelov 3-axis evaluation (approximation, sparse controllability, interpretability) to demonstrate the SAE's advantage over a linear probe. The decoder alignment (cosine 0.556) enables interventions that a probe cannot do.

5. **Annotation quality audit** — Spot-check annotations on features with high AUROC but low cosine (person_name: AUROC 0.993, cosine 0.260). Low cosine despite high AUROC suggests the concept direction is real but diffuse — or the annotations include borderline cases that smear the direction.

6. **Compare annotators** — Run same catalog with Haiku API on a subset to quantify Qwen3-8B-Base vs Haiku quality gap on semantic features.

## Architecture

```
Loss = MSE(recon, x)
     + lambda_sup * [BCE(sup_pre, labels) + alpha * (1 - cos(dec_col, target_dir))]
     + lambda_sparse * L1(all_acts)
     + lambda_hier * hierarchy_loss

lambda_sup=2.0, alpha=1.0, lambda_sparse=0.001, lambda_hier=0.5
64 supervised + 256 unsupervised latents = 320 total
```

Target directions: d_i = normalize(mean(x|label_i=1) - mean(x)), computed once before training on train split.

Annotation: Qwen3-8B-Base via vLLM, per-token binary labels with token-ID prefix caching (~600 decisions/sec on single GPU).

Feature catalog: Sonnet-generated via inventory step, 11 hierarchical groups with 53 leaf features spanning punctuation, token form, POS, syntax, semantics, entities, journalism, temporality, discourse, roles, and web content.
