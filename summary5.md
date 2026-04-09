# Phase 3 — Supervised vs Pretrained SAE (v6.0, scaffold)

**Date:** 2026-04-09
**Status:** Code merged; awaiting vast.ai run to populate numbers.

## Question

Do supervised SAEs solve the three classic problems of pretrained SAEs —
feature splitting, entangled circuits, and noisy interventions — and is
**supervision itself** the causal factor, not the architecture or smaller
latent count?

## Setup

All three experiments reuse the Phase 2 Gemma-2-2B artifacts:
- `supervised_sae.pt` (60 supervised + 256 unsupervised = 316 latents)
- `activations.pt` at `blocks.20.hook_resid_post`
- `annotations.pt` (60 features from the flattened Sonnet catalog)
- Pretrained baseline: GemmaScope `gemma-scope-2b-pt-res` @ `layer_20/width_16k/average_l0_71`

Three-way comparison per experiment:

| Pool | Source | Count | Trained with supervision? |
|---|---|---|---|
| (S) | Our supervised SAE, supervised slice | 60 | **Yes** |
| (U) | Our supervised SAE, unsupervised slice | 256 | No |
| (P) | GemmaScope pretrained | 16 384 | No |

The (S) vs (U) gap is the architecture control: same SAE, same training data,
same loss *minus* the supervision term. If (U) ≈ (P) and (S) >> both, the
cause of the clean representations is **supervision**, not capacity or
architecture.

## Prerequisite fixes (v6.0)

### Fix 1 — GemmaScope pretrained SAE R²

| | v5.1 (broken) | v6.0 (fixed) |
|---|---|---|
| Pretrained SAE R² on Gemma-2-2B | `-6.85` | `TBD (expected > 0.90)` |

Cause: `evaluate.py` called `sae_lens.SAE.__call__` directly, which applies
`apply_b_dec_to_input` and/or activation normalization that don't match
GemmaScope's documented JumpReLU convention. Fix: route through
`inventory.load_sae()`, which uses the known-correct `PretrainedSAE` wrapper.

### Fix 2 — KL NaN for `comma` and `period`

| Feature | v5.1 `mean_kl` | v6.0 `mean_kl` |
|---|---|---|
| `comma` (764 active) | `NaN` | `TBD` |
| `period` (936 active) | `NaN` | `TBD` |

Cause: bfloat16 `log_softmax` underflow on Gemma's large-magnitude logits. Fix:
`.float()` cast before `log_softmax` in `causal.py`.

## Experiment A — Feature Splitting Quantification

### Headline table (to fill in after run)

| Pool | Mean top-1 coverage | Mean top-1 specificity | Median N@80% |
|---|---|---|---|
| **(S) supervised** | `TBD` | `TBD` | `TBD` |
| (U) unsupervised | `TBD` | `TBD` | `TBD` |
| (P) pretrained | `TBD` | `TBD` | `TBD` |

### Per-feature N@80% (to fill in after run)

| Feature | (S) | (U) | (P) | (S) top-1 specificity |
|---|---|---|---|---|
| `programming_keyword` | `TBD` | `TBD` | `TBD` | `TBD` |
| `digit_or_number` | `TBD` | `TBD` | `TBD` | `TBD` |
| `database_schema` | `TBD` | `TBD` | `TBD` | `TBD` |
| `place_name` | `TBD` | `TBD` | `TBD` | `TBD` |
| `food_and_cooking` | `TBD` | `TBD` | `TBD` | `TBD` |
| `section_header` | `TBD` | `TBD` | `TBD` | `TBD` |
| `markup_tag` | `TBD` | `TBD` | `TBD` | `TBD` |
| `article_or_determiner` | `TBD` | `TBD` | `TBD` | `TBD` |
| `capitalized_proper_noun` | `TBD` | `TBD` | `TBD` | `TBD` |
| `bracket_or_paren` | `TBD` | `TBD` | `TBD` | `TBD` |

### What success looks like

- (S) median N@80% = 1 (by construction — one latent is the concept)
- (U) median N@80% ≥ 3 (unsupervised latents split concepts like pretrained)
- (P) median N@80% ≥ 5 (pretrained splits even more due to 16K capacity)
- (S) top-1 specificity > 2× (U) top-1 specificity

### Output file

`pipeline_data/feature_splitting.json`

## Experiment B — Downstream Circuit Analysis

### Target behavior

Closing-bracket prediction at layer 20. Positions collected by scanning the
corpus for (seq, pos) where the next token is `)` and the preceding context
has an unmatched `(`.

### Method

Attribution patching: one forward+backward pass per sequence computes the
gradient of `logit_diff = logit(')') - mean(logit(non-bracket))` with respect
to the layer-20 residual stream. Per-latent contribution is then:

```
contribution(i, p) = acts_L[p, i] × (grad[p] · dec_col_L[i])
```

Aggregated as `total(i) = Σ_p |contribution(i, p)|`. `n_at_80` is the smallest
number of latents whose cumulative `|contribution|` reaches 80% of `Σ_i total(i)`.

Attribution patching is used instead of direct single-latent ablation because
16 384 individual ablations on the pretrained SAE would take hours; the
first-order approximation is sufficient for counting N@80%.

### Headline table (to fill in after run)

| Pool | Pool size | N@80% cumulative attribution |
|---|---|---|
| **(S) supervised** | 60 | `TBD` |
| (U) unsupervised | 256 | `TBD` |
| (P) pretrained | 16 384 | `TBD` |

Baseline `logit_diff` over ~200-400 positions: `TBD`

### Top-5 latents per pool (to fill in after run)

**Supervised top-5:**
1. `TBD`
2. `TBD`
3. `TBD`
4. `TBD`
5. `TBD`

**Unsupervised top-5:** `TBD`
**Pretrained top-5:** `TBD`

### What success looks like

- (S) concentrates in ≤ 3 latents, ideally with `bracket_or_paren` as the top contributor
- (U) spreads across 10-20 latents
- (P) spreads across 10-30 latents

### Output file

`pipeline_data/circuit_comparison.json`

## Experiment C — Intervention Precision

### Headline table (to fill in after run)

| Feature | Pool | KL on P_pos | KL on P_neg | Targeting ratio |
|---|---|---|---|---|
| `digit_or_number` | (S) | `TBD` | `TBD` | `TBD` |
| | (U) | `TBD` | `TBD` | `TBD` |
| | (P) | `TBD` | `TBD` | `TBD` |
| `markup_tag` | (S) | `TBD` | `TBD` | `TBD` |
| | (U) | `TBD` | `TBD` | `TBD` |
| | (P) | `TBD` | `TBD` | `TBD` |
| `database_schema` | (S) | `TBD` | `TBD` | `TBD` |
| | (U) | `TBD` | `TBD` | `TBD` |
| | (P) | `TBD` | `TBD` | `TBD` |
| `programming_keyword` | (S) | `TBD` | `TBD` | `TBD` |
| | (U) | `TBD` | `TBD` | `TBD` |
| | (P) | `TBD` | `TBD` | `TBD` |
| `article_or_determiner` | (S) | `TBD` | `TBD` | `TBD` |
| | (U) | `TBD` | `TBD` | `TBD` |
| | (P) | `TBD` | `TBD` | `TBD` |

### Aggregate targeting ratios

| Pool | Mean targeting ratio |
|---|---|
| **(S) supervised** | `TBD` |
| (U) unsupervised | `TBD` |
| (P) pretrained | `TBD` |

### What success looks like

- (S) targeting ratio ≥ 50 (very concentrated effect at positive positions)
- (U) and (P) targeting ratios ≤ 15 (diffuse, polysemantic interventions)
- (S) / (U) ratio ≥ (U) / (P) ratio (i.e., supervision matters more than the
  16 384 → 256 capacity reduction)

### Output file

`pipeline_data/intervention_precision.json`

## Prior work connection

- **Bricken et al. (2023), "Towards Monosemanticity."** Introduced feature
  splitting as the core problem with unsupervised SAEs — a single concept
  fractures across multiple polysemantic latents as SAE width increases. Our
  Experiment A directly quantifies this via N@80% on the 16 384-latent
  GemmaScope SAE.
- **Arditi et al. (2024), "Refusal in LLMs is mediated by a single direction."**
  Showed that a narrow behavior (refusal) reduces to one direction in the
  residual stream. Our supervised SAE generalizes this: instead of mining for
  a direction per concept post-hoc, we train one in, and verify it carries
  causal weight via `mean_kl` (Phase 2) and targeting ratio (Experiment C).
- **Makelov et al. (2024), "Principled Evaluations of SAEs."** Introduced the
  three-axis framework (approximation, sparse controllability, interpretability)
  our Phase 1 causal step implements. Phase 3 extends this with a
  within-SAE-architecture control (S vs U).

## How to reproduce

After the fixes are merged, run on a vast.ai RTX 5090 (32GB):

```bash
tmux new -s phase3
cd /workspace/Automated_Feature_Selection && git pull
export OPENROUTER_API_KEY="sk-or-..."

# Step 1: verify the two bug fixes on existing artifacts
python -m pipeline.run --step evaluate \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16
# Check evaluation.json -> pretrained_reconstruction.r2 > 0.9

python -m pipeline.run --step causal \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16
# Check causal.json -> comma/period mean_kl is finite

# Step 2: run the three experiments
python -m pipeline.run --step splitting \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16

python -m pipeline.run --step circuit \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16

python -m pipeline.run --step intervention \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16
```

Total wall time (estimated on RTX 5090, 32 GB):
- Experiment A: 10 min (encoding test set through both SAEs + greedy set cover)
- Experiment B: 20 min (one forward+backward per sequence × ~200 sequences)
- Experiment C: 15 min (2 × 3 × 5 = 30 forward-pass sweeps)

## Files

- `pipeline/feature_splitting.py` — Experiment A
- `pipeline/circuit.py` — Experiment B (attribution patching)
- `pipeline/intervention.py` — Experiment C
- `pipeline/evaluate.py` — Fix 1 (GemmaScope loader)
- `pipeline/causal.py` — Fix 2 (fp32 KL cast)
- `changes.md` — v6.0 entry with mathematical details

Once the run completes, replace all `TBD` entries in this file with the actual
numbers from the JSON outputs. Headline claims (success/failure bullets) should
be rewritten to match reality.
