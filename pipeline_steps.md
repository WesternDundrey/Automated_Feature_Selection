# Automated Feature Selection Pipeline

Implementation notes for the supervised SAE pipeline described in the proposal.
This document tracks design decisions, data flow, and mathematical details.

---

## Architecture Overview

```
pipeline/
  config.py       Configuration dataclass (all hyperparameters)
  inventory.py    Step 1: Feature inventory from pretrained SAE
  annotate.py     Step 2: Corpus preparation + LLM annotation
  train.py        Step 3: Supervised SAE training
  evaluate.py     Step 4: Held-out evaluation
  agreement.py    Step 5: Inter-annotator agreement (Cohen's kappa)
  ablation.py     Step 6: Ablation study (component importance)
  residual.py     Step 7: Explain the residual (iterative feature discovery)
  run.py          CLI orchestrator (python -m pipeline)
```

Data flows through `pipeline_data/`:
```
pipeline_data/
  top_activations.json     Top-k activating contexts per SAE latent
  raw_descriptions.json    Initial Claude explanations (one per latent)
  feature_catalog.json     Final hierarchical catalog (groups + leaves)
  tokens.pt                Tokenized corpus (N, seq_len) int64
  activations.pt           Residual stream at target layer (N, seq_len, d_model) float32
  annotations.pt           Binary feature labels (N, seq_len, n_features) float32
  split_indices.pt         Saved permutation for reproducible train/test split
  supervised_sae.pt        Trained model state dict
  supervised_sae_config.pt Model dimensions
  evaluation.json          Held-out metrics
  agreement.json           Inter-annotator agreement (Cohen's kappa per feature)
  ablation.json            Ablation study results (component importance)
  residual_features.json   Proposed new features from residual analysis
```

Every step checks for existing output files and skips if found (resumable).
Delete a specific file to re-run that step.

---

## Step 1: Feature Inventory (`inventory.py`)

### Purpose

Turn a pretrained SAE's opaque latent indices into a clean, hierarchical
dictionary of named features with precise descriptions.

### Process

1. **Load pretrained SAE.** Supports two backends:
   - `sae_lens`: `SAE.from_pretrained(release, sae_id)` — community standard,
     handles all GemmaScope / SAEBench formats.
   - Direct GemmaScope npz: downloads `params.npz` from HuggingFace
     (`google/gemma-scope-2b-pt-res`), loads W_enc, W_dec, b_enc, b_dec, threshold.
     Matches format in `agentic-delphi/delphi/sparse_coders/custom/gemmascope.py`.

2. **Select latents by firing rate.** sae_lens returns a sparsity tensor
   (log firing rate per latent). We select latents whose firing rate falls
   in [min_firing_rate, max_firing_rate] (default [0.0005, 0.1]).
   Rationale: too-rare features have insufficient data for explanation;
   too-frequent features are noise. We take up to `n_latents_to_explain`
   (default 500), sorted by firing rate descending.

3. **Collect top-activating examples.** Run the base model (Gemma-2-2B via
   transformer_lens) on the corpus. At the target layer, extract the residual
   stream and pass through the pretrained SAE encoder:

   ```
   z = JumpReLU_theta(W_enc @ (x - b_dec) + b_enc)
   ```

   For each selected latent, maintain a min-heap of top-k activating token
   contexts (21-token windows: 10 before, target, 10 after). This gives us
   the raw material for explanation.

4. **Generate initial descriptions with Claude Sonnet.** Batch latents
   into groups of 10. For each batch, build a prompt with Delphi-style
   `<<token>>` highlighting and activation strengths. Claude writes a
   precise 1-2 sentence description per latent. ~50 API calls for 500 latents.

5. **Organize into hierarchical catalog with Claude Sonnet.** Single large
   prompt with all 500 descriptions. Claude:
   - Rewrites for precision (operationally testable yes/no descriptions)
   - Groups into parent categories
   - Fills coverage gaps (if "red" exists, adds "blue", "green", etc.)
   - Removes vague/redundant features
   - Outputs JSON: `{features: [{id, description, type, parent}, ...]}`

   Target: 50-200 final features (groups + leaves).

### Key design constraint

The description IS the specification. It defines what the feature should detect.
The evaluation measures whether the supervised SAE learned what was specified.
At no point do we use pretrained SAE activations as ground truth — they are
only used for initial exploration.

---

## Step 2: Corpus Preparation & LLM Annotation (`annotate.py`)

### Purpose

Generate token-level supervision for the feature catalog.

### Process

1. **Tokenize corpus.** Load OpenWebText (streaming), tokenize `n_sequences`
   (default 5000) sequences of `seq_len` (default 128) tokens each.

2. **Extract activations.** Run the base model on the tokenized corpus.
   Extract the residual stream at the target layer hook point:
   `blocks.{layer}.hook_resid_post`. Save as float32.
   This is what the supervised SAE will reconstruct.

3. **LLM annotation.** For each sequence, ask Claude Haiku to label which
   tokens match each feature description.

   **Prompt format:**
   ```
   Token sequence (index before each token):
   [0]The [1]capital [2]of [3]France [4]is [5]Paris

   Feature definitions:
   F0 (color_words.red): Token is the color red
   F1 (color_words.blue): Token is the color blue
   ...

   Reply ONLY with JSON: {"F0": [indices], "F1": [indices], ...}
   ```

   Features are chunked into groups of `features_per_annotation_call`
   (default 50) to keep prompts manageable. Annotation runs async with
   `max_annotation_concurrency` (default 20) concurrent requests.

   **Cost estimate (5000 sequences, 200 features, 4 chunks):**
   - 20,000 Haiku calls
   - ~60M input tokens, ~4M output tokens
   - ~$20 at standard pricing (negligible with org tokens)

4. **Propagate group labels.** For each group feature, set
   `label[group] = OR(label[child] for child in group.children)`.

### Output

`annotations.pt`: (N, seq_len, n_features) float32 tensor.
Binary labels: 1.0 = feature should be active, 0.0 = not.

---

## Step 3: Training (`train.py`)

### Architecture

```
SupervisedSAE(d_model, n_supervised, n_unsupervised, n_lista_steps=0):
    pre  = W_enc * x + b_enc              in R^{n_total}
    acts = ReLU(pre)                       in R^{n_total}
    x_hat = W_dec * acts                   in R^{d_model}

    # Optional LISTA refinement (Learned ISTA, Gregor & LeCun 2010)
    for i in 1..n_lista_steps:
        residual = x - x_hat
        delta = W_enc * residual + b_enc
        pre = pre + eta_i * delta          # eta_i is learnable per step
        acts = ReLU(pre)
        x_hat = W_dec * acts

    Supervised:   acts[:n_supervised]      — one latent per feature in catalog
    Unsupervised: acts[n_supervised:]      — absorb reconstruction residual
```

Decoder columns normalized to unit norm after each optimizer step.
This prevents the sparsity penalty from being gamed by shrinking decoder norms.

**LISTA refinement** iteratively re-encodes the reconstruction residual with a
learnable step size `eta_i` per iteration. This improves sparse recovery quality
by allowing the encoder to correct its initial estimate based on reconstruction
error. Disabled by default (`n_lista_steps=0`).

### Loss

```
L = MSE(x_hat, x)
  + lambda_sup   * scale(step) * BCE_balanced(sup_pre, labels)
  + lambda_sparse * L1(all_acts)
  + lambda_hier   * scale(step) * hierarchy_loss(sup_acts)
```

Where:
- `scale(step) = min(1, step / warmup_steps)` — linear ramp of supervised loss
- `BCE_balanced` uses per-feature `pos_weight = clamp(n_neg / n_pos, 100)`
  to handle class imbalance (most features are rare)
- `hierarchy_loss = mean_over_pairs(ReLU(max_child_act - parent_act))`
  penalizes children activating more strongly than their parent group

### Hyperparameters (defaults)

| Symbol | Value | Purpose |
|--------|-------|---------|
| n_unsupervised | 256 | Free latents for reconstruction residual |
| epochs | 15 | Training epochs |
| batch_size | 512 | Minibatch size |
| lr | 3e-4 | AdamW learning rate |
| lambda_sup | 2.0 | Supervised BCE weight |
| lambda_sparse | 1e-3 | L1 sparsity penalty |
| lambda_hier | 0.5 | Hierarchy consistency weight |
| warmup_steps | 500 | Linear ramp before full supervised loss |
| pos_weight cap | 100 | Max class imbalance correction |
| n_lista_steps | 0 | LISTA refinement iterations (0 = disabled) |

### Train/test split

Default 80/20 split at the token-position level (not sequence level, since
individual positions are i.i.d. inputs to the SAE). Split indices are saved
to `split_indices.pt` so that evaluation uses exactly the same split
regardless of PyTorch version or RNG state.

### Learning rate schedule

Cosine decay: constant learning rate for the first 2/3 of training,
then cosine annealing to 0 over the final 1/3. Implemented via
`LambdaLR` with a custom schedule function.

---

## Step 4: Evaluation (`evaluate.py`)

### Metrics

All computed on held-out test data (20% of corpus).

1. **Reconstruction quality**
   - MSE: `E[||x_hat - x||^2]`
   - R^2: `1 - MSE_SAE / MSE_baseline` where baseline predicts the mean

2. **Per-feature classification** (threshold: `sigmoid(pre_act) > 0.5`, i.e., `pre_act > 0`)
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1: harmonic mean
   - AUROC: threshold-free metric via trapezoidal rule (no sklearn dependency)
   - Ground truth = LLM annotation (NOT any pretrained SAE activation)

3. **Sparsity**
   - L0: mean number of active latents per token position
   - Reported separately for supervised and total (supervised + unsupervised)

4. **Hierarchy consistency**
   - For each (parent, child) pair: what fraction of positions where the child
     is active also have `parent_act >= child_act`?

### Non-circular evaluation

The ground truth for feature classification is the LLM specification.
The description defines what the feature should detect. The annotation is an
LLM's best-effort labeling of that description. The evaluation measures whether
the SAE learned the specified behavior — not whether it matches any unsupervised
SAE or transcoder.

---

## Step 5: Inter-Annotator Agreement (`agreement.py`)

### Purpose

Measure annotation reliability by re-annotating a subset of sequences
independently and computing Cohen's kappa per feature.

### Process

1. Select a subset of sequences (default: 100).
2. Run `annotate_corpus_async` independently `agreement_n_reruns` times (default: 2).
3. Apply `propagate_group_labels` to each run.
4. Compute Cohen's kappa between the first two runs for each feature:
   ```
   kappa = (p_observed - p_expected) / (1 - p_expected)
   ```
   Where p_observed = agreement rate, p_expected = chance agreement.

### Quality classification

| Kappa | Quality | Interpretation |
|-------|---------|---------------|
| >= 0.6 | Good | Clean labels, SAE can learn reliably |
| 0.3-0.6 | Moderate | Noisy labels, SAE may struggle |
| < 0.3 | Poor | Labels too noisy to train on |

Features with poor kappa may need description refinement or should be excluded.

---

## Step 6: Ablation Study (`ablation.py`)

### Purpose

Quantify the contribution of each pipeline component by training variants
with individual components disabled.

### Variants

| Variant | Override | Tests |
|---------|----------|-------|
| baseline | (none) | Full model performance |
| no_hierarchy | lambda_hier = 0 | Value of hierarchy loss |
| no_warmup | warmup_steps = 0 | Value of supervised loss warmup |
| no_unsupervised | n_unsupervised = 0 | Value of free latents |
| no_sparsity | lambda_sparse = 0 | Value of L1 penalty |
| no_lista | n_lista_steps = 0 | Value of LISTA refinement (only if baseline uses LISTA) |

### Metrics

For each variant: R^2, mean F1, L0, hierarchy consistency.
Deltas from baseline highlight which components matter most.

---

## Step 7: Explain the Residual (`residual.py`)

### Purpose

Identify what the trained SAE fails to capture and propose new features
to reduce reconstruction error.

### Process

1. Load trained SAE and compute per-position reconstruction error on a sample.
2. Find the top-k positions with highest MSE.
3. Extract 21-token context windows around high-error positions.
4. Send contexts to Claude (Sonnet) with the existing feature catalog, asking:
   - What patterns appear in the high-error positions?
   - What 5-15 new features would reduce reconstruction error?
5. Output proposed features with rationale.

### Iterative use

The proposed features can be manually merged into `feature_catalog.json`
and the pipeline re-run from the annotation step onward. This implements
the "explain the residual" loop from the proposal.

---

## Running the Pipeline

### Prerequisites

```bash
pip install torch transformer_lens sae-lens datasets anthropic tqdm huggingface_hub
```

Set `ANTHROPIC_API_KEY` environment variable.

### Full pipeline

```bash
python -m pipeline
```

### Individual steps

```bash
python -m pipeline.run --step inventory
python -m pipeline.run --step annotate
python -m pipeline.run --step train
python -m pipeline.run --step evaluate
python -m pipeline.run --step agreement   # inter-annotator reliability
python -m pipeline.run --step ablation    # component importance
python -m pipeline.run --step residual    # propose new features
```

### Configuration overrides

```bash
python -m pipeline.run --layer 16 --n_sequences 10000 --epochs 20
```

### Programmatic usage

```python
from pipeline.config import Config
from pipeline.run import main

cfg = Config(
    model_name="google/gemma-2-2b",
    target_layer=20,
    n_sequences=5000,
    epochs=15,
)
# Edit cfg further, then run
```

---

## Vast.ai Setup Notes

### GPU Requirements

Peak VRAM is ~7 GB (during Steps 1-2, when Gemma-2-2B is loaded in bfloat16).
Training (Step 3) uses only ~3 GB (the base model is freed before training).

| GPU | VRAM | Cost (approx) | Notes |
|-----|------|---------------|-------|
| RTX 3090 | 24 GB | ~$0.20-0.40/hr | More than enough, good value |
| RTX 4090 | 24 GB | ~$0.30-0.50/hr | Recommended — fast and cheap |
| A100 40GB | 40 GB | ~$1.50-2.00/hr | Overkill for this pipeline |

An 8 GB card (e.g., RTX 3070) works if you reduce `activation_collection_batch_size`
and `corpus_batch_size` from 8 to 4. Any 12 GB+ card runs with default settings.

### Setup

1. Rent an on-demand instance with a PyTorch template (e.g., `pytorch/pytorch:2.x-cuda12.x`).
   Disk space: 50 GB minimum. Use `tmux` so SSH disconnects don't kill your run.
2. Clone the repo, install requirements:
   ```bash
   git clone <repo-url> supsae && cd supsae
   pip install -r pipeline/requirements.txt
   ```
3. Set `ANTHROPIC_API_KEY`:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
4. Run `python -m pipeline` (or step-by-step with `--step`).
5. Copy results back: `scp -P <port> -r root@<ip>:~/supsae/pipeline_data/ ./pipeline_data/`
6. **Stop the instance** when done (vast.ai dashboard → Stop). You are billed while it runs.

### Per-Step Resource Profile

| Step | GPU | API | Duration (est.) |
|------|-----|-----|-----------------|
| Inventory | Heavy (model + SAE) | ~50 Sonnet calls (~$2.50) | 30-60 min |
| Annotation | Brief (activation extraction) then idle | ~20k Haiku calls (~$20) | 1-2 hrs |
| Training | Moderate (small SAE) | None | 15-30 min |
| Evaluation | Light | None | < 5 min |
| Agreement | None (model loaded for tokenizer only) | ~200 Haiku calls (~$0.20) | 5-15 min |
| Ablation | Moderate (trains 5-6 SAE variants) | None | 1-2 hrs |
| Residual | Light | 1 Sonnet call (~$0.10) | < 5 min |

**Cost-saving tip:** The annotation step is API-bound, not GPU-bound. After activation
extraction completes (tokens.pt and activations.pt are saved), the GPU is idle during
the ~20k Haiku calls. You can run annotation locally after copying the cached files,
avoiding ~1-2 hours of GPU rental.

---

## Relation to Proposal Outline

| Proposal Section | Pipeline Step | File |
|-----------------|---------------|------|
| "Take a pretrained SAE" | Step 1 | inventory.py: load_sae() |
| "Use Delphi to generate initial explanations" | Step 1 | inventory.py: explain_features() (Claude replaces Delphi) |
| "Rewrite into short, precise descriptions" | Step 1 | inventory.py: organize_hierarchy() |
| "Organize into hierarchical dictionary" | Step 1 | inventory.py: organize_hierarchy() |
| "Enforce coverage constraints" | Step 1 | organize prompt: fill gaps / symmetry |
| "Take a large text corpus" | Step 2 | annotate.py: prepare_corpus() |
| "Ask LLM to label token positions" | Step 2 | annotate.py: annotate_corpus() |
| "Train supervised sparse dictionary" | Step 3 | train.py: train_supervised_sae() |
| "Evaluate on held-out data" | Step 4 | evaluate.py: evaluate() |
| "Explain the residual" | Step 7 | residual.py: run() |

### Not yet implemented (future extensions)

- **Bootstrap from existing SAE features**: currently we select by firing rate.
  Could instead cluster related features and propose higher-level abstractions.
- **Graded intensity targets**: currently binary (0/1). Could extend to
  continuous activation targets.
- **Cross-prompt circuit aggregation**: using the hierarchical features to
  build stable high-level circuits across different prompts.
- **SAEBench integration**: automated benchmark scoring against community baselines.

---

## Changes from v1.0 Demo Notebook

| Aspect | Demo (supervised_sae_demo.ipynb) | Pipeline (pipeline/) |
|--------|----------------------------------|---------------------|
| Scope | One circuit (rabbit -> habit) | Full SAE layer |
| Features | 5-10 from circuit tracing | 50-200 from pretrained SAE |
| Description source | Circuit context (edges, logits) | Top-activating examples |
| Organization | Simple Claude grouping | Full hierarchy with coverage constraints |
| Annotation | 250 sequences, sync | 5000 sequences, async |
| Training data | Single Colab run | Resumable, cached steps |
| Runtime | ~25 min Colab T4 | ~2-4 hours vast.ai A100 |
| API cost | <$1 | ~$20-30 |
