# Supervised SAE — Change Log

> Rigorous, objective audit trail. Updated on every material change to `supsae/`.

---

## [v3.0] — LISTA Refinement, Ablation, Agreement, Residual Analysis, AUROC

**Date:** 2026-03-13

### Motivation

The pipeline was functional but lacked: (a) the architectural improvement most
supported by recent results (LISTA), (b) principled annotation quality measurement,
(c) ablation evidence for which components matter, (d) the "explain the residual"
iterative loop from the proposal, and (e) AUROC as a threshold-free metric.

### New Architecture: LISTA Refinement (`train.py`)

SupervisedSAE now supports optional LISTA (Learned ISTA) refinement iterations.
After the initial encode→decode pass, the model computes the reconstruction
residual, re-encodes it, and updates pre-activations with a learnable step size η:

```
pre₀ = W_enc · x + b_enc
acts₀ = ReLU(pre₀)
recon₀ = W_dec · acts₀

For i = 1..n_lista_steps:
    residual = x - recon_{i-1}
    delta = W_enc · residual + b_enc
    pre_i = pre_{i-1} + η_i · delta
    acts_i = ReLU(pre_i)
    recon_i = W_dec · acts_i
```

Each η_i is a learnable scalar (initialized to 0.1). This iterative refinement
(from Gregor & LeCun, ICML 2010) was the single largest improvement in the
synthsaebench experiments, improving F1 from 0.88 to 0.95.

Config: `n_lista_steps: int = 0` (backward compatible, 0 = disabled).
CLI: `python -m pipeline --lista 1`.

The model config checkpoint now saves `n_lista_steps` so evaluate.py can
reconstruct the correct architecture.

### New Step: Inter-Annotator Agreement (`agreement.py`)

Measures annotation reliability by re-running LLM annotation independently
on a subset of sequences and computing Cohen's kappa per feature:

```
κ = (p_observed - p_expected) / (1 - p_expected)
```

Features are classified as:
- **Good** (κ ≥ 0.6): clean labels, will train well
- **Moderate** (0.3 ≤ κ < 0.6): some noise, may need description refinement
- **Poor** (κ < 0.3): noisy labels, description is ambiguous or subjective

This is the principled answer to "but aren't the LLM labels noisy?" — it tells
you exactly which features have clean labels and which don't.

Config: `agreement_n_sequences: int = 100`, `agreement_n_reruns: int = 2`.
CLI: `python -m pipeline.run --step agreement`.

### New Step: Ablation Study (`ablation.py`)

Trains 5-6 model variants with individual components disabled:

| Variant | Override |
|---------|----------|
| baseline | (full model) |
| no_hierarchy | λ_hier = 0 |
| no_warmup | warmup_steps = 0 |
| no_unsupervised | n_unsupervised = 0 |
| no_sparsity | λ_sparse = 0 |
| no_lista | n_lista_steps = 0 (only if baseline uses LISTA) |

Outputs a comparison table with R², mean F1, L0, and hierarchy consistency,
plus deltas from baseline. This is table 1 of any paper.

CLI: `python -m pipeline.run --step ablation`.

### New Step: Explain the Residual (`residual.py`)

Implements the iterative loop from the proposal:

1. Load trained SAE and compute per-position reconstruction error
2. Find the top-k highest-error token positions
3. Extract context windows around those positions
4. Ask Claude to identify patterns in what the SAE is missing
5. Propose 5-15 new features to reduce reconstruction error

Output: `residual_features.json` with proposed features, rationales,
and error statistics. Features can be manually merged into
`feature_catalog.json` for the next iteration.

Config: `residual_n_samples: int = 500`, `residual_top_k_positions: int = 100`,
`residual_model: str = "claude-sonnet-4-6"`.
CLI: `python -m pipeline.run --step residual`.

### AUROC Added to Evaluation (`evaluate.py`)

Per-feature AUROC (Area Under the ROC Curve) is now computed alongside F1.
Uses raw pre-activation logits as scores — no threshold needed.
Implemented without sklearn dependency (trapezoidal rule on sorted scores).

Mean AUROC is reported and saved to `evaluation.json`.

### Evaluation Split Consistency Fix (`evaluate.py`)

Evaluation now uses the same seeded `torch.randperm` shuffle as training,
ensuring the test set is identical. Previously, evaluate.py used an
unshuffled sequential split while train.py used a shuffled split.

### What Did NOT Change

- Core loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Non-circular evaluation principle
- Pipeline steps 1-4 (inventory, annotate, train, evaluate)
- All existing hyperparameter defaults
- Annotation prompts and format

### New CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--lista` | int | LISTA refinement steps (default 0) |
| `--step agreement` | | Run inter-annotator agreement |
| `--step ablation` | | Run ablation study |
| `--step residual` | | Run explain-the-residual |

---

## [v2.2] — Vulnerability Fixes & Operational Hardening

**Date:** 2026-03-13

### Motivation

Systematic review identified 18 issues spanning security, data loss, correctness, and
performance. This release addresses all actionable items without changing the core
architecture or evaluation philosophy.

### Security

**1. Unsafe `torch.load` in root-level files** (`evaluate.py`)
Added `weights_only=True` to `torch.load` calls in the toy GPT-2 evaluation script.
Without this, PyTorch's pickle-based deserialization can execute arbitrary code from
a malicious `.pt` file. The `pipeline/` files already had this correctly.

**2. `.gitignore` created**
Excludes `pipeline_data/` (multi-GB tensors), `__pycache__/`, `.env`, `.DS_Store`,
Jupyter checkpoints, and cloned dependencies (`circuit-tracer/`, `agentic-delphi/`).

### Data Loss Prevention

**3. Annotation failure logging** (`annotate.py`)
When all retries are exhausted, `logger.warning()` now reports the sequence index,
chunk range, retry count, and error message. Previously, failures were entirely silent.

**4. Annotation wave checkpointing** (`annotate.py`)
After each wave of 100 sequences, partial results are saved to
`annotations_partial.pt`. If the process crashes at call 19,999 of 20,000, only the
last wave needs to be re-run. The partial file is deleted on successful completion.

**5. Dynamic `max_tokens` for annotation** (`annotate.py`)
Scaled to `max(500, n_features_in_chunk * 30)`. With 50 features per chunk, the
previous hard-coded 500 tokens could truncate the JSON response, silently zeroing
all labels for that chunk.

**6. Mid-training checkpoints** (`train.py`)
Model state is saved every 5 epochs to `supervised_sae_epoch{N}.pt`. If training
crashes at epoch 14/15, the epoch-10 checkpoint is available.

### Correctness

**7. Shuffled train/test split** (`train.py`)
The split now uses `torch.randperm(n_total)` (seeded for reproducibility) instead of
taking the first 80% in document order. OpenWebText is not randomly ordered, so the
previous split introduced distribution shift between train and test sets.

**8. Per-epoch validation monitoring** (`train.py`)
After each training epoch, reconstruction MSE, supervised BCE, and R² are computed on
the held-out test set. This detects overfitting to noisy LLM annotations before the
final evaluation step.

**9. Topologically ordered group propagation** (`annotate.py`)
Group labels (`group = OR(children)`) are now propagated bottom-up: leaf-adjacent
groups first, then their parents. Previously, nested hierarchies (group → sub-group →
leaf) could produce incorrect labels if the outer group was processed before its
sub-group children.

**10. String-aware JSON extraction** (`annotate.py`, `inventory.py`)
`_extract_json_object()` now tracks `in_string` and `escape` state so that braces
inside JSON string values (e.g., `"description": "uses {curly} braces"`) don't
corrupt the depth counter.

**11. Retry for `organize_hierarchy`** (`inventory.py`)
The single most critical Sonnet call (produces the entire feature catalog) now retries
up to 3 times with exponential backoff. Previously, a transient API failure required
re-running the full inventory step.

### Performance & Usability

**12. Vectorized activation collection** (`inventory.py`)
Replaced the O(n_latents × batch × seq) triple-nested Python loop with vectorized
candidate extraction: `(lat_acts > threshold).nonzero()` finds only positions above
the current heap minimum, then iterates only those. Reduces inner loop iterations
from ~512k per batch to typically <1k.

**13. CUDA fallback** (`config.py`)
`__post_init__` checks `torch.cuda.is_available()` and falls back to CPU with a
warning. Previously, running without a GPU produced a confusing CUDA error.

**14. Accurate tokenization progress bar** (`annotate.py`)
Progress bar now tracks collected sequences (not dataset iteration), so it correctly
shows progress toward `n_sequences`.

**15. Pinned dependency versions** (`requirements.txt`)
Added upper-bound pins to prevent breaking changes from major version bumps:
`torch>=2.0,<3.0`, `transformer_lens>=2.0,<3.0`, `sae-lens>=4.0,<6.0`,
`anthropic>=0.40.0,<1.0`.

### What Did NOT Change

- SupervisedSAE architecture (split latent space, ReLU, no decoder bias)
- Loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Evaluation metrics and non-circular evaluation principle
- Pipeline step structure (inventory → annotate → train → evaluate)
- All existing hyperparameter defaults
- Root-level toy pipeline files remain as-is (validation only, not primary)

---

## [v2.1] — Training Robustness & Reproducibility

**Date:** 2026-03-13

### Motivation

v2.0 pipeline was functionally correct but lacked reproducibility guarantees, had no
learning rate scheduling, and silently lost annotation data on transient API failures.
These are conservative, additive changes — no architecture or design philosophy changes.

### Changes

**1. Reproducibility: Random seed setting** (`config.py`, `train.py`)

New `seed` field in Config (default 42). `set_seed(cfg.seed)` is called at the start
of training, setting `random`, `numpy`, `torch`, and `torch.cuda` seeds.
CLI: `python -m pipeline --seed 123`.

**2. Cosine LR decay over final 1/3** (`train.py`)

Learning rate schedule:
```
lr(step) = lr_0                                          if step < 2T/3
lr(step) = lr_0 · ½(1 + cos(π · (step - 2T/3) / (T/3)))  if step ≥ 2T/3
```
where T = total training steps. The first 2/3 of training runs at constant lr (letting
the supervised loss warmup ramp work unimpeded). The final 1/3 cosine-decays to ~0,
stabilizing final weights. Implemented via `torch.optim.lr_scheduler.LambdaLR`.
Current lr is logged per epoch.

**3. Retry with exponential backoff for annotation** (`annotate.py`)

API calls now retry up to `annotation_max_retries` (default 3) with exponential backoff
(1s, 2s, 4s). Previously, any transient API error (rate limit, network timeout) silently
dropped that chunk's labels to zero. With 20,000 Haiku calls at ~$20, losing data to
transient failures is wasteful.

New Config fields: `annotation_max_retries: int = 3`, `annotation_retry_base_delay: float = 1.0`.

**4. Robust JSON extraction in annotation** (`annotate.py`)

Replaced fragile regex `r"\{[^{}]+\}"` (which fails on any nested braces) with a proper
brace-matching parser `_extract_json_object()`. Finds the first `{`, counts brace depth,
extracts the complete object, then calls `json.loads`. Handles nested structures correctly.

### What Did NOT Change

- SupervisedSAE architecture (split latent space, ReLU, no decoder bias)
- Loss function (MSE + class-balanced BCE + L1 + hierarchy)
- Evaluation metrics and non-circular evaluation principle
- Pipeline step structure (inventory → annotate → train → evaluate)
- All existing hyperparameter defaults

---

## [v2.0] — Automated Feature Selection Pipeline

**Date:** 2026-03-12

### Motivation

v1.0/v1.1 demonstrated the supervised SAE concept on a single cherry-picked circuit
(rabbit→habit, ~5-10 features, 250 sequences, <$1 API cost). This was a proof of concept.

v2.0 scales this to the full proposal: start from a pretrained SAE with thousands of
latents, automatically propose a clean hierarchical feature dictionary, annotate a large
corpus, and train a supervised SAE whose features correspond to the specified descriptions
by construction.

### New: `pipeline/` Package

Seven files implementing the end-to-end pipeline from the proposal:

| File | Role |
|---|---|
| `pipeline/config.py` | Configuration dataclass — all hyperparameters |
| `pipeline/inventory.py` | **Step 1.** Load pretrained SAE, explain latents (Claude Sonnet), organize into hierarchical catalog |
| `pipeline/annotate.py` | **Step 2.** Tokenize corpus, extract activations, LLM-annotate tokens (Claude Haiku, async) |
| `pipeline/train.py` | **Step 3.** Train supervised SAE with class-balanced BCE + hierarchy loss |
| `pipeline/evaluate.py` | **Step 4.** Held-out evaluation: per-feature F1, R², hierarchy consistency |
| `pipeline/run.py` | CLI orchestrator: `python -m pipeline --layer 20 --n_sequences 5000` |
| `pipeline/__main__.py` | Module entry point |

### Step 1: Feature Inventory — The Novel Part

This is what makes the pipeline more than just "supervised SAE training":

1. **Load a pretrained SAE** (GemmaScope JumpReLU via sae_lens, or direct npz).
   The pretrained SAE has d_sae latents (e.g., 16,384). Most are uninterpretable
   or redundant.

2. **Select by firing rate.** Use sae_lens sparsity metadata to pick latents in
   [min_firing_rate, max_firing_rate]. Default: [0.0005, 0.1]. Too rare = no data
   for explanation. Too frequent = noise.

3. **Collect top-activating examples.** Run the base model on the corpus, extract
   residual stream at the target layer, encode through the pretrained SAE:
   ```
   z = JumpReLU_θ(W_enc · (x − b_dec) + b_enc)
   ```
   Track top-k activating contexts per selected latent using min-heaps.

4. **Explain with Claude Sonnet.** Delphi-style prompt with `<<target>>` highlighting
   and activation strengths. Batched (10 latents per call). Produces initial
   natural-language descriptions.

5. **Organize with Claude Sonnet.** Single large prompt. Claude rewrites for
   precision, groups into hierarchy, fills coverage gaps (symmetry: if "red"
   exists → add "blue", "green", etc.), removes vague features. Target: 50-200
   final features. Outputs `feature_catalog.json`.

### Step 2: LLM Annotation at Scale

- Async annotation with `asyncio` + `anthropic.AsyncAnthropic`
- Default 20 concurrent requests, features chunked (50/call)
- 5000 sequences × ~4 chunks = 20,000 Haiku calls
- ~$20 with standard pricing; negligible with org tokens
- Group labels propagated: `group = OR(children)`
- All intermediate outputs cached to disk (resumable)

### Step 3: Training

Same architecture as v1.0 but with more features (50-200 supervised vs 5-10):

```
L = MSE(x̂, x)
  + λ_sup · ramp(step) · BCE_balanced(sup_pre, labels)
  + λ_sparse · ‖acts‖₁
  + λ_hier · ramp(step) · mean_pairs(ReLU(max_child − parent))
```

Class-balanced BCE: `pos_weight_f = clamp(n_neg_f / n_pos_f, 100)` per feature.
Hierarchy loss enforces `act(parent) ≥ act(child)`.
Decoder columns unit-normalized after each optimizer step.

Default hyperparameters: 15 epochs, lr = 3 × 10⁻⁴, warmup = 500 steps.

### Step 4: Evaluation

All metrics on held-out 20% (never seen during training):

| Metric | Formula | Measures |
|---|---|---|
| R² | 1 − MSE_SAE / MSE_mean | Reconstruction quality |
| Per-feature F1 | F1(z_f > 0, A(x, D(f))) | Does latent match its description? |
| L0 | mean active latents per position | Sparsity |
| Hierarchy consistency | P(act_parent ≥ act_child \| child active) | Structural coherence |

Ground truth remains the LLM annotation — not any pretrained SAE activation.
This preserves the non-circular evaluation principle from v1.0.

### Pretrained SAE Loading

Two backends, tried in order:

1. **sae_lens** (preferred): `SAE.from_pretrained(release, sae_id)`.
   Handles GemmaScope, SAEBench, and other standard formats.

2. **GemmaScope npz** (fallback): Direct download from
   `google/gemma-scope-2b-pt-res` via `huggingface_hub`. Loads
   W_enc (d_sae × d_model), W_dec, b_enc, b_dec, threshold.
   Format matches `agentic-delphi/delphi/sparse_coders/custom/gemmascope.py`.

### Data Flow

```
Pretrained SAE (GemmaScope 16k)
    ↓ select by firing rate
500 latents × 20 top examples
    ↓ Claude Sonnet explains (batched)
500 initial descriptions
    ↓ Claude Sonnet organizes + fills gaps
100-200 hierarchical features (feature_catalog.json)
    ↓
Corpus (5000 seqs)
    ↓ model forward pass
Activations (5000 × 128 × d_model)
    ↓ Claude Haiku annotates (async)
Labels (5000 × 128 × n_features)
    ↓
SupervisedSAE training
    ↓
Evaluation (held-out F1, R², hierarchy consistency)
```

### New: `pipeline_steps.md`

Detailed implementation documentation (design decisions, math, vast.ai setup).

### File Inventory After v2.0

| File | Role |
|---|---|
| `supervised_sae_demo.ipynb` | v1.0 demo. Cherry-picked rabbit→habit. |
| `pipeline/` | **v2.0. Automated feature selection pipeline.** |
| `pipeline_steps.md` | Implementation documentation. |
| `model.py`, `train.py`, etc. | Toy GPT-2 pipeline (validation only). |
| `circuit-tracer/` | Cloned dependency. |
| `agentic-delphi/` | Cloned dependency (Eleuther Delphi). |

### Budget Estimate (v2.0)

| Call type | Est. calls | Est. cost |
|---|---|---|
| Sonnet (explanations, 50 batches) | ~50 | ~$2 |
| Sonnet (organization, 1 call) | ~1 | ~$0.50 |
| Haiku (annotation, 20k calls) | ~20,000 | ~$20 |
| **Total** | | **~$22** |
| GPU (vast.ai A100, ~4 hrs) | | ~$8 |

---

## [v1.1] — Colab Compatibility + Progress Visualization

**Date:** 2026-03-06

### Changes to `supervised_sae_demo.ipynb`

1. **Install cell rewritten for Colab.** Previous `pip install circuit-tracer` may not
   resolve correctly. New cell: pins `numpy<2.0` (fixes `dtype size changed` binary
   incompatibility error on Colab where C extensions were compiled against numpy 1.x),
   `git clone`s circuit-tracer from source, `pip install -e ./circuit-tracer`.

2. **tqdm progress bars added to every heavy operation:**
   - Model loading: wall-clock timer
   - Circuit attribution: wall-clock timer
   - Description generation (Sonnet): tqdm over features + timer
   - Activation extraction: tqdm over 300 texts + timer
   - LLM annotation (Haiku): tqdm with live postfix stats (total positives, seq/s, ETA)
   - Training (both models): tqdm per epoch with live R², recon loss, sup loss
   - Unsupervised latent matching: tqdm over features

3. **GPU info printed at startup** (`torch.cuda.get_device_name()`, VRAM).

4. **`tqdm.auto`** imported for Colab-compatible HTML progress bars.

---

## [v1.0] — Complete Rebuild from Proposal

**Date:** 2026-03-06

### Why v0.x Was Scrapped

The v0.x experiment (rabbit_habit_supervised_sae.ipynb) had a fatal design flaw: it used
Per-Layer Transcoder (PLT) activations as ground truth while the motivating proposal claims
transcoders are mechanistically unfaithful. Using the thing you're trying to improve as your
gold standard is circular reasoning. Additionally:

- CLERP labels were empty at runtime (feature IDs from `prune_graph` did not match the
  URL-embedded CLERP map)
- Neuronpedia API returned no data for top activation examples
- The "controlled comparison" measured F1 against transcoder activations, not against the
  LLM specification that defines feature correctness

### What Changed

**New file:** `supervised_sae_demo.ipynb` — complete rewrite from the proposal.

**Deleted logic (conceptual):** No transcoder ground truth. No CLERP dependency. No
Neuronpedia API dependency.

### Design: Cherry-Picked Demonstration

The notebook is a **hyper cherry-picked** demonstration of the supervised SAE concept,
not a full evaluation framework. It shows one concrete case of the approach working:

1. **Circuit tracing** (CLT) identifies which features matter in the rabbit→habit circuit
2. **LLM description** (Claude Sonnet 4.6) writes precise, circuit-aware descriptions
   using upstream/downstream edges and the logit target — not just activation examples
3. **LLM annotation** (Claude Haiku) labels corpus tokens per description
4. **Supervised SAE training** produces latents that correspond to descriptions by construction
5. **Held-out evaluation** confirms the latent fires where the description predicts
6. **Causal intervention** ablates the latent and measures P("habit") reduction

### Non-Circular Evaluation

The ground truth is the **LLM specification itself** — not any SAE or transcoder.

Let D(f) be the natural-language description of feature f, generated by Claude Sonnet.
Let A(x, D(f)) ∈ {0, 1} be the annotation: does token x match description D(f)?
This annotation is produced by Claude Haiku.

Training: minimize `MSE(x̂, x) + λ · BCE(z_f, A(x, D(f))) + λ_s · ‖z‖₁`
where z_f is the pre-activation of supervised latent f.

Evaluation (on held-out sequences never seen during training):
- **Feature recovery:** F1(z_f > 0, A(x, D(f))) — does the SAE fire where the
  description says it should?
- **vs. Unsupervised:** max over all unsupervised latents of F1(z_k > 0, A(x, D(f)))
  — what's the best the unsupervised SAE can do for each concept?
- **Reconstruction:** R² = 1 − MSE_SAE / MSE_mean
- **Causal faithfulness:** zero-ablate supervised latent, measure ΔP(target logit)

At no point does any transcoder activation appear as a label or evaluation target.

### Architecture (same as v0.x, unchanged)

```
SupervisedSAE(d_model, n_supervised, n_unsupervised):
  pre  = W_enc · x + b_enc    ∈ ℝ^{n_total}
  acts = ReLU(pre)             ∈ ℝ^{n_total}
  x̂   = W_dec · acts          ∈ ℝ^d

  Supervised latents: acts[:n_supervised]
  Unsupervised latents: acts[n_supervised:]
  Decoder columns normalized to unit norm after each step.
```

### Training Hyperparameters

| Symbol | Value | Purpose |
|---|---|---|
| n_unsupervised | 256 | Free capacity for reconstruction residual |
| Epochs | 10 | — |
| Batch size | 512 | — |
| lr | 3 × 10⁻⁴ | AdamW |
| λ_sup | 2.0 | Class-balanced BCE weight |
| λ_sparse | 10⁻³ | L1 on all activations |
| Warmup | 300 steps | Linear ramp of supervised loss |
| pos_weight | clamp(n_neg/n_pos, 100) | Per-feature class balance |

### File Inventory After v1.0

| File | Role |
|---|---|
| `supervised_sae_demo.ipynb` | **Primary.** Cherry-picked demonstration. |
| `rabbit_habit_supervised_sae.ipynb` | Deprecated. v0.x with circular ground truth. |
| `model.py`, `train.py`, `evaluate.py`, etc. | Toy GPT-2 pipeline (validation only). |
| `circuit-tracer/` | Cloned dependency (not modified). |
| `arena_12_sections.txt` | ARENA 1.4.2 reference material. |

### Budget

| Call type | Est. calls | Est. cost |
|---|---|---|
| Sonnet (descriptions + hierarchy) | ~15 | ~$0.10 |
| Haiku (annotation, 250 seqs) | ~250 | ~$0.35 |
| **Total** | | **~$0.45** |
