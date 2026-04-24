# Running the Supervised SAE Pipeline

## vast.ai Setup

- **Template**: PyTorch (CUDA 13.x for Blackwell GPUs)
- **GPU**: RTX 5090 (32GB) or RTX 4090 (24GB)
- **Disk**: 50GB
- **On-start script**: `https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh`

The on-start script clones the repo and downloads Qwen3-4B-Base.

### Install Dependencies

**Do NOT reinstall torch, vllm, or numpy.** The vast.ai image has custom CUDA 13.x builds that don't exist on PyPI. Reinstalling them breaks Blackwell GPU support.

```bash
cd /workspace/Automated_Feature_Selection

# Step 1: sae-lens + transformer-lens WITHOUT their deps
# (they pin numpy<2 which would break vllm)
uv pip install --system --no-deps sae-lens transformer-lens

# Step 2: their actual dependencies (minus torch/numpy/vllm)
uv pip install --system -r pipeline/requirements.txt
uv pip install --system babe plotly-express

# Step 3: API key for Sonnet (feature inventory step)
export OPENROUTER_API_KEY="sk-or-..."
```

pip warnings about numpy/beartype/huggingface-hub version conflicts are safe to ignore — the code works fine with the installed versions.

## Quick Test (Manual Catalog)

```bash
python -m pipeline.run --catalog pipeline/test_catalog.json \
    --local-annotator --n_sequences 500 --epochs 15
```

8 flat surface features, hybrid BCE+direction loss. ~10 min on a 4090.

## Full Automated Pipeline (Sonnet Inventory)

```bash
python -m pipeline.run --local-annotator --full-desc \
    --n_latents 100 --n_sequences 500 --epochs 15
```

Sonnet explains 100 pretrained SAE latents → organizes into feature catalog → Qwen3-4B-Base annotates → hybrid train → evaluate.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | gpt2 | Target model (TransformerLens name) |
| `--layer` | 8 | Target layer |
| `--n_latents` | 500 | Latents to explain from pretrained SAE |
| `--n_sequences` | 5000 | Corpus sequences |
| `--epochs` | 15 | Training epochs |
| `--local-annotator` | off | Use local model via vLLM |
| `--annotator-model` | Qwen/Qwen3-4B-Base | HuggingFace model ID |
| `--full-desc` | off | Full description suffix (~10 tok, more accurate) |
| `--catalog` | (auto) | Manual feature catalog JSON path |
| `--supervision` | hybrid | Loss mode: hybrid, mse, bce |
| `--no-mse` | off | Shortcut for `--supervision bce` |
| `--step` | (all) | Run single step: inventory, annotate, train, evaluate, causal, ioi, validate-annotator, composition, layer-sweep, ... (see `pipeline/run.py --help` for the full list) |
| `--output_dir` | pipeline_data | Output directory |
| `--device` | cuda | Device |
| `--seed` | 42 | Random seed |

## Two Models, Two Roles

| Role | Default | Runs on |
|------|---------|---------|
| **Target model** (activations) | GPT-2 Small | GPU (subprocess) |
| **Annotator** (token labeling) | Qwen3-4B-Base via vLLM | GPU |

Activation extraction runs in a subprocess to isolate CUDA context. vLLM starts with a clean heap.

## Annotation Modes

| Mode | Suffix | Non-cached tokens | Accuracy |
|------|--------|-------------------|----------|
| F-index (default) | `F3? ` | ~3 | Lower (index lookup) |
| Full-desc (`--full-desc`) | `description? ` | ~8-10 | Higher (direct question) |

Both modes cache the system prefix + text tokens + token name. Only the feature question is non-cached.

## Loss Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Hybrid** (default) | `--supervision hybrid` | BCE selectivity + cosine direction alignment |
| MSE | `--supervision mse` | MSE magnitude + direction (with negative supervision) |
| BCE | `--supervision bce` | Legacy BCE only (no decoder alignment) |

## Gemma-2-2B

Same pipeline, different flags. Requires RTX 5090 (32GB) or equivalent.

```bash
python -m pipeline.run \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16 \
    --local-annotator --full-desc --flat \
    --n_sequences 1000 --epochs 15
```

Then Phase 1 validation:
```bash
python -m pipeline.run --step causal \
    --model google/gemma-2-2b --layer 20 \
    --sae_release gemma-scope-2b-pt-res \
    --sae_id "layer_20/width_16k/average_l0_71" \
    --model-dtype bfloat16
python -m pipeline.run --step ablation \
    --model google/gemma-2-2b --layer 20 \
    --model-dtype bfloat16
```

| Aspect | GPT-2 Small | Gemma-2-2B |
|--------|-------------|------------|
| d_model | 768 | 2304 |
| Layers | 12 | 26 |
| Target layer | 8 | 20 |
| SAE activation | ReLU | JumpReLU |
| dtype | float32 | bfloat16 |
| VRAM (model) | ~0.5GB | ~5GB |

The pipeline auto-adapts: `SupervisedSAE(d_model=2304)` from activation shape, GemmaScope npz loading already in `inventory.py`.

## Causal Validation (Makelov Framework)

```bash
python -m pipeline.run --step causal
```

Three-axis evaluation: approximation (sufficiency/necessity), sparse controllability (greedy feature editing), interpretability (F1 vs known attributes).

## Publication-track experiments (v8.0)

Assumes the full pipeline has run once and produced `supervised_sae.pt`, `annotations.pt`, `evaluation.json`.

### Composition (K-way joint ablation linearity)

```bash
python -m pipeline.run --step composition \
    --layer 9 --sae_id "blocks.9.hook_resid_pre"
```

Picks the top-5 causally-relevant features, ablates every `K ∈ {2, 3}` subset individually and jointly, measures `linearity = 1 - |KL_joint - Σ KL_individual| / max(...)`. Reported per pool: supervised, best-match unsupervised, best-match pretrained. For K=2 also logs decoder cosine and reports `corr(decoder_cos, linearity)`. Output: `pipeline_data/composition.json`. ~10-20 min.

### Promote loop (U→S capacity transfer)

Supersedes `--step discover-loop` (deprecated in v8.1). Uses the U slice of the already-trained supervised SAE as the proposal pool rather than training a fresh unsup SAE per round.

```bash
python -m pipeline.run --step promote-loop \
    --layer 9 --sae_id "blocks.9.hook_resid_pre" \
    --local-annotator --full-desc \
    --n_sequences 1000 --epochs 15 \
    --promote-top-k 20 --promote-max-iters 5 \
    --promote-post-train-f1-floor 0.30 \
    --promote-cos-threshold 0.6
```

Per round: rank U latents by ΔR² on val → describe top-K via Sonnet → crispness gate → merge (cosine + separability) → annotate → retrain → drop new features below post-training F1 floor → verify capacity transfer (new top-U ΔR² should fall below transfer_ratio × old). Output: `pipeline_data/promote_loop/round_{N}/` with `summary.json`, `dropped.json`, `capacity_transfer.json`, `post_training_dropped.json`, `crispness.json`.

### Layer sweep (cross-layer orchestrator)

```bash
python -m pipeline.run --step layer-sweep \
    --layers 4,6,8,9,10,11 \
    --local-annotator --full-desc \
    --n_latents 500 --n_sequences 1000 --epochs 15
```

For each layer, runs inventory → annotate → train → evaluate → causal → intervention under `pipeline_data/layer_sweep/layer_{N}/`. Idempotent: re-invoke to resume. Optional flags: `--sweep-skip-intervention`, `--sweep-skip-causal`. Output: `pipeline_data/layer_sweep/layer_sweep_summary.json`. 3-4 hr for 6 layers (full), 1.5-2 hr with `--sweep-skip-intervention`.

### IOI behavioral benchmark

```bash
python -m pipeline.run --step ioi --local-annotator --n_sequences 500
```

Generates IOI sentences with ground-truth labels, trains a supervised SAE on those labels, compares decoder columns against Makelov's mean dictionary directions. With `--local-annotator` also runs Q2 (LLM annotation vs ground-truth comparison). Output: `pipeline_data/ioi_validation.json`.

## Output Files

| File | Step | Description |
|------|------|-------------|
| `feature_catalog.json` | 1 | Feature catalog |
| `tokens.pt` | 2 | Tokenized corpus |
| `activations.pt` | 2 | Residual stream activations |
| `annotations.pt` | 2 | Binary feature labels |
| `target_directions.pt` | 3 | Mean dictionary vectors |
| `supervised_sae.pt` | 3 | Trained model |
| `evaluation.json` | 4 | Metrics (F1, AUROC, FVE, cosine, baselines) |

## Cost

| Step | Model | Cost |
|------|-------|------|
| Inventory | Sonnet 4.6 (API) | ~$2 |
| Annotation (local) | Qwen3-4B-Base (GPU) | $0 |
| Annotation (API) | Haiku 4.5 | ~$30 |

Use `tmux` so SSH disconnects don't kill the run.
