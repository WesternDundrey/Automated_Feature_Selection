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

# sae-lens and transformer-lens pin numpy<2 which conflicts with vllm.
# --no-deps skips the pin. Their real deps are installed separately.
pip install --no-deps sae-lens transformer-lens
pip install -r pipeline/requirements.txt

# Set API key for Sonnet (feature inventory step)
export OPENROUTER_API_KEY="sk-or-..."
```

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
| `--step` | (all) | Run single step: inventory, annotate, train, evaluate, causal, ioi, validate-annotator |
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

## Causal Validation (Makelov Framework)

```bash
python -m pipeline.run --step causal
```

Three-axis evaluation: approximation (sufficiency/necessity), sparse controllability (greedy feature editing), interpretability (F1 vs known attributes).

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
