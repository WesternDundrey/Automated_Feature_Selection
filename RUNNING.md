# Running the Supervised SAE Pipeline

## Prerequisites

```bash
# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
uv pip install --system -r pipeline/requirements.txt

# vLLM for local annotation (prefix caching)
uv pip install --system vllm

# API key for Step 1 (feature inventory uses Claude Sonnet via OpenRouter)
export OPENROUTER_API_KEY="sk-or-..."
```

## Quick Start

### Full pipeline (local annotation, recommended)

```bash
python -m pipeline.run --local-annotator --n_sequences 50000 --epochs 15
```

### API annotation (if no local GPU for annotator)

```bash
python -m pipeline.run --n_sequences 2000 --epochs 15
```

### Individual steps

```bash
python -m pipeline.run --step inventory    # Step 1: feature descriptions from pretrained SAE
python -m pipeline.run --step annotate     # Step 2: LLM annotation of corpus
python -m pipeline.run --step train        # Step 3: train supervised SAE
python -m pipeline.run --step evaluate     # Step 4: held-out evaluation
python -m pipeline.run --step causal       # Step 8: causal validation (zero-ablation)
```

### Optional analysis steps (run after Steps 1-4)

```bash
python -m pipeline.run --step agreement    # Step 5: inter-annotator reliability
python -m pipeline.run --step ablation     # Step 6: ablation study
python -m pipeline.run --step residual     # Step 7: propose new features from residual
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | openai-community/gpt2 | Target model to analyze |
| `--layer` | 8 | Target layer |
| `--n_latents` | 500 | Latents to explain from pretrained SAE |
| `--n_sequences` | 5000 | Corpus sequences for annotation |
| `--epochs` | 15 | Training epochs |
| `--lista` | 0 | LISTA refinement steps (0 = disabled) |
| `--local-annotator` | off | Use local model via vLLM with prefix caching |
| `--annotator-model` | Qwen/Qwen3-8B | HuggingFace model ID for local annotator |
| `--no-mse` | off | Use legacy BCE supervision instead of MSE |
| `--output_dir` | pipeline_data | Output directory |
| `--device` | cuda | Device (cuda/cpu) |
| `--seed` | 42 | Random seed |
| `--step` | (all) | Run only this step |

## Two Models, Two Roles

| Role | What it does | Default | Runs on |
|------|-------------|---------|---------|
| **Target model** | The model being analyzed. Forward pass extracts activations. | GPT-2 Small (124M) | GPU |
| **Annotator model** | Labels tokens with feature presence/absence. | Qwen3-8B (local via vLLM) | GPU |

These never run simultaneously. The target model is freed from GPU before annotation starts.

## Why GPT-2 Small?

GPT-2 Small (124M params, d_model=768) is the validation target because Makelov et al.
provide analytically derived ground-truth feature directions for it. If the supervised
SAE's learned decoder directions converge to Makelov's directions, the MSE feature
dictionary loss works. Any SAE failures are attributable to the pipeline, not annotation
noise — GPT-2 features are trivially simple (punctuation, word categories, syntactic roles)
and Qwen3-8B labels them near-perfectly.

## vast.ai Setup

### Hardware

- **GPU:** RTX 5090 (32GB) or RTX 4090 (24GB)
- **Disk:** 50GB (activations.pt ~ 20GB at 50K sequences with GPT-2's d_model=768)
- **Template:** PyTorch

### Onstart Script

Use the provisioning script directly as vast.ai on-start URL:
```
https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
```

Set `OPENROUTER_API_KEY` as an environment variable in the vast.ai template.

The script installs uv, vLLM, clones the repo, installs deps, and pre-downloads
Qwen3-8B. GPT-2 Small (~500MB) downloads on first run.

### Run

```bash
cd /workspace/Automated_Feature_Selection
python -m pipeline.run --local-annotator --n_sequences 50000 --epochs 15
```

Use `tmux` so SSH disconnects don't kill the run.

### Copy results

```bash
scp -P <port> -r root@<ip>:/workspace/Automated_Feature_Selection/pipeline_data/ ./pipeline_data/
```

**Stop the instance when done.**

## Cost

| Step | Model | API Calls | Est. Cost |
|------|-------|-----------|-----------|
| Inventory | Sonnet 4.6 | ~40 | ~$1.90 |
| Annotation | Qwen3-8B (local) | 0 | $0 |
| **Total** | | ~40 | **~$2** (+ GPU rental) |

GPU rental: ~$0.30-0.50/hr on vast.ai, pipeline takes ~4-8 hours with local annotation.

## Resumability

Each step checks for its output files and skips if they exist.
Delete specific files to re-run individual steps:

```bash
rm pipeline_data/feature_catalog.json    # re-run inventory
rm pipeline_data/annotations.pt          # re-run annotation
rm pipeline_data/supervised_sae.pt       # re-run training
rm pipeline_data/evaluation.json         # re-run evaluation
```

## Output Files

| File | Step | Description |
|------|------|-------------|
| `top_activations.json` | 1 | Top-k activating contexts per SAE latent |
| `raw_descriptions.json` | 1 | Initial Claude explanations |
| `feature_catalog.json` | 1 | Hierarchical feature catalog (groups + leaves) |
| `tokens.pt` | 2 | Tokenized corpus (N, seq_len) |
| `activations.pt` | 2 | Residual stream activations (N, seq_len, d_model) |
| `annotations.pt` | 2 | Binary feature labels (N, seq_len, n_features) |
| `target_directions.pt` | 3 | Makelov-style conditional mean directions |
| `split_indices.pt` | 3 | Saved train/test permutation |
| `supervised_sae.pt` | 3 | Trained model state dict |
| `supervised_sae_config.pt` | 3 | Model architecture config |
| `evaluation.json` | 4 | Per-feature F1/AUROC/FVE/cosine, R², baselines |
| `agreement.json` | 5 | Cohen's kappa per feature |
| `ablation.json` | 6 | Ablation study results |
| `residual_features.json` | 7 | Proposed new features |
| `causal.json` | 8 | Per-feature KL divergence from zero-ablation |
