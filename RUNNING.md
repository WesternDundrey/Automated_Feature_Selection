# Running the Supervised SAE Pipeline

## Prerequisites

```bash
# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
uv pip install --system -r pipeline/requirements.txt

# API key for Step 1 (feature inventory uses Claude Sonnet)
export OPENROUTER_API_KEY="sk-or-..."

# HuggingFace token for gated models (Gemma-2-2B)
export HF_TOKEN="hf_..."
```

## Quick Start

### API annotation (default)

```bash
python -m pipeline.run --n_sequences 2000 --n_latents 100 --epochs 15
```

### Local annotation (Ollama + gpt-oss:20b)

```bash
# Start Ollama with gpt-oss:20b (do this once, or in onstart script)
ollama serve &
ollama pull gpt-oss:20b

# Run with local annotation — 10x more data, zero annotation API cost
python -m pipeline.run --local-annotator --n_sequences 50000 --epochs 15
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
| `--model` | google/gemma-2-2b | Target model to analyze |
| `--layer` | 20 | Target layer |
| `--n_latents` | 500 | Latents to explain from pretrained SAE |
| `--n_sequences` | 5000 | Corpus sequences for annotation |
| `--epochs` | 15 | Training epochs |
| `--lista` | 0 | LISTA refinement steps (0 = disabled) |
| `--local-annotator` | off | Use local model instead of API for annotation |
| `--annotator-model` | gpt-oss:20b | Local annotator model |
| `--annotator-backend` | ollama | Local backend: ollama, vllm, or hf |
| `--no-mse` | off | Use legacy BCE supervision instead of MSE |
| `--output_dir` | pipeline_data | Output directory |
| `--device` | cuda | Device (cuda/cpu) |
| `--seed` | 42 | Random seed |
| `--step` | (all) | Run only this step |

## Two Models, Two Roles

The pipeline uses two completely independent models:

| Role | What it does | Default | Runs on |
|------|-------------|---------|---------|
| **Target model** | The model being analyzed. Forward pass extracts activations. | Gemma-2-2B | GPU |
| **Annotator model** | Labels tokens with feature presence/absence. | Claude Haiku (API) or gpt-oss:20b (local) | API or GPU |

These never run simultaneously. The target model is freed from GPU before annotation starts.

## Validation Mode (GPT-2 Small)

To validate the pipeline against known analytical features (Makelov et al.):

```bash
python -m pipeline.run --model openai-community/gpt2 --layer 8 \
    --local-annotator --n_sequences 50000 --epochs 15
```

GPT-2 Small (124M params, d_model=768) is ~3x cheaper on memory/compute.
Features are simple (punctuation, common words, syntactic roles) making
annotation trivially easy for gpt-oss:20b. Any SAE failures are attributable
to the training pipeline, not annotation noise.

## vast.ai Setup

### Hardware

- **GPU:** RTX 5090 (32GB) or RTX 4090 (24GB)
- **Disk:** 50GB minimum, 80GB recommended (activations.pt can be 25GB+ at 50K sequences)
- **Template:** PyTorch

### Onstart Script

```bash
#!/bin/bash

export HF_TOKEN="hf_..."
export OPENROUTER_API_KEY="sk-or-..."

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Install Ollama + pull annotator model
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5
ollama pull gpt-oss:20b

# Clone repo and install deps
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/supsae
cd /workspace/supsae
uv pip install --system -r pipeline/requirements.txt

# Pre-download target model
python -c "
from huggingface_hub import login; login(token='$HF_TOKEN')
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-2-2b')
AutoModelForCausalLM.from_pretrained('google/gemma-2-2b', torch_dtype='bfloat16')
"
```

### Run

```bash
cd /workspace/supsae

# Local annotation, 50K sequences
python -m pipeline.run --local-annotator --n_sequences 50000 --epochs 15

# Or GPT-2 validation mode
python -m pipeline.run --model openai-community/gpt2 --layer 8 \
    --local-annotator --n_sequences 50000 --epochs 15
```

Use `tmux` so SSH disconnects don't kill the run.

### Copy results

```bash
scp -P <port> -r root@<ip>:/workspace/supsae/pipeline_data/ ./pipeline_data/
```

**Stop the instance when done.**

## Cost Comparison

### API annotation (default)

| Step | Model | API Calls | Est. Cost |
|------|-------|-----------|-----------|
| Inventory | Sonnet 4.6 | ~40 | ~$1.90 |
| Annotation | Haiku 4.5 | ~10,000 | ~$30 |
| **Total** | | ~10,040 | **~$32** |

### Local annotation (Ollama)

| Step | Model | API Calls | Est. Cost |
|------|-------|-----------|-----------|
| Inventory | Sonnet 4.6 | ~40 | ~$1.90 |
| Annotation | gpt-oss:20b | 0 (local) | $0 |
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
