# Running the Supervised SAE Pipeline

## Prerequisites

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
uv pip install --system -r pipeline/requirements.txt
uv pip install --system vllm

export OPENROUTER_API_KEY="sk-or-..."  # for Step 1 (Sonnet)
```

Or use the vast.ai provisioning script:
```
https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
```

## Validation First (IOI)

Before running on general text, validate the training procedure on IOI
where ground truth is known:

```bash
# Q1: Does the SAE training work? (no LLM, no API cost)
python -m pipeline.run --step ioi --n_sequences 500

# Q2: Does the LLM annotator work? (needs --local-annotator)
python -m pipeline.run --step ioi --local-annotator --n_sequences 500
```

Q1 trains a supervised SAE on IOI ground-truth labels and compares decoder
columns against Makelov's mean dictionary (cosine similarity). If cosine > 0.9,
the training works. Then Q2 tests the annotator.

## Full Pipeline

```bash
# With manual catalog (skip Sonnet inventory, no API cost for annotation)
python -m pipeline.run --catalog pipeline/gpt2_catalog.json \
    --local-annotator --n_sequences 2000 --epochs 10

# With auto-generated catalog (Sonnet inventory, ~$2 API cost)
python -m pipeline.run --local-annotator --n_latents 100 \
    --n_sequences 2000 --epochs 10

# API annotation only (no local GPU for annotator)
python -m pipeline.run --n_sequences 2000 --epochs 10
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | gpt2 | Target model (TransformerLens name) |
| `--layer` | 8 | Target layer |
| `--n_latents` | 500 | Latents to explain from pretrained SAE |
| `--n_sequences` | 5000 | Corpus sequences |
| `--epochs` | 15 | Training epochs |
| `--lista` | 0 | LISTA refinement steps |
| `--local-annotator` | off | Use local model via vLLM |
| `--annotator-model` | Qwen/Qwen3-8B-Base | HuggingFace model ID |
| `--batch-positions` | off | Full-sequence JSON output (vs per-token) |
| `--probe-gate` | 0.1 | Skip LLM for probe-confident negatives (0=disabled) |
| `--catalog` | (auto) | Manual feature catalog JSON path |
| `--no-mse` | off | Legacy BCE supervision |
| `--output_dir` | pipeline_data | Output directory |
| `--device` | cuda | Device |
| `--seed` | 42 | Random seed |
| `--step` | (all) | Run single step: inventory, annotate, train, evaluate, causal, ioi |

## Two Models, Two Roles

| Role | Default | Runs on |
|------|---------|---------|
| **Target model** (activations) | GPT-2 Small | GPU |
| **Annotator** (token labeling) | Qwen3-8B-Base via vLLM | GPU |

Never run simultaneously. Target model freed before annotation starts.

## Causal Validation (Makelov Framework)

After training, run the three-axis evaluation from Makelov et al. 2024:

```bash
python -m pipeline.run --step causal
```

- **Approximation**: sufficiency/necessity of SAE reconstructions (logit diff)
- **Sparse controllability**: greedy feature editing on IOI pairs (IIA metric)
- **Interpretability**: F1 of features against known attributes

## vast.ai Setup

- **GPU**: RTX 5090 (32GB) or RTX 4090 (24GB)
- **Disk**: 50GB
- **Template**: PyTorch

### Run

```bash
cd /workspace/Automated_Feature_Selection
python -m pipeline.run --step ioi --n_sequences 500  # validate first
python -m pipeline.run --local-annotator --n_sequences 2000 --epochs 10
```

Use `tmux` so SSH disconnects don't kill the run.

## Cost

| Step | Model | Cost |
|------|-------|------|
| Inventory | Sonnet 4.6 (API) | ~$2 |
| Annotation (local) | Qwen3-8B-Base (GPU) | $0 |
| Annotation (API) | Haiku 4.5 | ~$30 |
| IOI validation | none | $0 |

## Output Files

| File | Step | Description |
|------|------|-------------|
| `feature_catalog.json` | 1 | Hierarchical feature catalog |
| `tokens.pt` | 2 | Tokenized corpus |
| `activations.pt` | 2 | Residual stream activations |
| `annotations.pt` | 2 | Binary feature labels |
| `target_directions.pt` | 3 | Mean dictionary vectors |
| `supervised_sae.pt` | 3 | Trained model |
| `evaluation.json` | 4 | Metrics (F1, AUROC, FVE, cosine, baselines) |
| `causal.json` | 8 | Makelov three-axis results |
| `ioi_validation.json` | ioi | Q1/Q2 validation results |
