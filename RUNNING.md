# Running the Supervised SAE Pipeline

## Prerequisites

```bash
pip install -r pipeline/requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

Requires a GPU with >= 12 GB VRAM for Steps 1-2 (model loading).
Training (Step 3+) runs on ~3 GB. CPU works but is slow.

## Quick Start

### Full pipeline (Steps 1-4)

```bash
python -m pipeline
```

### Individual steps

```bash
python -m pipeline.run --step inventory    # Step 1: feature descriptions from pretrained SAE
python -m pipeline.run --step annotate     # Step 2: LLM annotation of corpus
python -m pipeline.run --step train        # Step 3: train supervised SAE
python -m pipeline.run --step evaluate     # Step 4: held-out evaluation
```

### Optional analysis steps (run after Steps 1-4)

```bash
python -m pipeline.run --step agreement    # Step 5: inter-annotator reliability (Cohen's kappa)
python -m pipeline.run --step ablation     # Step 6: ablation study (component importance)
python -m pipeline.run --step residual     # Step 7: propose new features from reconstruction error
```

## Configuration

Override defaults via CLI flags:

```bash
python -m pipeline.run --layer 16 --n_sequences 10000 --epochs 20 --lista 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | google/gemma-2-2b | Base model |
| `--layer` | 20 | Target layer |
| `--n_latents` | 500 | Latents to explain from pretrained SAE |
| `--n_sequences` | 5000 | Corpus sequences for annotation |
| `--epochs` | 15 | Training epochs |
| `--lista` | 0 | LISTA refinement steps (0 = disabled) |
| `--output_dir` | pipeline_data | Output directory |
| `--device` | cuda | Device (cuda/cpu) |
| `--seed` | 42 | Random seed |
| `--step` | (all) | Run only this step |

Or override programmatically:

```python
from pipeline.config import Config
cfg = Config(target_layer=16, n_sequences=2000, n_lista_steps=1)
```

## Resumability

Each step checks for its output files and skips if they exist.
Delete specific files to re-run individual steps:

```bash
rm pipeline_data/feature_catalog.json    # re-run inventory
rm pipeline_data/annotations.pt          # re-run annotation
rm pipeline_data/supervised_sae.pt       # re-run training
rm pipeline_data/evaluation.json         # re-run evaluation
```

Annotation has crash recovery: if interrupted mid-run, it resumes from the
last completed wave (100-sequence batches).

## Cost Estimates (Default Config)

| Step | Model | API Calls | Est. Cost |
|------|-------|-----------|-----------|
| Inventory | Sonnet 4.6 | ~40 | ~$1.90 |
| Annotation | Haiku 4.5 | ~10,000 | ~$30 |
| Agreement | Haiku 4.5 | ~400 | ~$1.20 |
| Residual | Sonnet 4.6 | 1 | ~$0.04 |
| **Total** | | ~10,440 | **~$33** |

Annotation dominates. To reduce cost:
- `--n_sequences 2000` cuts annotation to ~$12
- `--n_sequences 500` for a trial run: ~$3 total

GPU cost (vast.ai): ~$0.30-0.50/hr on RTX 4090, pipeline takes ~2-4 hours.

## Trial Run

For a quick end-to-end validation (~$2-3 API, ~15 min):

```bash
python -m pipeline.run --n_sequences 500 --n_latents 100 --epochs 5
```

## vast.ai Setup

1. Rent RTX 4090 (24 GB, ~$0.40/hr) with PyTorch template
2. Use `tmux` so SSH disconnects don't kill the run
3. Clone, install, set API key:
   ```bash
   git clone <repo-url> supsae && cd supsae
   pip install -r pipeline/requirements.txt
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
4. Run: `python -m pipeline`
5. Copy results: `scp -P <port> -r root@<ip>:~/supsae/pipeline_data/ ./pipeline_data/`
6. **Stop the instance** when done

**Cost-saving tip:** After activation extraction (tokens.pt + activations.pt saved),
the GPU is idle during annotation (~1-2 hrs of Haiku calls). You can copy these
files locally and run annotation without the GPU.

## Output Files

All outputs go to `pipeline_data/` (configurable via `--output_dir`):

| File | Step | Description |
|------|------|-------------|
| `top_activations.json` | 1 | Top-k activating contexts per SAE latent |
| `raw_descriptions.json` | 1 | Initial Claude explanations |
| `feature_catalog.json` | 1 | Hierarchical feature catalog (groups + leaves) |
| `tokens.pt` | 2 | Tokenized corpus (N, seq_len) |
| `activations.pt` | 2 | Residual stream activations (N, seq_len, d_model) |
| `annotations.pt` | 2 | Binary feature labels (N, seq_len, n_features) |
| `split_indices.pt` | 3 | Saved train/test permutation |
| `supervised_sae.pt` | 3 | Trained model state dict |
| `supervised_sae_config.pt` | 3 | Model architecture config |
| `evaluation.json` | 4 | Per-feature P/R/F1/AUROC, reconstruction R^2 |
| `agreement.json` | 5 | Cohen's kappa per feature |
| `ablation.json` | 6 | Ablation study results |
| `residual_features.json` | 7 | Proposed new features |

## Toy Validation Pipeline

The `toy/` directory contains a minimal GPT-2 validation pipeline (separate from the main pipeline).
It validates the supervised training loop on cheap, programmatically-annotated features (colors, days, months).

```bash
cd toy
python extract.py    # download wikitext-2, extract GPT-2 activations
python annotate.py   # programmatic token-level labeling
python train.py      # train supervised SAE
python evaluate.py   # evaluate
```
