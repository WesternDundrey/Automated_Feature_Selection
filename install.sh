#!/bin/bash
# One-shot install. Run once on a fresh vast.ai instance.
# Usage: bash install.sh
set -e

echo "=== Installing pipeline dependencies ==="

# Step 1: vllm (pins torch, numpy>=2, transformers>=4.56)
echo "--- Step 1: vllm ---"
pip install "vllm>=0.18"

# Step 2: sae-lens + transformer-lens WITHOUT their deps
# (they pin numpy<2 which conflicts with vllm's numpy>=2)
echo "--- Step 2: sae-lens + transformer-lens (no deps) ---"
pip install "sae-lens>=5.0" "transformer-lens>=2.8" --no-deps

# Step 3: Install all their actual dependencies manually
# (from sae-lens pyproject.toml + transformer-lens pyproject.toml)
echo "--- Step 3: sub-dependencies ---"
pip install \
    jaxtyping \
    einops \
    fancy-einsum \
    typeguard \
    beartype \
    better-abc \
    rich \
    wandb \
    plotly \
    plotly-express \
    safetensors \
    sentencepiece \
    accelerate \
    protobuf \
    nltk \
    python-dotenv \
    pyyaml \
    typing-extensions \
    simple-parsing \
    tenacity \
    pandas \
    transformers-stream-generator

# Step 4: Remaining pipeline deps
echo "--- Step 4: pipeline deps ---"
pip install "datasets>=3.0" "huggingface_hub>=0.27" "openai>=1.60" "tqdm>=4.66"

# Verify
echo ""
echo "=== Verifying imports ==="
python -c "
from vllm import LLM
from sae_lens import SAE
import transformer_lens
import torch
import numpy as np
print(f'  vllm:            OK')
print(f'  sae_lens:        OK')
print(f'  transformer_lens: OK')
print(f'  torch:           {torch.__version__}')
print(f'  numpy:           {np.__version__}')
print(f'  All imports successful.')
"

echo ""
echo "=== Install complete ==="
echo "Run:  python -m pipeline.run --step validate-annotator --local-annotator --n_sequences 50"
