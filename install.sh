#!/bin/bash
# One-shot install. Run this once on a fresh vast.ai instance.
# Usage: bash install.sh

set -e

echo "Installing dependencies..."

# Install vllm FIRST (it pins exact torch + numpy + transformers versions)
pip install vllm>=0.18

# Install sae-lens + transformer-lens (ignore their numpy<2 pin)
pip install sae-lens>=5.0 --no-deps
pip install transformer-lens>=2.8 --no-deps

# Install the actual deps that sae-lens/transformer-lens need
# (minus numpy which is already installed by vllm)
pip install jaxtyping einops fancy-einsum torchtyping typeguard wandb

# Install remaining pipeline deps
pip install datasets>=3.0 huggingface_hub>=0.27 openai>=1.60 tqdm>=4.66

# Verify
python -c "
from vllm import LLM
from sae_lens import SAE
import transformer_lens
import torch
print(f'vllm OK, torch {torch.__version__}, numpy OK')
print(f'sae_lens OK, transformer_lens OK')
print('All imports successful.')
"

echo "Done. Run:"
echo "  python -m pipeline.run --step validate-annotator --local-annotator --n_sequences 50"
