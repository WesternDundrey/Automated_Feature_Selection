#!/bin/bash
# One-shot install. Run once on a fresh vast.ai instance.
# Usage: bash install.sh
set -e

echo "=== Installing pipeline dependencies ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Single uv install with override to force numpy>=2
# uv handles version resolution better than pip
echo "--- Installing all dependencies ---"
uv pip install --system \
    "vllm>=0.18" \
    "sae-lens>=5.0" \
    "transformer-lens>=2.8" \
    "datasets>=3.0" \
    "huggingface_hub>=0.27" \
    "openai>=1.60" \
    "tqdm>=4.66" \
    --override <(echo "numpy>=2.0")

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
