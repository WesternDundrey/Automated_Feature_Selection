#!/bin/bash
# One-shot install for vast.ai instances.
# Usage: bash install.sh
#
# IMPORTANT: Does NOT touch torch, vllm, or numpy.
# Blackwell GPUs (5090, etc.) ship with custom CUDA 13.x builds
# that don't exist on PyPI. Reinstalling them breaks everything.
set -e

echo "=== Installing pipeline dependencies ==="
echo "  (preserving pre-installed torch, vllm, numpy)"

# sae-lens and transformer-lens with --no-deps to skip their numpy<2 pin
pip install --no-deps sae-lens transformer-lens

# All their real dependencies (minus torch/numpy/vllm which are pre-installed)
pip install \
    datasets \
    huggingface-hub \
    openai \
    tqdm \
    jaxtyping \
    einops \
    fancy-einsum \
    typeguard \
    wandb \
    safetensors \
    accelerate \
    beartype \
    better-abc \
    sentencepiece \
    rich \
    protobuf \
    pydantic \
    python-dotenv \
    pyyaml \
    simple-parsing \
    tenacity \
    plotly \
    nltk \
    typing-extensions \
    transformers \
    transformers-stream-generator \
    pandas

# Verify
echo ""
echo "=== Verifying ==="
python -c "
import sys
ok = True
for mod, name in [
    ('vllm', 'vllm'),
    ('sae_lens', 'sae-lens'),
    ('transformer_lens', 'transformer-lens'),
    ('torch', 'torch'),
    ('numpy', 'numpy'),
    ('datasets', 'datasets'),
    ('einops', 'einops'),
]:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name:<20} {ver}')
    except ImportError as e:
        print(f'  {name:<20} FAILED: {e}')
        ok = False

import torch, numpy as np
print(f'  torch CUDA:          {torch.version.cuda}')
print(f'  numpy:               {np.__version__}')
if not ok:
    sys.exit(1)
print('\nAll imports OK.')
"

echo ""
echo "=== Install complete ==="
