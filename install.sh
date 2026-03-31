#!/bin/bash
# One-shot install. Run once on a fresh vast.ai instance.
# Usage: bash install.sh
#
# The numpy conflict: transformer-lens pins numpy<2, vllm needs numpy>=2.
# Solution: install vllm first, then sae-lens+transformer-lens with --no-deps,
# then every single one of their real dependencies manually (minus numpy).
set -e

echo "=== Installing pipeline dependencies ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Step 1: vllm (brings torch, numpy>=2, transformers)
echo "--- Step 1/3: vllm ---"
uv pip install --system "vllm>=0.18"

# Step 2: sae-lens + transformer-lens WITHOUT deps (skip numpy<2 pin)
echo "--- Step 2/3: sae-lens + transformer-lens (--no-deps) ---"
uv pip install --system --no-deps "sae-lens>=5.0" "transformer-lens>=2.8"

# Step 3: Every dependency of sae-lens and transformer-lens, EXCEPT numpy
# Sources:
#   transformer-lens 2.18: pyproject.toml
#   sae-lens 6.39: pyproject.toml
echo "--- Step 3/3: all sub-dependencies ---"
uv pip install --system \
    "accelerate>=0.23.0" \
    "beartype>=0.14.1,<0.15.0" \
    "better-abc>=0.0.3,<0.0.4" \
    "datasets>=3.1.0" \
    "einops>=0.6.0" \
    "fancy-einsum>=0.0.3" \
    "huggingface-hub>=0.23.2,<1.0" \
    "jaxtyping>=0.2.11" \
    "protobuf>=3.20.0" \
    "rich>=12.6.0" \
    "sentencepiece" \
    "tqdm>=4.64.1" \
    "transformers>=4.57" \
    "transformers-stream-generator>=0.0.5,<0.0.6" \
    "typeguard>=4.2,<5.0" \
    "typing-extensions>=4.10.0,<5.0.0" \
    "wandb>=0.13.5" \
    "pandas>=2.1" \
    "babe>=0.0.7,<0.0.8" \
    "nltk>=3.8.1,<4.0.0" \
    "plotly>=5.19.0" \
    "plotly-express>=0.4.1" \
    "python-dotenv>=1.0.1" \
    "pyyaml>=6.0.1,<7.0.0" \
    "safetensors>=0.4.2,<1.0.0" \
    "simple-parsing>=0.1.6,<0.2.0" \
    "tenacity>=9.0.0" \
    "openai>=1.60"

# Verify
echo ""
echo "=== Verifying imports ==="
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
    ('wandb', 'wandb'),
    ('safetensors', 'safetensors'),
    ('accelerate', 'accelerate'),
    ('jaxtyping', 'jaxtyping'),
    ('rich', 'rich'),
    ('plotly', 'plotly'),
]:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name:<20} {ver}')
    except ImportError as e:
        print(f'  {name:<20} FAILED: {e}')
        ok = False

if not ok:
    print('\nSome imports failed. Check errors above.')
    sys.exit(1)

import numpy as np
v = int(np.__version__.split('.')[0])
if v < 2:
    print(f'\nERROR: numpy {np.__version__} < 2.0 — vllm will break.')
    sys.exit(1)

print(f'\nAll imports successful. numpy={np.__version__} (>=2.0 OK)')
"

echo ""
echo "=== Install complete ==="
echo "Run:  python -m pipeline.run --catalog pipeline/test_catalog.json --local-annotator --n_sequences 500 --epochs 15"
