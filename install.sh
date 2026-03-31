#!/bin/bash
# One-shot install. Run once on a fresh vast.ai instance.
# Usage: bash install.sh
#
# Handles the numpy version conflict:
#   sae-lens pins numpy<2, vllm needs numpy>=2.
#   Solution: install vllm first (pulls numpy>=2), then sae-lens/transformer-lens
#   with --no-deps to skip their numpy<2 pin, then their real deps separately.
set -e

echo "=== Installing pipeline dependencies ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Step 1: Install vllm (brings numpy>=2 and torch)
echo "--- Step 1: vllm + numpy>=2 ---"
uv pip install --system "vllm>=0.18"

# Step 2: Install sae-lens and transformer-lens WITHOUT their deps
# (their numpy<2 pin would downgrade numpy and break vllm)
echo "--- Step 2: sae-lens + transformer-lens (--no-deps) ---"
uv pip install --system --no-deps "sae-lens>=5.0" "transformer-lens>=2.8"

# Step 3: Install the actual dependencies of sae-lens/transformer-lens
# that don't conflict with numpy>=2
echo "--- Step 3: remaining dependencies ---"
uv pip install --system \
    "datasets>=3.0" \
    "huggingface_hub>=0.27" \
    "openai>=1.60" \
    "tqdm>=4.66" \
    "jaxtyping" \
    "einops" \
    "fancy-einsum" \
    "torchtyping" \
    "typeguard<4.0" \
    "wandb" \
    "safetensors" \
    "pydantic"

# Verify all imports
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
if int(np.__version__.split('.')[0]) < 2:
    print(f'\nWARNING: numpy {np.__version__} < 2.0 — vllm may break.')
    print('Run: uv pip install --system \"numpy>=2.0\"')

print('\nAll imports successful.')
"

echo ""
echo "=== Install complete ==="
echo "Run:  python -m pipeline.run --catalog pipeline/test_catalog.json --local-annotator --n_sequences 500 --epochs 15"
