#!/bin/bash
# One-shot install for vast.ai instances. ALWAYS uses uv pip — the user
# has a durable preference for uv over pip across all install commands
# in this project (faster, better dep resolution, fewer surprises).
# Usage: bash install.sh
#
# ============================================================
# THREE COMPARTMENTS (per the project's durable install rule):
#   1. --no-deps sae-lens transformer-lens   (skip their numpy<2 pin)
#   2. all the real deps from requirements.txt
#   3. --force-reinstall vllm                 (because compartment 1
#      and 2 leave the env in a state where vllm needs a clean
#      reinstall to resolve cleanly — don't optimize this away)
#
# torch and numpy are NOT touched: vast.ai's PyTorch CUDA 13.x image
# ships them pre-built for Blackwell. Replacing them breaks the GPU.
# vllm IS reinstalled from PyPI because there's a numpy/dep mismatch
# between the lens packages and vllm that only --force-reinstall
# resolves cleanly.
# ============================================================
set -e

# Sanity check: uv must be on PATH. If it isn't, install it via the
# official one-liner so the user doesn't have to think about it.
if ! command -v uv > /dev/null 2>&1; then
    echo "=== uv not found, installing via official installer ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin or ~/.cargo/bin depending
    # on shell; source the env file it writes so the rest of this
    # script can find it.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv > /dev/null 2>&1; then
        echo "ERROR: uv install completed but 'uv' still not on PATH."
        echo "Add ~/.local/bin to PATH manually and re-run install.sh."
        exit 1
    fi
fi
echo "=== uv: $(uv --version) ==="

# `uv pip` doesn't auto-detect the system Python venv on vast.ai
# images (it expects a project venv by default). Pass --system so it
# installs into the active Python environment, matching what plain
# `pip` did before.
UV_PIP="uv pip install --system"
UV_UNINSTALL="uv pip uninstall --system"

echo ""
echo "=== Compartment 1: sae-lens + transformer-lens (--no-deps) ==="
$UV_PIP --no-deps sae-lens transformer-lens

echo ""
echo "=== Compartment 2: pipeline deps (incl. Delphi runtime deps) ==="
# All their real dependencies (minus torch/numpy/vllm).
# Bundled with the v8.15-v8.18.6 Delphi runtime deps so a single
# bash install.sh covers the whole pipeline including
# --step delphi-score.
$UV_PIP \
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
    pandas \
    orjson \
    blobfile \
    httpx \
    faiss-cpu \
    sentence-transformers \
    aiofiles \
    "anyio>=4.8.0" \
    "asyncer>=0.0.8" \
    fire \
    flask \
    eai-sparsify \
    bitsandbytes

echo ""
echo "=== Compartment 3: vllm (--force-reinstall) ==="
$UV_PIP --force-reinstall vllm

# Recurring vast.ai gotcha: torchcodec gets dragged in by some
# transformers/datasets module and tries to dlopen libavutil.so.56
# at import time. If the image doesn't have FFmpeg 4 system libs,
# every pipeline run dies with `Could not load this library:
# /venv/main/.../libtorchcodec_core4.so`. We don't need video
# decoding — uninstall it preemptively.
$UV_UNINSTALL torchcodec 2>/dev/null || true

# Delphi (EleutherAI) for --step delphi-score and the inventory
# / promote-loop gates. Cloned alongside the supsae checkout if
# not already present, then installed editable so its own
# pyproject.toml deps fill in any gaps the explicit list above
# missed (and any future deps Delphi adds).
DELPHI_DIR="${DELPHI_DIR:-./delphi-eleutherai}"
if [ ! -d "$DELPHI_DIR" ]; then
    echo ""
    echo "=== Cloning EleutherAI/delphi to $DELPHI_DIR ==="
    git clone https://github.com/EleutherAI/delphi.git "$DELPHI_DIR"
fi
echo ""
echo "=== Installing Delphi as editable package ==="
$UV_PIP -e "$DELPHI_DIR"

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
    ('delphi.scorers.classifier.detection', 'delphi'),
    ('faiss', 'faiss-cpu'),
    ('sentence_transformers', 'sentence-transformers'),
]:
    try:
        m = __import__(mod, fromlist=['_'])
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name:<25} {ver}')
    except ImportError as e:
        print(f'  {name:<25} FAILED: {e}')
        ok = False

import torch, numpy as np
print(f'  torch CUDA:               {torch.version.cuda}')
print(f'  numpy:                    {np.__version__}')

if not ok:
    sys.exit(1)
print()
print('All imports OK.')
"

echo ""
echo "=== Install complete ==="
echo ""
echo "Next:"
echo "  1. export OPENROUTER_API_KEY=sk-or-..."
echo "  2. python -m pipeline.run --layer 9 \\"
echo "       --sae_id blocks.9.hook_resid_pre \\"
echo "       --local-annotator --full-desc --n_sequences 1000"
