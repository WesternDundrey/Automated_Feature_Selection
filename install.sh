#!/bin/bash
# One-shot install for vast.ai instances.
# Usage: bash install.sh
#
# ============================================================
# CRITICAL — DO NOT REINSTALL torch / vllm / numpy.
# ============================================================
# Blackwell GPUs (5090, etc.) ship with custom CUDA 13.x builds
# baked into the vast.ai PyTorch image. Those builds are NOT on
# PyPI. Reinstalling vllm from PyPI silently replaces the
# Blackwell-custom build with a generic build that lacks the
# right kernels — manifests as EngineCore subprocess hanging
# at cold-start (after parallel-state init, before model load).
#
# History: v8.15 added orjson/blobfile/httpx to requirements.txt
# for Delphi. That edit is fine on its own, but a careless
# `pip install --force-reinstall vllm` in install instructions
# afterward tripped this exact bug. install.sh now adds Delphi
# deps directly so users never need to bypass requirements.txt
# or touch vllm.
# ============================================================
set -e

echo "=== Installing pipeline dependencies ==="
echo "  (preserving pre-installed torch, vllm, numpy)"

# sae-lens and transformer-lens with --no-deps to skip their numpy<2 pin
pip install --no-deps sae-lens transformer-lens

# All their real dependencies (minus torch/numpy/vllm which are pre-installed).
# Bundled with the v8.15-v8.18.6 Delphi runtime deps so a single bash
# install.sh covers the whole pipeline including --step delphi-score.
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

# Recurring vast.ai gotcha: torchcodec gets dragged in by some
# transformers/datasets module and tries to dlopen libavutil.so.56
# at import time. If the image doesn't have FFmpeg 4 system libs,
# every pipeline run dies with `Could not load this library:
# /venv/main/.../libtorchcodec_core4.so`. We don't need video
# decoding — uninstall it preemptively.
pip uninstall -y torchcodec 2>/dev/null || true

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
echo "=== Installing Delphi as editable package (its pyproject.toml fills any dep gaps) ==="
pip install -e "$DELPHI_DIR"

# Verify — both that imports work AND that we did NOT clobber vllm.
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

# Sanity check that vllm is the BLACKWELL-CUSTOM build, not a clobbered
# PyPI install. Pre-installed builds typically live under /venv/main/
# or similar; PyPI builds usually live under /usr/local/lib/python*/.
# Heuristic: print the install location so the user can compare.
import vllm, os
print(f'  vllm location:            {os.path.dirname(vllm.__file__)}')

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
