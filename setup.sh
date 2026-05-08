#!/bin/bash
# vast.ai on-start script.
# Hosted at the anonymous repository mirror referenced in the ICML
# submission's "Software, Data, and Reproducibility" section.
# Set OPENROUTER_API_KEY in vast.ai template env vars.
#
# This script runs ONCE on instance start. It does NOT install python
# packages — that's install.sh's job. This is the lightweight bootstrap:
# tools, repo clone, model pre-download.

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

# Clone the (anonymized) repository. Reviewers should set REPO_URL to the
# anonymous mirror referenced in the paper. Camera-ready will use a
# direct GitHub URL.
: "${REPO_URL:?Set REPO_URL to the anonymous repository mirror}"
git clone "$REPO_URL" /workspace/Automated_Feature_Selection
cd /workspace/Automated_Feature_Selection

# Pre-download annotator model so the first vLLM cold-start doesn't
# include an 8GB download (which looks like a hang).
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Base')" || true

# Git config if provided
[ -n "$GIT_USERNAME" ] && [ -n "$GIT_EMAIL" ] && \
  git config --global user.name "$GIT_USERNAME" && \
  git config --global user.email "$GIT_EMAIL"

echo ""
echo "==========================================="
echo "Bootstrap complete. To install python deps:"
echo "==========================================="
echo "  cd /workspace/Automated_Feature_Selection"
echo "  bash install.sh        # installs all pipeline deps"
echo "                         # — preserves pre-installed vllm/torch/numpy"
echo ""
echo "Then:"
echo "  export OPENROUTER_API_KEY=sk-or-..."
echo "  python -m pipeline.run --step validate-annotator \\"
echo "    --local-annotator --n_sequences 50"
echo "==========================================="
