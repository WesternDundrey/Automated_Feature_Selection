#!/bin/bash
# vast.ai on-start script.
# URL: https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
# Set OPENROUTER_API_KEY in vast.ai template env vars.
#
# This script runs ONCE on instance start. It does NOT install python
# packages — that's install.sh's job. This is the lightweight bootstrap:
# tools, repo clone, model pre-download.
#
# v8.18.26: Delphi removed entirely; no Delphi clone here.

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

# supsae repo
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/Automated_Feature_Selection
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
