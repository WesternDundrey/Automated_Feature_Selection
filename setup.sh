#!/bin/bash
# vast.ai on-start script.
# URL: https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
# Set OPENROUTER_API_KEY in vast.ai template env vars.

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/Automated_Feature_Selection
cd /workspace/Automated_Feature_Selection

# Pre-download annotator model
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Base')" || true

# Git config if provided
[ -n "$GIT_USERNAME" ] && [ -n "$GIT_EMAIL" ] && \
  git config --global user.name "$GIT_USERNAME" && \
  git config --global user.email "$GIT_EMAIL"

echo ""
echo "Ready. Run:"
echo "  cd /workspace/Automated_Feature_Selection"
echo "  python -m pipeline.run --step validate-annotator --local-annotator --n_sequences 50"
