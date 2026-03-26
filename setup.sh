#!/bin/bash
# Automated Feature Selection — vast.ai provisioning script
#
# Set these environment variables in the vast.ai template:
#   OPENROUTER_API_KEY   — for Step 1 (Claude Sonnet feature inventory)
#   GIT_USERNAME         — (optional) for git commits
#   GIT_EMAIL            — (optional) for git commits
#
# On-start URL:
#   https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# repo + deps
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/Automated_Feature_Selection
cd /workspace/Automated_Feature_Selection
uv pip install --system -r pipeline/requirements.txt

# vLLM first (needs numpy>=2), then sae-lens (needs numpy<2, force override)
uv pip install --system vllm>=0.18
uv pip install --system sae-lens>=5.0 transformer-lens>=2.8 --no-deps

# pre-download annotator model
python -c "from huggingface_hub import snapshot_download; snapshot_download('cyankiwi/Qwen3.5-9B-AWQ-4bit')" || true

# git config if provided
[ -n "$GIT_USERNAME" ] && [ -n "$GIT_EMAIL" ] && \
  git config --global user.name "$GIT_USERNAME" && \
  git config --global user.email "$GIT_EMAIL"

echo ""
echo "Setup complete. Run:"
echo "  cd /workspace/Automated_Feature_Selection"
echo "  python -m pipeline.run --step ioi --n_sequences 500  # validate training (Q1)"
echo "  python -m pipeline.run --local-annotator --n_sequences 2000 --epochs 10  # full run"
