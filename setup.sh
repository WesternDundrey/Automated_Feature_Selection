#!/bin/bash
# Automated Feature Selection — vast.ai provisioning script
#
# On-start URL:
#   https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
#
# Set in vast.ai template env vars:
#   OPENROUTER_API_KEY   — for Step 1 (Claude Sonnet feature inventory)

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

# uv (fast pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# Clone repo
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/Automated_Feature_Selection
cd /workspace/Automated_Feature_Selection

# Install all deps. Override numpy>=2 (sae-lens pins <2 but works fine with 2.x)
uv pip install --system -r pipeline/requirements.txt --overrides pipeline/overrides.txt

# Pre-download annotator model
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')" || true

# Git config if provided
[ -n "$GIT_USERNAME" ] && [ -n "$GIT_EMAIL" ] && \
  git config --global user.name "$GIT_USERNAME" && \
  git config --global user.email "$GIT_EMAIL"

echo ""
echo "Setup complete. Run:"
echo "  cd /workspace/Automated_Feature_Selection"
echo "  python -m pipeline.run --step validate-annotator --local-annotator --n_sequences 50"
