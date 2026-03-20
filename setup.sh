#!/bin/bash
# vast.ai on-start script
# Env vars: OPENROUTER_API_KEY, GIT_USERNAME (opt), GIT_EMAIL (opt)

apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
ollama pull gpt-oss:20b

# repo + deps
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/Automated_Feature_Selection
cd /workspace/Automated_Feature_Selection
uv pip install --system -r pipeline/requirements.txt

# git config if provided
[ -n "$GIT_USERNAME" ] && [ -n "$GIT_EMAIL" ] && \
  git config --global user.name "$GIT_USERNAME" && \
  git config --global user.email "$GIT_EMAIL"
