#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Automated Feature Selection — vast.ai provisioning script
#
# Set these environment variables in the vast.ai template:
#   OPENROUTER_API_KEY   — for Step 1 (Claude Sonnet feature inventory)
#   GIT_USERNAME         — (optional) for git commits
#   GIT_EMAIL            — (optional) for git commits
#
# Usage: paste this URL as the "on-start script" in vast.ai:
#   https://raw.githubusercontent.com/WesternDundrey/Automated_Feature_Selection/main/setup.sh
# ──────────────────────────────────────────────────────────────────

echo "Starting setup"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── System packages ───────────────────────────────────────────────
echo "Installing system packages"
apt-get update -qq && apt-get install -y -qq curl git tmux > /dev/null 2>&1
echo "System packages done"

# ── uv (fast Python package manager) ─────────────────────────────
echo "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
echo "uv done"

# ── Ollama (local LLM inference) ─────────────────────────────────
echo "Installing Ollama"
curl -fsSL https://ollama.com/install.sh | sh
echo "Ollama installed"

echo "Starting Ollama daemon"
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

echo "Pulling gpt-oss:20b (annotator model, ~12GB download)"
ollama pull gpt-oss:20b
echo -e "${GREEN}gpt-oss:20b ready${NC}"

# ── Clone repo + install deps ────────────────────────────────────
echo "Cloning repo"
git clone https://github.com/WesternDundrey/Automated_Feature_Selection.git /workspace/repo
cd /workspace/repo

echo "Installing Python dependencies"
uv pip install --system -r pipeline/requirements.txt
echo -e "${GREEN}Dependencies installed${NC}"

# ── Validate environment variables ────────────────────────────────
SENTINEL="PLEASE_SET"
unset_vars=()

if [ "$OPENROUTER_API_KEY" = "$SENTINEL" ] || [ -z "$OPENROUTER_API_KEY" ]; then
    unset_vars+=("OPENROUTER_API_KEY")
fi

if [ ${#unset_vars[@]} -gt 0 ]; then
    echo -e "${RED}WARNING: The following are unset: ${unset_vars[*]}${NC}"
    echo -e "${YELLOW}Step 1 (inventory) needs OPENROUTER_API_KEY for Claude Sonnet.${NC}"
    echo -e "${YELLOW}Set it before running: export OPENROUTER_API_KEY=sk-or-...${NC}"
else
    echo -e "${GREEN}OPENROUTER_API_KEY is set${NC}"
fi

# ── Git config (optional) ────────────────────────────────────────
if [ -n "$GIT_USERNAME" ] && [ "$GIT_USERNAME" != "$SENTINEL" ] && \
   [ -n "$GIT_EMAIL" ] && [ "$GIT_EMAIL" != "$SENTINEL" ]; then
    git config --global user.name "$GIT_USERNAME"
    git config --global user.email "$GIT_EMAIL"
    echo -e "${GREEN}Git configured as $GIT_USERNAME <$GIT_EMAIL>${NC}"
else
    echo -e "${YELLOW}GIT_USERNAME / GIT_EMAIL not set — skipping git config${NC}"
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
  Setup complete"
echo ""
echo "  Target model:    GPT-2 Small (downloads on first run)"
echo "  Annotator:       gpt-oss:20b via Ollama (ready)"
echo "  Supervision:     MSE feature dictionary (Makelov et al.)"
echo ""
echo "  To run:"
echo "    cd /workspace/repo"
echo "    python -m pipeline.run --local-annotator --n_sequences 50000 --epochs 15"
echo ""
echo "  Use tmux so SSH disconnects don't kill the run."
echo "════════════════════════════════════════════════════════════"
