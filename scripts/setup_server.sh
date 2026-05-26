#!/usr/bin/env bash
# setup_server.sh — one-shot bring-up for a fresh Ubuntu 24.04 Server VM
#
# Idempotent: safe to re-run if a step fails. Each step checks before doing.
#
# Run on the server (NOT on the laptop) after first SSH:
#
#     curl -fsSL https://raw.githubusercontent.com/clemenjuan/autops-demo/main/scripts/setup_server.sh -o setup_server.sh
#     bash setup_server.sh
#
# Or, after the repo is cloned:
#     bash scripts/setup_server.sh
#
# What it does (in order):
#  1. Update apt and install: git, curl, build-essential, openjdk-17-jre-headless
#  2. Install uv (via the official installer) if not present
#  3. Clone the repo to ~/autops-demo if not already there
#  4. Verify orekit-data.zip is at repo root (errors out with scp instructions if missing)
#  5. Run `uv sync --extra dev --extra orbital --extra llm`
#  6. Run pytest (expects 574 passed, 1 skipped)
#  7. Run scripts/check_ollama.py against TUM Ollama
#
# Assumptions:
#  - You're logged in as a regular user with sudo rights (NOT root)
#  - SSH outbound works (for git clone, uv installer, apt)
#  - The repo's main branch is up-to-date with what you want to run

set -euo pipefail

REPO_URL="https://github.com/clemenjuan/autops-demo.git"
REPO_DIR="${HOME}/autops-demo"
OLLAMA_HOST_DEFAULT="https://ollama.sps.ed.tum.de"

log()  { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\n\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. apt packages
# ---------------------------------------------------------------------------
log "Updating apt and installing base packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl ca-certificates build-essential \
    openjdk-17-jre-headless \
    rsync

# Confirm JVM version
java -version 2>&1 | head -1 || { err "Java install failed"; exit 1; }

# ---------------------------------------------------------------------------
# 2. uv
# ---------------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
else
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin; add to current shell PATH
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        err "uv install failed — manually source ~/.local/bin/env or restart shell"
        exit 1
    fi
fi

# Ensure future shells have uv on PATH (idempotent)
if ! grep -q '\.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
fi

# ---------------------------------------------------------------------------
# 3. Clone repo
# ---------------------------------------------------------------------------
if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Repo already at ${REPO_DIR}; pulling latest..."
    (cd "${REPO_DIR}" && git pull --ff-only)
else
    log "Cloning ${REPO_URL} -> ${REPO_DIR}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# ---------------------------------------------------------------------------
# 4. orekit-data.zip
# ---------------------------------------------------------------------------
if [[ ! -f "${REPO_DIR}/orekit-data.zip" ]]; then
    err "orekit-data.zip not found at ${REPO_DIR}/orekit-data.zip"
    err "It's gitignored — you must transfer it from your laptop:"
    err ""
    err "    # From your LAPTOP (PowerShell):"
    err "    scp \"\$env:USERPROFILE\\autops-demo\\orekit-data.zip\" <user>@<server-ip>:~/autops-demo/"
    err ""
    err "Or download fresh from https://gitlab.orekit.org/orekit/orekit-data"
    err ""
    err "Re-run this script after the file is in place."
    exit 1
fi
log "orekit-data.zip found ($(du -h orekit-data.zip | cut -f1))"

# ---------------------------------------------------------------------------
# 5. uv sync
# ---------------------------------------------------------------------------
log "Running uv sync (this may take a few minutes on first run)..."
UV_LINK_MODE=copy uv sync --extra dev --extra orbital --extra llm

# ---------------------------------------------------------------------------
# 6. Tests
# ---------------------------------------------------------------------------
log "Running pytest..."
if uv run pytest tests/ -o "addopts=" --quiet --no-header; then
    log "Tests passed."
else
    err "Tests failed — investigate before continuing."
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Ollama sanity check
# ---------------------------------------------------------------------------
log "Checking Ollama connectivity..."
export OLLAMA_HOST="${OLLAMA_HOST:-${OLLAMA_HOST_DEFAULT}}"
if uv run python scripts/check_ollama.py; then
    log "Ollama check passed."
else
    warn "Ollama check failed. Possible causes:"
    warn "  - OLLAMA_HOST not reachable from this VM (firewall, wrong VLAN)"
    warn "  - Model qwen3.5:122b not loaded on the Ollama server"
    warn "  - Auth required (paste error into chat)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "Setup complete."
log "Next step: run the live hd sweep with"
log "    cd ${REPO_DIR}"
log "    uv run autops batch configs/experiments/eventsat_sas_sda_hybr_hd_ah.yaml --episodes 3 --steps 1440 --log-level DEBUG"
log ""
log "Add OLLAMA_HOST to ~/.bashrc if you want it set in future shells:"
log "    echo 'export OLLAMA_HOST=${OLLAMA_HOST}' >> ~/.bashrc"
