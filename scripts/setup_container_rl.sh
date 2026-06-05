#!/bin/bash

# AUTOPS Enroot setup for the LRZ Slurm system.
# - download/import the NVIDIA image if the .sqsh file is missing
# - create the Enroot container if it does not exist
# - install Java/Orekit system deps + Python deps with uv
# - write a marker file in the project directory

set -e

# Project directory on the server.
# Change this if your remote checkout lives somewhere else.
PROJECT_DIR="/dss/dsshome1/00/$USER/autops-agentic-framework"

# NVIDIA image to import with Enroot.
NVIDIA_IMAGE_URI="docker://nvcr.io#nvidia/pytorch:25.04-py3"
NVIDIA_IMAGE="$PROJECT_DIR/.enroot/nvidia+pytorch+25.04-py3.sqsh"

# Container name.
CONTAINER_NAME="autops-agentic-framework"

# Marker file used to avoid reinstalling dependencies every time.
MARKER_FILE="$PROJECT_DIR/.container_autops_installed"

container_exists() {
    enroot list 2>/dev/null | awk '{print $1}' | grep -qx "$CONTAINER_NAME"
}

if [ ! -f "$PROJECT_DIR/orekit-data.zip" ]; then
    echo "ERROR: orekit-data.zip not found at $PROJECT_DIR/orekit-data.zip"
    echo "Copy it to the project root before running this setup."
    exit 1
fi

# Import the NVIDIA image if it is not already present.
if [ ! -f "$NVIDIA_IMAGE" ]; then
    if [ ! -f "$HOME/enroot/.credentials" ]; then
        echo "ERROR: NVIDIA NGC credentials not found at $HOME/enroot/.credentials"
        echo "Create that file with your NGC API key before importing $NVIDIA_IMAGE_URI"
        echo 'Expected lines: machine nvcr.io login $oauthtoken password <KEY>'
        echo '                machine authn.nvidia.com login $oauthtoken password <KEY>'
        exit 1
    fi

    echo "Importing NVIDIA PyTorch image..."
    mkdir -p "$(dirname "$NVIDIA_IMAGE")"
    enroot import -o "$NVIDIA_IMAGE" "$NVIDIA_IMAGE_URI"
else
    echo "NVIDIA image already exists: $NVIDIA_IMAGE"
fi

# Create the container if it does not exist.
if ! container_exists; then
    echo "Creating Enroot container..."
    enroot create --name "$CONTAINER_NAME" "$NVIDIA_IMAGE"
else
    echo "Container already exists."
fi

# Install dependencies if the marker is missing.
if [ ! -f "$MARKER_FILE" ]; then
    echo "Installing dependencies in the container..."
    enroot start --root --rw --mount "$PROJECT_DIR:/workspace" "$CONTAINER_NAME" bash -c "
        set -e

        apt-get update -qq
        apt-get install -y -qq \
            git curl ca-certificates build-essential \
            openjdk-17-jre-headless rsync

        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH=\"/root/.local/bin:\$PATH\"
            ln -sf /root/.local/bin/uv /usr/local/bin/uv
            ln -sf /root/.local/bin/uvx /usr/local/bin/uvx || true
        fi

        cd /workspace
        export UV_LINK_MODE=copy
        uv sync --extra dev --extra rl --extra orbital --extra llm

        java -version 2>&1 | head -1
        uv run python -c \"import torch, ray; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('ray', ray.__version__)\"

        touch /workspace/.container_autops_installed
        echo 'Container setup complete with all dependencies installed!'
    "
else
    echo "Dependencies already installed. Container is ready to use."
fi

echo "Container '$CONTAINER_NAME' is ready to use!"
