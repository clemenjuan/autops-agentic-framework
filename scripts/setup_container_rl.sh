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
NVIDIA_IMAGE_TAG="25.02-py3"
NVIDIA_IMAGE_URI="docker://nvcr.io#nvidia/pytorch:$NVIDIA_IMAGE_TAG"
NVIDIA_IMAGE="$PROJECT_DIR/.enroot/nvidia+pytorch+$NVIDIA_IMAGE_TAG.sqsh"

# Container name. Keep the image version in the name to avoid reusing a
# container created from a different NVIDIA image.
CONTAINER_NAME="autops-agentic-framework-pytorch-25-02"

# Marker file used to avoid reinstalling dependencies every time.
MARKER_FILE="$PROJECT_DIR/.container_autops_pytorch_25_02_installed"

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
CREATED_CONTAINER=0
if ! container_exists; then
    echo "Creating Enroot container..."
    enroot create --name "$CONTAINER_NAME" "$NVIDIA_IMAGE"
    CREATED_CONTAINER=1
else
    echo "Container already exists."
fi

# Install dependencies if the marker is missing, or if this node just created
# a fresh Enroot container that does not yet have container-local tools.
if [ ! -f "$MARKER_FILE" ] || [ "$CREATED_CONTAINER" -eq 1 ]; then
    echo "Installing dependencies in the container..."
    enroot start --root --rw --mount "$PROJECT_DIR:/workspace" "$CONTAINER_NAME" bash -c "
        set -e
        export DEBIAN_FRONTEND=noninteractive

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
        export UV_PYTHON_DOWNLOADS=never

        SYSTEM_PYTHON=\"\$(command -v python)\"
        \"\$SYSTEM_PYTHON\" -c \"import sys, torch; print('system_python', sys.executable); print('system_torch', torch.__version__); print('system_torch_cuda', torch.version.cuda)\"

        uv venv --clear --python \"\$SYSTEM_PYTHON\" --system-site-packages .venv
        . .venv/bin/activate
        export UV_PYTHON=\"\$SYSTEM_PYTHON\"
        python -c \"import sys; print('venv_python', sys.executable)\"

        # Do not sync the rl extra here: it contains torch, which must come
        # from the NVIDIA container to match the container CUDA stack.
        uv sync --active --no-managed-python --extra dev --extra orbital --extra llm
        uv pip install --python .venv/bin/python 'numpy<2' gymnasium 'ray[rllib]==2.44.0'

        java -version 2>&1 | head -1
        python -c \"import torch, ray; print('torch', torch.__version__); print('torch_file', torch.__file__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('ray', ray.__version__); raise SystemExit(0 if torch.cuda.is_available() else 42)\"

        touch /workspace/.container_autops_pytorch_25_02_installed
        echo 'Container setup complete with all dependencies installed!'
    "
else
    echo "Dependencies already installed. Container is ready to use."
fi

echo "Container '$CONTAINER_NAME' is ready to use!"
