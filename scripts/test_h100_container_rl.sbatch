#!/bin/bash

# Slurm job to test CUDA/PyTorch visibility on an H100 node.
# Submit from the project root with:
#   mkdir -p logs/slurm
#   sbatch scripts/test_h100_container_rl.sbatch

#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --job-name=autops-h100-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/autops_h100_test_%j.out
#SBATCH --error=logs/slurm/autops_h100_test_%j.err

set -e

# Change this if your remote checkout lives somewhere else.
PROJECT_DIR="/dss/dsshome1/00/$USER/autops-agentic-framework"
CONTAINER_NAME="autops-agentic-framework-pytorch-25-02"
MARKER_FILE="$PROJECT_DIR/.container_autops_pytorch_25_02_installed"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Project directory: $PROJECT_DIR"
echo "Container name: $CONTAINER_NAME"

cd "$PROJECT_DIR"

if [ ! -f "$MARKER_FILE" ] || \
   ! enroot list 2>/dev/null | awk '{print $1}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container is not ready on this node. Running setup now..."
    bash scripts/setup_container_rl.sh
fi

echo "Host GPU check:"
nvidia-smi

echo "Container CUDA/PyTorch check:"
enroot start --mount "$PROJECT_DIR:/workspace" "$CONTAINER_NAME" bash -lc "
    set -e
    cd /workspace
    . .venv/bin/activate

    echo 'Active Python / torch spec:'
    python -c 'import importlib.util, sys; spec = importlib.util.find_spec(\"torch\"); print(\"python\", sys.executable); print(\"torch_spec\", spec.origin if spec else None)'

    echo 'Active venv torch:'
    python -c 'import sys, torch, ray; print(\"python\", sys.executable); print(\"torch\", torch.__version__); print(\"torch_file\", torch.__file__); print(\"torch_cuda\", torch.version.cuda); print(\"cuda_available\", torch.cuda.is_available()); print(\"device_count\", torch.cuda.device_count()); print(\"device_name\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO CUDA\"); print(\"ray\", ray.__version__); raise SystemExit(0 if torch.cuda.is_available() else 42)'
"

echo "H100 CUDA/PyTorch test passed."
