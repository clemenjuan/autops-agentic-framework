#!/bin/bash

# Slurm job for AUTOPS training/runs inside the prepared Enroot container.
# Submit with:
#   sbatch -p lrz-hgx-h100-94x4 scripts/run_autops_rl.sbatch
# or, if H100 is not available to your project:
#   sbatch -p lrz-hgx-a100-80x4 scripts/run_autops_rl.sbatch

##SBATCH -p lrz-hgx-h100-94x4
#SBATCH --job-name=autops-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/autops_run_%j.out
#SBATCH --error=logs/slurm/autops_run_%j.err

set -e

# Change this if your remote checkout lives somewhere else.
PROJECT_DIR="/dss/dsshome1/00/$USER/autops-agentic-framework"
CONTAINER_NAME="autops-agentic-framework-pytorch-25-02"
MARKER_FILE="$PROJECT_DIR/.container_autops_pytorch_25_02_installed"

# Default PPO training command.
AUTOPS_CONFIG="configs/experiments/eventsat_sas_sda_subm_le_ah.yaml"
AUTOPS_TIMESTEPS="100000"
TRAIN_COMMAND="autops train $AUTOPS_CONFIG --timesteps $AUTOPS_TIMESTEPS"
EVAL_COMMAND="autops run $AUTOPS_CONFIG --analyze"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Container name: $CONTAINER_NAME"
echo "Train command: $TRAIN_COMMAND"
echo "Eval command: $EVAL_COMMAND"

cd "$PROJECT_DIR"

if [ ! -f "$MARKER_FILE" ] || \
   ! enroot list 2>/dev/null | awk '{print $1}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container is not ready on this node. Running setup now..."
    bash scripts/setup_container_rl.sh
fi

nvidia-smi

enroot start --mount "$PROJECT_DIR:/workspace" "$CONTAINER_NAME" bash -c "
    set -e
    cd /workspace
    . .venv/bin/activate
    python -c \"import torch; print('torch', torch.__version__); print('torch_file', torch.__file__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 42)\"
    $TRAIN_COMMAND
    $EVAL_COMMAND
"

echo "Run job finished."
