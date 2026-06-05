#!/bin/bash

# Slurm job for AUTOPS training/runs inside the prepared Enroot container.
# Submit with:
#   sbatch -p <gpu-partition> scripts/run_autops_rl.sbatch

##SBATCH -p <gpu-partition>
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
CONTAINER_NAME="autops-agentic-framework"

# Default PPO training command.
AUTOPS_CONFIG="configs/experiments/eventsat_sas_sda_subm_le_ah.yaml"
AUTOPS_TIMESTEPS="100000"
TRAIN_COMMAND="uv run --no-sync autops train $AUTOPS_CONFIG --timesteps $AUTOPS_TIMESTEPS"
EVAL_COMMAND="uv run --no-sync autops run $AUTOPS_CONFIG --analyze"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Container name: $CONTAINER_NAME"
echo "Train command: $TRAIN_COMMAND"
echo "Eval command: $EVAL_COMMAND"

cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/.container_autops_installed" ]; then
    echo "ERROR: container setup marker is missing."
    echo "Run first: sbatch scripts/setup_container_rl.sbatch"
    exit 1
fi

if ! enroot list 2>/dev/null | awk '{print $1}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: Enroot container '$CONTAINER_NAME' does not exist."
    echo "Run first: sbatch scripts/setup_container_rl.sbatch"
    exit 1
fi

nvidia-smi

enroot start --mount "$PROJECT_DIR:/workspace" "$CONTAINER_NAME" bash -c "
    set -e
    cd /workspace
    uv run python -c \"import torch; print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())\"
    $TRAIN_COMMAND
    $EVAL_COMMAND
"

echo "Run job finished."
