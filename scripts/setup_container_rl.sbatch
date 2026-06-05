#!/bin/bash

# Slurm job for the initial AUTOPS container setup.
# Submit from the project root with:
#   sbatch scripts/setup_container_rl.sbatch

#SBATCH -p lrz-cpu
#SBATCH --job-name=autops-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/autops_setup_%j.out
#SBATCH --error=logs/slurm/autops_setup_%j.err

set -e

# Change this if your remote checkout lives somewhere else.
PROJECT_DIR="/dss/dsshome1/00/$USER/autops-agentic-framework"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Project directory: $PROJECT_DIR"

cd "$PROJECT_DIR"
bash scripts/setup_container_rl.sh

echo "Setup job finished."
