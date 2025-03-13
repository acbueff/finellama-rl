#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --job-name=debug_rope
#SBATCH --output=/home/x_anbue/finellama-rl/logs/debug_rope_%j.out
#SBATCH --error=/home/x_anbue/finellama-rl/logs/debug_rope_%j.err

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p /home/x_anbue/finellama-rl/logs

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Get the current directory (project root)
PROJECT_ROOT="/home/x_anbue/finellama-rl"
cd $PROJECT_ROOT
echo "Project root directory: $PROJECT_ROOT"

echo "Running debug script..."
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  ${APPTAINER_ENV} \
  python ${PROJECT_ROOT}/debug_rope_scaling.py

echo "Debug script execution complete." 