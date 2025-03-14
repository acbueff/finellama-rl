#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0-2:00:00
#SBATCH -J model_test
#SBATCH -o logs/model_test_%j.out
#SBATCH -e logs/model_test_%j.err

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p logs

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Model ID to test
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"

echo "Starting model load test"
echo "Model: $MODEL_ID"
echo "Using container: $APPTAINER_ENV"

# Run the test script using the container
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  ${APPTAINER_ENV} \
  python test_model_load.py \
  --model_id $MODEL_ID \
  --use_4bit \
  --use_cache

echo "Test completed" 