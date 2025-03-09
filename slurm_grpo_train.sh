#!/bin/bash
#SBATCH -A berzelius-2024-329         # Project name and partition
#SBATCH --gpus=1 -C "fat"             # 1 node with GPU
#SBATCH -t 0-72:00:00                 # Maximum runtime of 72 hours
#SBATCH -J grpo_train                 # Job name
#SBATCH -o logs/grpo_train_%j.out     # Standard output log
#SBATCH -e logs/grpo_train_%j.err     # Standard error log

# Source the .env file if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. HF_TOKEN may not be set." 
    echo "This might cause authentication issues with the Hugging Face API."
fi

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
# Token should be set from .env file now

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/grpo_finetuned
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/checkpoints/grpo

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Paths and configurations
MODEL_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/finemath-llama-3b"
DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/data"
SAVE_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/grpo_finetuned"
CHECKPOINT_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/checkpoints/grpo"
RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo"
CONFIG_PATH="configs/grpo_config.yaml"

# Step 1: Execute the GRPO training script within the Apptainer environment
echo "Starting GRPO training..."
apptainer exec --nv ${APPTAINER_ENV} python3 scripts/run_grpo_training.py \
  --config ${CONFIG_PATH} \
  --model_path ${MODEL_PATH} \
  --data_dir ${DATA_PATH} \
  --output_dir ${SAVE_PATH} \
  --checkpoint_dir ${CHECKPOINT_PATH} \
  --log_dir ${RESULTS_PATH}/logs

echo "GRPO training completed. Starting evaluation..."

# Step 2: Evaluate the GRPO-finetuned model
apptainer exec --nv ${APPTAINER_ENV} python3 scripts/run_grpo_eval.py \
  --config configs/eval_config.yaml \
  --model_path ${SAVE_PATH} \
  --data_dir ${DATA_PATH} \
  --output_dir ${RESULTS_PATH} \
  --eval_gsm8k \
  --eval_math

echo "GRPO evaluation completed." 