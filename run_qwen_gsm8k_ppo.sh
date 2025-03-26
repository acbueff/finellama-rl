#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project/account name
#SBATCH --gpus=1 -C "fat"          # Request 1 GPU on fat partition for 80GB VRAM
#SBATCH --mem=0                     # Request all available memory on the node
#SBATCH -t 0-48:00:00              # Increased maximum runtime to 48 hours
#SBATCH -J qwen_gsm8k_ppo          # Job name
#SBATCH -o logs/qwen_gsm8k_ppo_%j.out  # Standard output log
#SBATCH -e logs/qwen_gsm8k_ppo_%j.err  # Standard error log

# Source environment variables if .env file exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. HF_TOKEN may not be set."
    echo "This might cause authentication issues with the Hugging Face API."
fi

# Set HuggingFace environment variables to use storage appropriately
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/ppo_finetuned
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results
mkdir -p logs

# Define path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Define paths
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
PPO_CONFIG="configs/qwen_gsm8k_ppo_config.yaml"
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/ppo_finetuned"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results"

echo "Starting PPO training for Qwen2.5 3B on GSM8K"
echo "Model config: $MODEL_CONFIG"
echo "PPO config: $PPO_CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Evaluation data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Using container: $APPTAINER_ENV"

# Ensure the script is executable
chmod +x qwen_gsm8k_ppo_train.py

# Run the training script within the Apptainer container
echo "Running training script inside container..."
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  ${APPTAINER_ENV} python qwen_gsm8k_ppo_train.py \
  --model_config $MODEL_CONFIG \
  --ppo_config $PPO_CONFIG \
  --train_data $TRAIN_DATA \
  --eval_data $EVAL_DATA \
  --output_dir $OUTPUT_DIR \
  --results_dir $RESULTS_DIR \
  --eval_steps 200 \
  --eval_examples 50 \
  --seed 42

echo "PPO training and evaluation completed"
echo "Model saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR" 