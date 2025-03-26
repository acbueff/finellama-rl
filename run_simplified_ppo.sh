#!/bin/bash
#SBATCH -A berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0-24:00:00
#SBATCH -J ppo_simplified
#SBATCH -o logs/ppo_simplified_%j.out
#SBATCH -e logs/ppo_simplified_%j.err

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Define paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"
DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/data"
PPO_SAVE_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/ppo_finetuned"
PPO_RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/ppo"

# Make sure the directories exist
mkdir -p ${PPO_SAVE_PATH}
mkdir -p ${PPO_RESULTS_PATH}

# Run simplified PPO training
echo "Starting simplified PPO fine-tuning with Qwen2.5"
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN \
  ${APPTAINER_ENV} python custom_ppo_train.py \
  --model_id ${MODEL_ID} \
  --train_data_path ${DATA_PATH}/ppo/train.json \
  --output_dir ${PPO_SAVE_PATH} \
  --batch_size 4 \
  --num_epochs 3

# Evaluate the fine-tuned model
echo "Evaluating PPO-finetuned Qwen2.5 model on GSM8K"
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN \
  ${APPTAINER_ENV} python custom_gsm8k_eval.py \
  --model_id ${PPO_SAVE_PATH} \
  --gsm8k_path /proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json \
  --output_dir ${PPO_RESULTS_PATH} \
  --batch_size 8 \
  --use_4bit

echo "Simplified PPO fine-tuning and evaluation completed" 