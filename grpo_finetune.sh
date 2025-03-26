#!/bin/bash
#SBATCH -A berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0-48:00:00
#SBATCH -J grpo_qwen
#SBATCH -o logs/grpo_qwen_%j.out
#SBATCH -e logs/grpo_qwen_%j.err

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Define paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"
DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/data"
GRPO_SAVE_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/grpo_finetuned"
GRPO_RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo"

# Run GRPO training
echo "Starting GRPO fine-tuning with Qwen2.5"
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN \
  ${APPTAINER_ENV} python scripts/run_grpo_training.py \
  --config configs/grpo_config.yaml \
  --model_path ${MODEL_ID} \
  --train_data_path ${DATA_PATH}/grpo/train.json \
  --eval_data_path ${DATA_PATH}/grpo/validation.json \
  --output_dir ${GRPO_SAVE_PATH}

# Evaluate the fine-tuned model
echo "Evaluating GRPO-finetuned Qwen2.5 model on GSM8K"
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN \
  ${APPTAINER_ENV} python custom_gsm8k_eval.py \
  --model_id ${GRPO_SAVE_PATH} \
  --gsm8k_path /proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json \
  --output_dir ${GRPO_RESULTS_PATH} \
  --batch_size 8 \
  --use_4bit

echo "GRPO fine-tuning and evaluation completed" 