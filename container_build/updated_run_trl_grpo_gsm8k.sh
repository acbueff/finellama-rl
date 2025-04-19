#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project account
#SBATCH --gpus=1 -C "fat"                # Request 1 GPU on fat partition for 80GB VRAM
#SBATCH --mem=0                          # Request all available memory on the node
#SBATCH -t 0-72:00:00                    # Increase maximum runtime to 72 hours
#SBATCH -J trl_grpo_gsm8k                # Job name
#SBATCH -o logs/trl_grpo_gsm8k_%j.out    # Standard output log
#SBATCH -e logs/trl_grpo_gsm8k_%j.err    # Standard error log

# Enable error tracing to debug issues
set -e
set -x

# Create logs directory if it doesn't exist
mkdir -p logs

# Set HuggingFace environment variables
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned/checkpoints
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned/progress
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo/logs
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo/tensorboard

# Create TRL-optimized config 
TEMP_CONFIG="/tmp/trl_grpo_config_optimized.yaml"
cp configs/qwen_gsm8k_grpo_config.yaml $TEMP_CONFIG

# Add TRL-specific settings to config
cat << EOF >> $TEMP_CONFIG

# TRL GRPO specific settings
loss_type: "dr_grpo"  # Use the recommended Dr. GRPO loss formulation
mask_truncated_completions: true  # Better for stability
disable_reward_scaling: true  # Recommended by Dr. GRPO paper
ds3_gather_for_generation: false  # More memory-efficient, but slower generation
sync_ref_model: true  # Periodic update of reference model 
ref_model_sync_steps: 100  # Sync reference model every 100 steps
ref_model_mixup_alpha: 0.6  # Mix 60% current policy, 40% old reference policy
use_liger_loss: false  # Disable Liger-specific GRPO loss
kl_penalty_weight: 0.05  # Reduced KL penalty weight
epsilon: 0.2  # PPO clipping parameter
epsilon_high: 0.28  # Upper bound for asymmetric clipping (DAPO recommendation)
reward_weights: [1.0, 0.5, 0.2]  # Weights for reward functions [correctness, reasoning, brevity]
EOF

echo "Created TRL-optimized config at $TEMP_CONFIG"

# Define container and paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
GRPO_CONFIG="$TEMP_CONFIG"
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo"

echo "Starting TRL GRPO training for Qwen2.5 3B on GSM8K"
echo "Model config: $MODEL_CONFIG"
echo "GRPO config: $GRPO_CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Evaluation data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Using container: $APPTAINER_ENV"

# Make script executable
chmod +x trl_grpo_gsm8k_train.py

# Verify container and script
echo "Verifying container imports..."
apptainer exec --nv $APPTAINER_ENV python -c "import trl; from trl import GRPOConfig; import torch; print(f'TRL version: {trl.__version__}, PyTorch: {torch.__version__}')"

# Verify script existence
ls -la trl_grpo_gsm8k_train.py
ls -la configs/qwen2.5_3b_config.yaml
ls -la configs/qwen_gsm8k_grpo_config.yaml

# Run the training script inside the container (SIMPLIFIED, NO BACKGROUND MONITORING)
echo "Running training script inside container..."
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  --env PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  $APPTAINER_ENV \
  python -u trl_grpo_gsm8k_train.py \
    --model_config $MODEL_CONFIG \
    --grpo_config $GRPO_CONFIG \
    --train_data $TRAIN_DATA \
    --eval_data $EVAL_DATA \
    --output_dir $OUTPUT_DIR \
    --results_dir $RESULTS_DIR \
    --seed 42

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "TRL GRPO training and evaluation completed successfully"
else
  echo "TRL GRPO training exited with code $TRAINING_EXIT_CODE"
  # Create failure notification
  mkdir -p "${OUTPUT_DIR}/errors"
  echo "Training failed with exit code ${TRAINING_EXIT_CODE}" > "${OUTPUT_DIR}/errors/failure_$(date +%Y%m%d_%H%M%S).txt"
fi

echo "Model saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR"

# Cleanup
rm -f $TEMP_CONFIG 