#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project/account name
#SBATCH --gpus=1 -C "fat"          # Request 1 GPU on fat partition for 80GB VRAM
#SBATCH --mem=0                     # Request all available memory on the node
#SBATCH -t 0-72:00:00              # Increase maximum runtime to 72 hours
#SBATCH -J qwen_gsm8k_grpo         # Job name
#SBATCH -o logs/qwen_gsm8k_grpo_%j.out  # Standard output log
#SBATCH -e logs/qwen_gsm8k_grpo_%j.err  # Standard error log

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

# Aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Add generation timeout to prevent hanging
export GENERATION_TIMEOUT_SECONDS=20  # Reduced timeout for faster fallbacks

# Create all output directories with detailed structure
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned/checkpoints
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned/progress
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo/logs
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo/tensorboard
mkdir -p logs

# Create temporary config file with more aggressive generation settings
TEMP_CONFIG="/tmp/qwen_gsm8k_grpo_config_optimized.yaml"
cp configs/qwen_gsm8k_grpo_config.yaml $TEMP_CONFIG

# Modify config to be even more aggressive with timeouts and generation parameters
cat << EOF >> $TEMP_CONFIG

# Additional optimization settings added by run script
optimizer_config:
  offload_optimizer: true
  offload_param: true
  overlap_comm: true
  contiguous_gradients: true
  sub_group_size: 1e5
  reduce_bucket_size: 1e5
  stage3_prefetch_bucket_size: 1e5
  stage3_param_persistence_threshold: 1e5
  allgather_bucket_size: 1e5
  reduce_scatter: true

# Extremely reduced generation settings to avoid timeouts
generation_optimization:
  micro_batch_size: 1
  very_short_generation: true
  force_recompute_attn: false
  min_response_tokens: 16
  max_response_tokens: 128
  ultra_fast_sampling_params: true
  simple_greedy_if_timeout: true
EOF

echo "Created optimized config at $TEMP_CONFIG"

# Setup monitoring script to run in background
echo "Setting up monitoring..."
(
  while true; do
    date >> logs/grpo_monitoring_$(date +%Y%m%d).log
    nvidia-smi >> logs/grpo_monitoring_$(date +%Y%m%d).log
    ps aux | grep python >> logs/grpo_monitoring_$(date +%Y%m%d).log
    echo "---" >> logs/grpo_monitoring_$(date +%Y%m%d).log
    sleep 300  # Monitor every 5 minutes
  done
) &
MONITOR_PID=$!

# Define path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Define paths
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
GRPO_CONFIG=$TEMP_CONFIG  # Use our optimized config
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo"

echo "Starting GRPO training for Qwen2.5 3B on GSM8K"
echo "Model config: $MODEL_CONFIG"
echo "GRPO config: $GRPO_CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Evaluation data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Using container: $APPTAINER_ENV"

# Ensure the script is executable
chmod +x qwen_gsm8k_grpo_train.py

# Run the training script within the Apptainer container with monitoring
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
  --env GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  ${APPTAINER_ENV} python -u qwen_gsm8k_grpo_train.py \
  --model_config $MODEL_CONFIG \
  --grpo_config $GRPO_CONFIG \
  --train_data $TRAIN_DATA \
  --eval_data $EVAL_DATA \
  --output_dir $OUTPUT_DIR \
  --results_dir $RESULTS_DIR \
  --eval_steps 200 \
  --eval_examples 25 \
  --seed 42

TRAINING_EXIT_CODE=$?

# Stop monitoring script
kill $MONITOR_PID

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "GRPO training and evaluation completed successfully"
else
  echo "GRPO training exited with code $TRAINING_EXIT_CODE"
  
  # Create failure notification
  echo "Creating failure notification"
  mkdir -p "${OUTPUT_DIR}/errors"
  echo "Training failed with exit code ${TRAINING_EXIT_CODE}" > "${OUTPUT_DIR}/errors/failure_$(date +%Y%m%d_%H%M%S).txt"
fi

echo "Model saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR"

# Cleanup temporary config
rm -f $TEMP_CONFIG 