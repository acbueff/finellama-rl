#!/bin/bash
#SBATCH -A berzelius-2025-74              # Project name and partition
#SBATCH --gpus=1 -C "fat"                # 1 node with GPU
#SBATCH -t 0-12:00:00                    # Maximum runtime of 12 hours
#SBATCH -J gsm8k_eval                    # Job name
#SBATCH -o /home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.out   # Standard output log
#SBATCH -e /home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.err   # Standard error log

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k
mkdir -p /home/x_anbue/finellama-rl/logs

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Model and configuration paths
MODEL_ID="HuggingFaceTB/FineMath-Llama-3B"  # The original FineMath model
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
BATCH_SIZE=8
MAX_TOKENS=256
NUM_EXAMPLES="50"  # Set to -1 for all examples

# Get the current directory (project root)
PROJECT_ROOT="/home/x_anbue/finellama-rl"
cd $PROJECT_ROOT
echo "Project root directory: $PROJECT_ROOT"

# Make sure the custom eval script is executable
chmod +x ${PROJECT_ROOT}/gsm8k_fix_wrapper.py

echo "Starting GSM8K evaluation using fixed wrapper script..."
echo "Using model: $MODEL_ID"
echo "GSM8K data: $GSM8K_DATA_PATH"
echo "Number of examples: $NUM_EXAMPLES"
echo "Results will be saved to: $OUTPUT_DIR"

# Run the evaluation using our specialized wrapper
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  ${APPTAINER_ENV} \
  python ${PROJECT_ROOT}/gsm8k_fix_wrapper.py \
  --model_id ${MODEL_ID} \
  --output_dir ${OUTPUT_DIR} \
  --gsm8k_path ${GSM8K_DATA_PATH} \
  --num_examples ${NUM_EXAMPLES} \
  --max_tokens ${MAX_TOKENS} \
  --batch_size ${BATCH_SIZE} \
  --use_4bit

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 