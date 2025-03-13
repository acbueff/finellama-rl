#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=gsm8k_fixed_eval
#SBATCH --output=/home/x_anbue/finellama-rl/logs/gsm8k_fixed_eval_%j.out
#SBATCH --error=/home/x_anbue/finellama-rl/logs/gsm8k_fixed_eval_%j.err

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache" 
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k
mkdir -p /home/x_anbue/finellama-rl/logs

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Configuration
MODEL_ID="HuggingFaceTB/FineMath-Llama-3B"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
NUM_EXAMPLES="50"  # Set to -1 for all examples
MAX_TOKENS="256"

# Get the current directory (project root)
PROJECT_ROOT="/home/x_anbue/finellama-rl"
cd $PROJECT_ROOT
echo "Project root directory: $PROJECT_ROOT"

# Make sure the wrapper script is executable
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
  --use_4bit

echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 