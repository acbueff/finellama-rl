#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0-12:00:00
#SBATCH -J gsm8k_eval
#SBATCH -o logs/gsm8k_eval_%j.out
#SBATCH -e logs/gsm8k_eval_%j.err

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p logs
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Define paths and parameters
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
BATCH_SIZE=8
NUM_EXAMPLES=50
MAX_TOKENS=256

echo "Starting GSM8K evaluation with custom script"
echo "Model: $MODEL_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Using container: $APPTAINER_ENV"

# Run the custom evaluation script using the container
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  ${APPTAINER_ENV} \
  python custom_gsm8k_eval.py \
  --model_id $MODEL_ID \
  --gsm8k_path $GSM8K_DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --num_examples $NUM_EXAMPLES \
  --max_tokens $MAX_TOKENS \
  --batch_size $BATCH_SIZE \
  --use_4bit

echo "Evaluation completed"
echo "Results saved to $OUTPUT_DIR"
echo "Check the following files for results:"
echo "  - $OUTPUT_DIR/gsm8k_metrics.json"
echo "  - $OUTPUT_DIR/evaluation_summary.json"
echo "  - $OUTPUT_DIR/gsm8k_predictions.jsonl" 