#!/bin/bash
# Script to evaluate the baseline model on the GSM8K dataset
# This script uses the container environment for execution

# Ensure script fails on error
set -e

# Default config
MODEL_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/finemath-llama-3b"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k"
BATCH_SIZE=4
CONFIG_PATH="configs/eval_config.yaml"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --container)
      APPTAINER_ENV="$2"
      shift 2
      ;;
    --gsm8k_path)
      GSM8K_DATA_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--model_path PATH] [--output_dir DIR] [--batch_size SIZE] [--config PATH] [--container PATH] [--gsm8k_path PATH]"
      exit 1
      ;;
  esac
done

# Make sure required directories exist
mkdir -p "$OUTPUT_DIR"

echo "Starting baseline evaluation on GSM8K dataset..."
echo "Using model from: $MODEL_PATH"
echo "Using GSM8K data from: $GSM8K_DATA_PATH"
echo "Results will be saved to: $OUTPUT_DIR"
echo "Using container: $APPTAINER_ENV"

# Check if the container exists
if [ ! -f "$APPTAINER_ENV" ]; then
  echo "Error: Container file not found at $APPTAINER_ENV"
  echo "Please provide correct path with --container option"
  exit 1
fi

# Create a temporary custom eval config that specifically uses our downloaded GSM8K data
TMP_CONFIG="${OUTPUT_DIR}/tmp_gsm8k_eval_config.yaml"

# Run the Python script to create a custom config
apptainer exec --nv "$APPTAINER_ENV" python3 -c "
import yaml

# Load the original config
with open('${CONFIG_PATH}', 'r') as f:
    config = yaml.safe_load(f)

# Modify to only use GSM8K dataset
gsm8k_only = {'gsm8k': {
    'path': '${GSM8K_DATA_PATH}/gsm8k_test.json',
    'split': null,
    'subset': null,
    'max_samples': null
}}

config['eval_datasets'] = gsm8k_only

# Save the modified config
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)

print(f'Created custom GSM8K evaluation config at {TMP_CONFIG}')
"

# Run the evaluation script inside the container
apptainer exec --nv "$APPTAINER_ENV" python scripts/run_baseline_eval.py \
  --config "$TMP_CONFIG" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --datasets "gsm8k"

echo "Evaluation complete. Results saved to $OUTPUT_DIR"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 