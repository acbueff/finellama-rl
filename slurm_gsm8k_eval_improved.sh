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
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k"
CONFIG_PATH="configs/eval_config.yaml"
BATCH_SIZE=8

# Get the current directory (project root)
PROJECT_ROOT="/home/x_anbue/finellama-rl"
cd $PROJECT_ROOT
echo "Project root directory: $PROJECT_ROOT"

# Create temporary config path for GSM8K evaluation
TMP_CONFIG="${OUTPUT_DIR}/tmp_gsm8k_eval_config.yaml"

echo "Starting baseline evaluation on GSM8K dataset..."
echo "Using model: $MODEL_ID"
echo "Results will be saved to: $OUTPUT_DIR"

# Step 1: Create a custom config for GSM8K only
apptainer exec --nv ${APPTAINER_ENV} python3 -c "
import yaml

# Load the original config
with open('${CONFIG_PATH}', 'r') as f:
    config = yaml.safe_load(f)

# Modify to only use GSM8K dataset
gsm8k_only = {'gsm8k': {
    'path': '${GSM8K_DATA_PATH}/gsm8k_test.json',
    'split': None,
    'subset': None,
    'max_samples': None
}}

config['eval_datasets'] = gsm8k_only

# Save the modified config
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f)

print(f'Created custom GSM8K evaluation config at ${TMP_CONFIG}')
"

# Create a monkey patch to disable rope_scaling validation
MONKEY_PATCH="${OUTPUT_DIR}/disable_rope_validation.py"

cat > ${MONKEY_PATCH} << 'EOL'
"""
Monkey patch to disable rope_scaling validation in LlamaConfig
"""
from transformers.models.llama.configuration_llama import LlamaConfig

# Store the original validation method
original_validation = LlamaConfig._rope_scaling_validation

# Replace with no-op function
def no_validation(self):
    # This function does nothing
    pass

# Apply the monkey patch
LlamaConfig._rope_scaling_validation = no_validation
print("Applied monkey patch to disable rope_scaling validation")
EOL

# Create a simple script to run the evaluation with the monkey patch
EVAL_SCRIPT="${OUTPUT_DIR}/run_with_monkey_patch.py"

cat > ${EVAL_SCRIPT} << 'EOL'
#!/usr/bin/env python
"""
Script to run the evaluation with the monkey patch applied.
"""
import os
import sys

# Apply the monkey patch first
exec(open(os.path.join(os.environ["OUTPUT_DIR"], "disable_rope_validation.py")).read())

# Then import and run the evaluation script
sys.path.insert(0, os.getcwd())

from scripts.run_baseline_eval import main

# Run the evaluation with the provided arguments
if __name__ == "__main__":
    main(sys.argv[1:])
EOL

chmod +x ${EVAL_SCRIPT}

echo "Running GSM8K evaluation with disabled rope_scaling validation..."
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  --env OUTPUT_DIR="${OUTPUT_DIR}" \
  ${APPTAINER_ENV} \
  python ${EVAL_SCRIPT} \
  --config ${TMP_CONFIG} \
  --model_path ${MODEL_ID} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --datasets gsm8k

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 