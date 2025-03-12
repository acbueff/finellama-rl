#!/bin/bash
#SBATCH -A berzelius-2024-329
#SBATCH --gpus=1
#SBATCH -t 0-12:00:00
#SBATCH -J gsm8k_eval
#SBATCH -o /home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.out
#SBATCH -e /home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.err

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k
mkdir -p /home/x_anbue/finellama-rl/logs

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Paths and configurations
MODEL_ID="HuggingFaceTB/FineMath-Llama-3B"  # Using the original FineMath model
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

# Step 2: Install required packages and create a temporary fix for the rope_scaling issue
echo "Ensuring all required packages are installed..."
apptainer exec --nv ${APPTAINER_ENV} pip install peft==0.4.0 --user

# Create a temporary monkey patch for the rope_scaling validation
echo "Creating workaround for rope_scaling issue..."
MONKEY_PATCH_FILE="${OUTPUT_DIR}/monkey_patch.py"
cat > ${MONKEY_PATCH_FILE} << 'EOL'
# Monkey patch to handle the rope_scaling configuration issue
from transformers.models.llama.configuration_llama import LlamaConfig

# Keep reference to the original validation method
original_validation = LlamaConfig._rope_scaling_validation

# Create a no-op validation function
def no_validation(self):
    pass

# Apply the monkey patch
LlamaConfig._rope_scaling_validation = no_validation
print("Applied monkey patch for rope_scaling validation")
EOL

# Step 3: Run the evaluation with the monkey patch
echo "Running GSM8K evaluation..."
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  ${APPTAINER_ENV} \
  python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
# First load the monkey patch
exec(open('${MONKEY_PATCH_FILE}').read())
# Now import and run the evaluation script
import runpy
runpy.run_path('${PROJECT_ROOT}/scripts/run_baseline_eval.py', run_name='__main__', 
  init_globals={
    'cmd_line_args': [
      '--config', '${TMP_CONFIG}',
      '--model_path', '${MODEL_ID}',
      '--output_dir', '${OUTPUT_DIR}',
      '--batch_size', '${BATCH_SIZE}',
      '--datasets', 'gsm8k'
    ]
  }
)
"

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 