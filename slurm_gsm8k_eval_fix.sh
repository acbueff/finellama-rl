#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=gsm8k_eval
#SBATCH --output=/home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.out
#SBATCH --error=/home/x_anbue/finellama-rl/logs/gsm8k_eval_%j.err

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

# Step 2: Create a temporary patched version of the evaluator.py file
echo "Creating a patched evaluator to fix the rope_scaling issue..."

# Create a Python script to patch the evaluator
PATCH_SCRIPT="${OUTPUT_DIR}/patch_evaluator.py"
cat > ${PATCH_SCRIPT} << 'EOL'
import re
import os
import sys

# Path to the original evaluator file
evaluator_path = sys.argv[1]
patched_path = sys.argv[2]

# Read the original file
with open(evaluator_path, 'r') as f:
    content = f.read()

# Define the patch to insert
patch = '''
                        # First get the config
                        from transformers import AutoConfig
                        config = AutoConfig.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                        )
                        
                        # Ensure rope_scaling is properly set
                        if hasattr(config, "rope_scaling") and config.rope_scaling is None:
                            config.rope_scaling = {"type": "linear", "factor": 1.0}
                        elif hasattr(config, "rope_scaling") and "type" not in config.rope_scaling:
                            config.rope_scaling["type"] = "linear"
                            if "factor" not in config.rope_scaling:
                                config.rope_scaling["factor"] = 1.0
                        
                        print(f"Config rope_scaling: {config.rope_scaling if hasattr(config, 'rope_scaling') else None}")
                        
                        '''

# Find the point to insert our patch
pattern = r'(if use_4bit:.*?quantization_config = BitsAndBytesConfig.*?model = AutoModelForCausalLM\.from_pretrained\()'
matches = list(re.finditer(pattern, content, re.DOTALL))

if not matches:
    print("Could not find insertion point in the file.")
    sys.exit(1)

# Get the last match (most appropriate for insertion)
match = matches[-1]
start_idx = match.start(1)
end_idx = match.end(1)

# Replace the matched text with our patched version
patched_content = content[:start_idx] + patch + content[start_idx:]

# Save the patched file
with open(patched_path, 'w') as f:
    f.write(patched_content)

print(f"Successfully patched evaluator at {patched_path}")
EOL

# Copy the original evaluator file
ORIGINAL_EVALUATOR="${PROJECT_ROOT}/src/eval/evaluator.py"
PATCHED_EVALUATOR="${OUTPUT_DIR}/evaluator_patched.py"

# Run the patch script
apptainer exec --nv ${APPTAINER_ENV} python ${PATCH_SCRIPT} ${ORIGINAL_EVALUATOR} ${PATCHED_EVALUATOR}

# Create a backup of the original file
cp ${ORIGINAL_EVALUATOR} ${ORIGINAL_EVALUATOR}.bak

# Replace the original file with our patched version
cp ${PATCHED_EVALUATOR} ${ORIGINAL_EVALUATOR}

# Step 3: Run the evaluation with the patched evaluator
echo "Running GSM8K evaluation with patched evaluator..."
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  ${APPTAINER_ENV} \
  python ${PROJECT_ROOT}/scripts/run_baseline_eval.py \
  --config ${TMP_CONFIG} \
  --model_path ${MODEL_ID} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size ${BATCH_SIZE} \
  --datasets gsm8k

# Restore the original evaluator file
echo "Restoring original evaluator file..."
mv ${ORIGINAL_EVALUATOR}.bak ${ORIGINAL_EVALUATOR}

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}"
echo ""
echo "To view the results, check the following files:"
echo "  - ${OUTPUT_DIR}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${OUTPUT_DIR}/evaluation_summary.json (summary of results)"
echo "  - ${OUTPUT_DIR}/gsm8k_predictions.jsonl (model predictions)" 