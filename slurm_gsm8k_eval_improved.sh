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

# Step 2: Create a simple wrapper script to handle rope_scaling properly
WRAPPER_SCRIPT="${OUTPUT_DIR}/rope_scaling_wrapper.py"

cat > ${WRAPPER_SCRIPT} << 'EOL'
#!/usr/bin/env python
"""
Wrapper script that modifies the original run_baseline_eval.py script to handle rope_scaling properly.
"""
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

# Get original command-line arguments
orig_args = sys.argv[1:]

# Import the original script
sys.path.insert(0, os.getcwd())
from scripts.run_baseline_eval import parse_args, main

# Monkey-patch the MathEvaluator.load_model method to fix rope_scaling
from src.eval.evaluator import MathEvaluator

# Store the original method
original_load_model = MathEvaluator.load_model

def patched_load_model(self, model_name, model_path, model_type="baseline", use_gguf=False):
    """
    Patched version of load_model that handles rope_scaling properly.
    """
    print(f"Using patched load_model method with rope_scaling fix for {model_path}")
    
    try:
        if use_gguf:
            # Original GGUF loading logic
            return original_load_model(self, model_name, model_path, model_type, use_gguf)
        else:
            # Load tokenizer first
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                use_fast=False
            )
            
            # Ensure the tokenizer has pad_token set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load and fix config
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Fix rope_scaling
            if hasattr(config, "rope_scaling"):
                print(f"Original rope_scaling: {config.rope_scaling}")
                
                # Apply fixes
                if config.rope_scaling is None:
                    config.rope_scaling = {"type": "linear", "factor": 1.0}
                elif isinstance(config.rope_scaling, dict):
                    if "type" not in config.rope_scaling:
                        config.rope_scaling["type"] = "linear"
                    if "factor" not in config.rope_scaling:
                        config.rope_scaling["factor"] = 1.0
                
                print(f"Fixed rope_scaling: {config.rope_scaling}")
            
            # Determine if we should use 4-bit quantization
            use_4bit = self.config.get("use_4bit", True)
            
            # Load the model with our fixed config
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,  # Use our fixed config
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,  # Use our fixed config
                    torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True
                )
            
            # Store model and tokenizer
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            print(f"Successfully loaded model {model_name} with rope_scaling fix")
            return model, tokenizer
            
    except Exception as e:
        print(f"Error in patched load_model: {str(e)}")
        # Fall back to original method if patched version fails
        return original_load_model(self, model_name, model_path, model_type, use_gguf)

# Apply the monkey patch
MathEvaluator.load_model = patched_load_model

# Run the original main function with our patched class
if __name__ == "__main__":
    main(orig_args)
EOL

chmod +x ${WRAPPER_SCRIPT}

# Step 3: Run the evaluation with our wrapper script
echo "Running GSM8K evaluation with rope_scaling fix..."
apptainer exec --nv \
  --env PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
  ${APPTAINER_ENV} \
  python ${WRAPPER_SCRIPT} \
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