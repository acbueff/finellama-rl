#!/usr/bin/env python
"""
Script to patch the transformers library's LlamaConfig class to accept any rope_scaling format.
This is a more aggressive approach to fix the rope_scaling validation issue.
"""

import os
import argparse
import logging
import importlib.util
import types
import sys

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Patch LlamaConfig to fix rope_scaling validation")
    parser.add_argument(
        "--run_test",
        action="store_true",
        help="Run a test after patching to verify it works"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="HuggingFaceTB/FineMath-Llama-3B",
        help="Model ID to test if run_test is True"
    )
    return parser.parse_args()

def patch_llama_config():
    """
    Patch the LlamaConfig._rope_scaling_validation method to be more permissive.
    """
    # First, find the transformers library path
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        logger.info(f"Found LlamaConfig at {LlamaConfig.__module__}")
        
        # Get the module
        module_spec = importlib.util.find_spec("transformers.models.llama.configuration_llama")
        if not module_spec:
            logger.error("Could not find the module specification for LlamaConfig")
            return False
        
        module_path = module_spec.origin
        logger.info(f"Module path: {module_path}")
        
        # Define our patched validation method
        def patched_rope_scaling_validation(self):
            """
            Patched version that converts any rope_scaling to the expected format.
            """
            if self.rope_scaling is None:
                return
            
            if not isinstance(self.rope_scaling, dict):
                logger.warning(f"rope_scaling is not a dictionary: {self.rope_scaling}")
                self.rope_scaling = {"type": "linear", "factor": 1.0}
                return
            
            # If it's already a dictionary, make sure it has the required fields
            if "type" not in self.rope_scaling:
                # Default to linear scaling
                self.rope_scaling["type"] = "linear"
                logger.warning(f"Added missing 'type' to rope_scaling: {self.rope_scaling}")
            
            if "factor" not in self.rope_scaling:
                # Get factor if it exists under a different name
                if "scale" in self.rope_scaling:
                    self.rope_scaling["factor"] = self.rope_scaling["scale"]
                elif "scaling_factor" in self.rope_scaling:
                    self.rope_scaling["factor"] = self.rope_scaling["scaling_factor"]
                else:
                    # Default to 1.0
                    self.rope_scaling["factor"] = 1.0
                logger.warning(f"Added missing 'factor' to rope_scaling: {self.rope_scaling}")
            
            # If it has extra fields, we'll just leave them (being permissive)
            logger.info(f"Fixed rope_scaling: {self.rope_scaling}")
        
        # Replace the original method with our patched version
        LlamaConfig._rope_scaling_validation = patched_rope_scaling_validation
        
        logger.info("Successfully patched LlamaConfig._rope_scaling_validation")
        return True
    
    except ImportError:
        logger.error("Could not import LlamaConfig from transformers")
        return False
    except Exception as e:
        logger.error(f"Error patching LlamaConfig: {str(e)}")
        return False

def test_config_loading(model_id):
    """Test loading the config with our patch."""
    try:
        from transformers import AutoConfig
        
        logger.info(f"Testing config loading for {model_id}")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        if hasattr(config, "rope_scaling"):
            logger.info(f"Successfully loaded config with rope_scaling: {config.rope_scaling}")
            return True
        else:
            logger.warning(f"Loaded config but it doesn't have rope_scaling attribute")
            return False
    
    except Exception as e:
        logger.error(f"Error testing config loading: {str(e)}")
        return False

def create_wrapper_script():
    """Create a wrapper script that applies the patch before running any code."""
    wrapper_script = """#!/usr/bin/env python
# Auto-generated wrapper script to patch LlamaConfig before running any code

import sys
import os
import importlib.util

# First, make sure our patch script is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import and apply our patch
import patch_llama_config
patch_llama_config.patch_llama_config()

# Now run the original script
if len(sys.argv) < 2:
    print("Usage: wrapper.py <script_to_run> [args...]")
    sys.exit(1)

target_script = sys.argv[1]
target_args = sys.argv[2:]

# Load the target script as a module
spec = importlib.util.spec_from_file_location("target_module", target_script)
if spec is None:
    print(f"Could not load script: {target_script}")
    sys.exit(1)

module = importlib.util.module_from_spec(spec)
sys.argv = [target_script] + target_args
spec.loader.exec_module(module)
"""
    
    # Write the wrapper script
    wrapper_path = "llama_config_wrapper.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_script)
    
    # Make it executable
    os.chmod(wrapper_path, 0o755)
    
    logger.info(f"Created wrapper script at {wrapper_path}")
    logger.info("To use: python llama_config_wrapper.py <your_script.py> [args...]")
    
    return wrapper_path

def main():
    args = parse_args()
    
    # Patch the LlamaConfig class
    success = patch_llama_config()
    
    if success:
        # Create the wrapper script
        wrapper_path = create_wrapper_script()
        
        # Create a sample slurm script that uses the wrapper
        create_slurm_script(wrapper_path)
        
        if args.run_test:
            # Test loading the config
            test_success = test_config_loading(args.model_id)
            
            if test_success:
                logger.info("Config loading test passed! The patch is working.")
                logger.info("You can now run your evaluation using the wrapper script.")
            else:
                logger.error("Config loading test failed. The patch may not be working.")
        else:
            logger.info("Patch applied but not tested. Use --run_test to verify it works.")
    else:
        logger.error("Failed to patch LlamaConfig.")

def create_slurm_script(wrapper_path):
    """Create a sample slurm script that uses the wrapper."""
    slurm_script = f"""#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=gsm8k_eval_patched
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
TMP_CONFIG="${{OUTPUT_DIR}}/tmp_gsm8k_eval_config.yaml"

echo "Starting baseline evaluation on GSM8K dataset with LlamaConfig patch..."
echo "Using model: $MODEL_ID"
echo "Results will be saved to: $OUTPUT_DIR"

# Step 1: Create a custom config for GSM8K only
apptainer exec --nv ${{APPTAINER_ENV}} python3 -c "
import yaml

# Load the original config
with open('${{CONFIG_PATH}}', 'r') as f:
    config = yaml.safe_load(f)

# Modify to only use GSM8K dataset
gsm8k_only = {{'gsm8k': {{
    'path': '${{GSM8K_DATA_PATH}}/gsm8k_test.json',
    'split': None,
    'subset': None,
    'max_samples': None
}}}}

config['eval_datasets'] = gsm8k_only

# Save the modified config
with open('${{TMP_CONFIG}}', 'w') as f:
    yaml.dump(config, f)

print(f'Created custom GSM8K evaluation config at ${{TMP_CONFIG}}')
"

# Step 2: Run the evaluation using our wrapper script that patches LlamaConfig
echo "Running GSM8K evaluation with LlamaConfig patch..."
apptainer exec --nv \\
  --env PYTHONPATH="${{PROJECT_ROOT}}:$PYTHONPATH" \\
  ${{APPTAINER_ENV}} \\
  python {wrapper_path} ${{PROJECT_ROOT}}/scripts/run_baseline_eval.py \\
  --config ${{TMP_CONFIG}} \\
  --model_path ${{MODEL_ID}} \\
  --output_dir ${{OUTPUT_DIR}} \\
  --batch_size ${{BATCH_SIZE}} \\
  --datasets gsm8k

echo "Evaluation complete. Results saved to ${{OUTPUT_DIR}}"
echo ""
echo "To view the results, check the following files:"
echo "  - ${{OUTPUT_DIR}}/gsm8k_metrics.json (detailed metrics)"
echo "  - ${{OUTPUT_DIR}}/evaluation_summary.json (summary of results)"
echo "  - ${{OUTPUT_DIR}}/gsm8k_predictions.jsonl (model predictions)"
"""
    
    # Write the slurm script
    slurm_script_path = "slurm_gsm8k_eval_patched.sh"
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script)
    
    # Make it executable
    os.chmod(slurm_script_path, 0o755)
    
    logger.info(f"Created sample slurm script at {slurm_script_path}")
    logger.info("To use: sbatch slurm_gsm8k_eval_patched.sh")

if __name__ == "__main__":
    main() 