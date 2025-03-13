#!/usr/bin/env python
"""
Debug script to test the rope_scaling issue in more detail.
This script will try to load the model and report detailed information.
"""

import os
import sys
import logging
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

# Configure very verbose logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Import our patch function if available
try:
    sys.path.insert(0, os.getcwd())
    from patch_llama_config import patch_llama_config
    # Apply the patch
    patch_success = patch_llama_config()
    logger.info(f"Patch applied: {patch_success}")
except ImportError:
    logger.warning("patch_llama_config not found, running without patch")
    patch_success = False

def main():
    # Set HuggingFace environment variables
    os.environ["HF_DATASETS_CACHE"] = "/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
    os.environ["HF_HOME"] = "/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
    
    MODEL_ID = "HuggingFaceTB/FineMath-Llama-3B"
    OUTPUT_DIR = "/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k"
    
    logger.info(f"Testing model loading for {MODEL_ID}")
    
    try:
        # Step 1: Load config
        logger.info("Step 1: Loading config...")
        config = AutoConfig.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        
        if hasattr(config, "rope_scaling"):
            logger.info(f"rope_scaling in config: {config.rope_scaling}")
        else:
            logger.warning("No rope_scaling found in config")
        
        # Step 2: Load tokenizer
        logger.info("Step 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=False
        )
        logger.info("Tokenizer loaded successfully")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Step 3: Try to load the model
        logger.info("Step 3: Loading model...")
        try:
            # Use 4-bit quantization to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                config=config,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully")
            
            # Step 4: Check if GSM8K test data exists
            gsm8k_path = "/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
            if os.path.exists(gsm8k_path):
                logger.info(f"GSM8K test data exists at {gsm8k_path}")
                
                # Check its structure
                import json
                with open(gsm8k_path, 'r') as f:
                    data = json.load(f)
                
                logger.info(f"GSM8K test data type: {type(data)}")
                if isinstance(data, list):
                    logger.info(f"Number of examples: {len(data)}")
                    if len(data) > 0:
                        logger.info(f"First example keys: {data[0].keys()}")
                elif isinstance(data, dict):
                    logger.info(f"Keys in data: {data.keys()}")
            else:
                logger.error(f"GSM8K test data not found at {gsm8k_path}")
            
            # Step 5: Check if script run_baseline_eval.py exists
            script_path = Path("/home/x_anbue/finellama-rl/scripts/run_baseline_eval.py")
            if script_path.exists():
                logger.info(f"Script exists at {script_path}")
                
                # Check its permissions
                import stat
                st = os.stat(script_path)
                logger.info(f"Script permissions: {stat.filemode(st.st_mode)}")
                
                # Try to execute it with help flag
                import subprocess
                try:
                    result = subprocess.run(
                        ["python", script_path, "--help"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    logger.info(f"Script help output: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"Script help stderr: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error running script: {str(e)}")
            else:
                logger.error(f"Script not found at {script_path}")
            
            # Try to generate a simple example
            logger.info("Testing generation with the model...")
            input_text = "Solve the following math problem step by step: What is 2+2?"
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated text: {output_text}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main() 