#!/usr/bin/env python
"""
Script to test downloading and loading the FineMath-Llama-3B model to diagnose rope_scaling issues.
This can be run on the login node for debugging purposes.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test model download and loading")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="HuggingFaceTB/FineMath-Llama-3B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache",
        help="Directory to use for model cache"
    )
    parser.add_argument(
        "--test_generation",
        action="store_true",
        help="Whether to test generation with a simple math problem"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables for cache
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    
    logger.info(f"Testing model loading for {args.model_id}")
    logger.info(f"Using cache directory: {args.cache_dir}")
    
    # First attempt: Load just the tokenizer to see if we can access the model
    try:
        logger.info("Step 1: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            use_fast=False
        )
        logger.info("✓ Successfully loaded tokenizer")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        return
    
    # Second attempt: Try to load the config and check rope_scaling
    try:
        logger.info("Step 2: Loading model config...")
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        
        # Check and fix rope_scaling
        if hasattr(config, "rope_scaling"):
            logger.info(f"Original rope_scaling: {config.rope_scaling}")
            
            # Apply our fix
            if config.rope_scaling is None:
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                logger.info("Fixed: Set rope_scaling that was None")
            elif isinstance(config.rope_scaling, dict):
                if "type" not in config.rope_scaling:
                    config.rope_scaling["type"] = "linear"
                    logger.info("Fixed: Added missing 'type' to rope_scaling")
                if "factor" not in config.rope_scaling:
                    config.rope_scaling["factor"] = 1.0
                    logger.info("Fixed: Added missing 'factor' to rope_scaling")
            
            logger.info(f"Updated rope_scaling: {config.rope_scaling}")
        else:
            logger.info("No rope_scaling attribute found in config")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return
    
    # Third attempt: Try to load the model with our fixed config
    try:
        logger.info("Step 3: Loading model with fixed config...")
        
        # Use 4-bit quantization to reduce memory usage on login node
        use_4bit = True
        
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                config=config,  # Use our fixed config
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                config=config,  # Use our fixed config
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
        
        logger.info("✓ Successfully loaded model")
        
        # Test generation if requested
        if args.test_generation:
            logger.info("Step 4: Testing generation...")
            
            # Simple math problem
            problem = "What is the sum of 123 and 456?"
            prompt = f"Solve the following math problem step by step:\n{problem}\n\nSolution:"
            
            logger.info(f"Generating response for: {problem}")
            
            # Tokenize input
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            # Decode output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated response:\n{generated_text}")
    
    except Exception as e:
        logger.error(f"Failed to load or run model: {str(e)}")
        return
    
    logger.info("Test completed successfully.")

if __name__ == "__main__":
    main() 