#!/usr/bin/env python
"""
Simple script to test if we can load the Qwen2.5-7B-DPO-VP model
and run a basic inference.
"""

import os
import sys
import torch
import logging
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test model loading and inference")
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="SunnyLin/Qwen2.5-7B-DPO-VP",
        help="Path to model or Hugging Face model ID"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Whether to use 4-bit quantization"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use the model cache"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set cache environment variables if not using cache
    if not args.use_cache:
        logger.info("Disabling model cache")
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
        os.environ["HF_HOME"] = "/tmp/hf_home"
    
    try:
        # Step 1: Load config
        logger.info(f"Loading config for model: {args.model_id}")
        
        config = AutoConfig.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        
        logger.info(f"Config loaded successfully")
        
        # Step 2: Load tokenizer
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Step 3: Load model
        logger.info("Loading model")
        
        try:
            if args.use_4bit:
                # Configure 4-bit quantization
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    config=config,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            logger.info("Model loaded successfully")
        
            # Step 4: Run a simple inference test
            logger.info("Testing inference")
            
            test_input = "What is 2+2?"
            inputs = tokenizer(test_input, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            
            # Generate
            logger.info(f"Generating response for: '{test_input}'")
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode and print the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated text: {generated_text}")
            
            logger.info("Test completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model loading or inference: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error during config loading: {str(e)}")
        raise

if __name__ == "__main__":
    main() 