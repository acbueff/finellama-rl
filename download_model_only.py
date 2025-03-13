#!/usr/bin/env python
"""
Script to download model files to cache without loading the full model.
This is a lightweight alternative for login nodes that may have memory limitations.
"""

import os
import argparse
import logging
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Download model files to cache")
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables for cache
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    
    logger.info(f"Downloading model files for {args.model_id}")
    logger.info(f"Using cache directory: {args.cache_dir}")
    
    # Step 1: Download model using snapshot_download
    try:
        logger.info("Step 1: Downloading model snapshot...")
        local_path = snapshot_download(
            repo_id=args.model_id,
            cache_dir=args.cache_dir,
            local_dir_use_symlinks=False  # Actual files instead of symlinks
        )
        logger.info(f"Model files downloaded to: {local_path}")
    except Exception as e:
        logger.error(f"Failed to download model snapshot: {str(e)}")
        return
    
    # Step 2: Download tokenizer explicitly (sometimes missed by snapshot_download)
    try:
        logger.info("Step 2: Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, 
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            use_fast=False
        )
        logger.info("Tokenizer downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download tokenizer: {str(e)}")
    
    # Step 3: Download config explicitly
    try:
        logger.info("Step 3: Downloading config...")
        config = AutoConfig.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        
        # Check and log rope_scaling configuration
        if hasattr(config, "rope_scaling"):
            logger.info(f"Model has rope_scaling configuration: {config.rope_scaling}")
            
            # Print warning if rope_scaling is problematic
            if config.rope_scaling is None:
                logger.warning("rope_scaling is None, which may cause issues when loading the model")
            elif isinstance(config.rope_scaling, dict) and ("type" not in config.rope_scaling or "factor" not in config.rope_scaling):
                logger.warning(f"rope_scaling is missing required fields: {config.rope_scaling}")
        else:
            logger.info("Model does not have rope_scaling configuration")
        
        logger.info("Config downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download config: {str(e)}")
    
    logger.info("Download completed successfully.")
    logger.info(f"Model files are cached at: {args.cache_dir}")
    logger.info("You can now run the slurm job to evaluate the model.")

if __name__ == "__main__":
    main() 