#!/usr/bin/env python
"""
Script to download just the GSM8K math dataset and save it to a designated location.
This script is designed to run on a CPU without requiring Slurm.
"""

import os
import argparse
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download GSM8K math dataset")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--splits", 
        type=str, 
        default="train,test",
        help="Comma-separated list of splits to download (e.g., 'train,test')"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory (default: ~/.cache/huggingface)"
    )
    
    return parser.parse_args()

def main():
    """Main function to download GSM8K dataset."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set HuggingFace cache directory if provided
    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir
        logger.info(f"Set HuggingFace cache directory to: {args.cache_dir}")
    
    # Parse splits
    splits = args.splits.split(",")
    
    # Download each split
    for split in splits:
        logger.info(f"Downloading GSM8K {split} split...")
        
        dataset = load_dataset("openai/gsm8k", "main", split=split, cache_dir=args.cache_dir)
        
        logger.info(f"Downloaded {len(dataset)} examples")
        
        # Save the dataset to disk
        output_path = os.path.join(args.output_dir, f"gsm8k_{split}.json")
        dataset.to_json(output_path)
        
        logger.info(f"Saved {split} split to {output_path}")

if __name__ == "__main__":
    main() 