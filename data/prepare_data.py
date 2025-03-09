#!/usr/bin/env python
"""
Script to prepare and split the FineMath dataset.
This script downloads and processes mathematical datasets for training and evaluation.
"""

import os
import argparse
import logging
import yaml
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets

from data_utils import MathDatasetProcessor

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare math datasets for RL training")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="../configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--split_ratios", 
        type=str, 
        default="0.8,0.1,0.1",
        help="Comma-separated list of train, validation, test split ratios (must sum to 1)"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="gsm8k,hendrycks_math",
        help="Comma-separated list of datasets to process"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples per dataset (None for all)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--process_for", 
        type=str, 
        default="all",
        choices=["all", "ppo", "grpo", "baseline"],
        help="What to process the data for (RL method or all)"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_gsm8k(
    processor: MathDatasetProcessor,
    max_samples: Optional[int] = None,
    split: str = 'train'
) -> Dataset:
    """Prepare the GSM8K dataset."""
    dataset = processor.load_dataset(
        dataset_name="gsm8k",
        subset="main",
        split=split,
        max_samples=max_samples
    )
    
    # Check the keys in the dataset
    logger.info(f"GSM8K dataset keys: {dataset[0].keys()}")
    
    # GSM8K uses "question" and "answer"
    return dataset

def prepare_math_dataset(
    processor: MathDatasetProcessor,
    max_samples: Optional[int] = None,
    split: str = 'train',
    subjects: Optional[List[str]] = None
) -> Dataset:
    """Prepare the MATH dataset."""
    if subjects:
        datasets = []
        for subject in subjects:
            subject_dataset = processor.load_dataset(
                dataset_name="hendrycks_math",
                subset=subject,
                split=split,
                max_samples=max_samples
            )
            datasets.append(subject_dataset)
        dataset = concatenate_datasets(datasets)
    else:
        # Load all subjects
        dataset = processor.load_dataset(
            dataset_name="hendrycks_math",
            subset=None,
            split=split,
            max_samples=max_samples
        )
    
    # Check the keys in the dataset
    logger.info(f"MATH dataset keys: {dataset[0].keys()}")
    
    # MATH uses "problem" and "solution"
    # Rename to match our standardized keys
    dataset = dataset.rename_column("problem", "question")
    dataset = dataset.rename_column("solution", "answer")
    
    return dataset

def prepare_custom_dataset(
    processor: MathDatasetProcessor,
    dataset_path: str,
    max_samples: Optional[int] = None
) -> Dataset:
    """Prepare a custom dataset from a local file."""
    dataset = processor.load_dataset(
        dataset_path=dataset_path,
        max_samples=max_samples
    )
    
    # Log the keys in the dataset
    logger.info(f"Custom dataset keys: {dataset[0].keys()}")
    
    # Ensure required keys are present
    required_keys = ["question", "answer"]
    for key in required_keys:
        if key not in dataset[0]:
            raise ValueError(f"Custom dataset missing required key: {key}")
    
    return dataset

def main():
    """Main function to prepare datasets."""
    args = parse_args()
    config = load_config(args.config)
    
    # Parse split ratios
    split_ratios = [float(r) for r in args.split_ratios.split(",")]
    assert len(split_ratios) == 3, "Must provide 3 split ratios (train, validation, test)"
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model_name_or_path", "huggyllama/finemath-llama-3b"),
        use_fast=config.get("use_fast_tokenizer", True)
    )
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset processor
    processor = MathDatasetProcessor(
        tokenizer=tokenizer,
        max_length=config.get("max_length", 2048),
        seed=args.seed
    )
    
    # Process each dataset
    dataset_names = args.datasets.split(",")
    all_datasets = []
    
    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        
        if dataset_name == "gsm8k":
            dataset = prepare_gsm8k(processor, args.max_samples)
        elif dataset_name == "hendrycks_math" or dataset_name == "math":
            dataset = prepare_math_dataset(processor, args.max_samples)
        elif os.path.exists(dataset_name):
            dataset = prepare_custom_dataset(processor, dataset_name, args.max_samples)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}, skipping.")
            continue
            
        all_datasets.append(dataset)
    
    # Combine datasets if multiple were specified
    if len(all_datasets) > 1:
        combined_dataset = concatenate_datasets(all_datasets)
        logger.info(f"Combined {len(all_datasets)} datasets with total {len(combined_dataset)} examples")
    else:
        combined_dataset = all_datasets[0]
        logger.info(f"Using single dataset with {len(combined_dataset)} examples")
    
    # Split dataset
    dataset_splits = processor.split_dataset(
        combined_dataset,
        train_ratio=split_ratios[0],
        val_ratio=split_ratios[1],
        test_ratio=split_ratios[2]
    )
    
    logger.info(f"Split dataset into:")
    logger.info(f"  Train: {len(dataset_splits['train'])} examples")
    logger.info(f"  Validation: {len(dataset_splits['validation'])} examples")
    logger.info(f"  Test: {len(dataset_splits['test'])} examples")
    
    # Process data for the specified RL method
    if args.process_for == "all" or args.process_for == "baseline":
        # Process for regular fine-tuning / evaluation
        processed_splits = {}
        for split_name, split_data in dataset_splits.items():
            processed_splits[split_name] = processor.prepare_dataset(split_data)
        
        # Save to baseline directory
        baseline_dir = os.path.join(args.output_dir, "baseline")
        processor.save_dataset_splits(processed_splits, baseline_dir)
        logger.info(f"Saved baseline processed data to {baseline_dir}")
    
    if args.process_for == "all" or args.process_for == "ppo":
        # Process for PPO
        ppo_splits = {}
        for split_name, split_data in dataset_splits.items():
            ppo_splits[split_name] = processor.prepare_for_ppo(split_data)
        
        # Save to PPO directory
        ppo_dir = os.path.join(args.output_dir, "ppo")
        processor.save_dataset_splits(ppo_splits, ppo_dir)
        logger.info(f"Saved PPO processed data to {ppo_dir}")
    
    if args.process_for == "all" or args.process_for == "grpo":
        # Process for GRPO
        grpo_splits = {}
        for split_name, split_data in dataset_splits.items():
            grpo_splits[split_name] = processor.prepare_for_grpo(split_data)
        
        # Save to GRPO directory
        grpo_dir = os.path.join(args.output_dir, "grpo")
        processor.save_dataset_splits(grpo_splits, grpo_dir)
        logger.info(f"Saved GRPO processed data to {grpo_dir}")
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main() 