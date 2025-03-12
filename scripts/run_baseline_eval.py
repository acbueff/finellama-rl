#!/usr/bin/env python
"""
Script to evaluate the baseline FineMath-Llama-3B model.
"""

import os
import sys
import argparse
import logging
import yaml
import json
from typing import Dict, Any

import torch
from transformers import set_seed

from src.models.finemathllama import FineMathLlamaModel
from src.eval.evaluator import MathEvaluator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate baseline FineMath-Llama-3B model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="huggyllama/finemath-llama-3b",
        help="Path to baseline model (or Hugging Face model ID)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/baseline",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        default=None,
        help="Comma-separated list of datasets to evaluate on (None for all configured datasets)"
    )
    parser.add_argument(
        "--use_gguf",
        type=bool,
        default=False,
        help="Whether to use GGUF model loading (via ctransformers)"
    )
    
    return parser.parse_args(args)

def main(cmd_line_args=None):
    """Run baseline model evaluation."""
    args = parse_args(cmd_line_args)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation configuration
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    # Override seed in configuration
    eval_config["seed"] = args.seed
    
    # Add GGUF configuration if specified
    if args.use_gguf:
        eval_config["use_gguf"] = True
    
    # Create evaluator
    evaluator = MathEvaluator(config=eval_config)
    
    # Load model for evaluation
    logger.info(f"Loading baseline model from {args.model_path}")
    evaluator.load_model("baseline", args.model_path, model_type="baseline", use_gguf=args.use_gguf)
    
    # Load evaluation datasets
    evaluator.load_evaluation_datasets()
    
    # Save configuration (with any modifications)
    with open(os.path.join(args.output_dir, "eval_config.yaml"), 'w') as f:
        yaml.dump(eval_config, f)
    
    # Determine which datasets to evaluate on
    if args.datasets:
        dataset_names = args.datasets.split(",")
        # Filter to only include datasets that are loaded
        dataset_names = [name for name in dataset_names if name in evaluator.eval_datasets]
    else:
        dataset_names = list(evaluator.eval_datasets.keys())
    
    logger.info(f"Evaluating on datasets: {', '.join(dataset_names)}")
    
    # Evaluate on each dataset
    for dataset_name in dataset_names:
        logger.info(f"Evaluating baseline model on {dataset_name}")
        
        metrics = evaluator.evaluate_model(
            model_name="baseline",
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            max_new_tokens=eval_config.get("generation", {}).get("max_new_tokens", 512),
            temperature=eval_config.get("generation", {}).get("temperature", 0.7),
            do_sample=eval_config.get("generation", {}).get("do_sample", False)
        )
        
        # Log important metrics
        logger.info(f"Results for {dataset_name}:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name}: {metric_value}")
                
    # Save a summary of all results
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    
    # Collect all results
    summary = {}
    for dataset_name in dataset_names:
        metrics_file = os.path.join(args.output_dir, f"{dataset_name}_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            summary[dataset_name] = metrics
    
    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    # Check if we're running from a monkey patched environment
    if 'cmd_line_args' in globals():
        main(globals()['cmd_line_args'])
    else:
        main() 