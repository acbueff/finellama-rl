#!/usr/bin/env python
"""
Script to evaluate the PPO-optimized FineMath-Llama-3B model.
"""

import os
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PPO-optimized FineMath-Llama-3B model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to PPO-optimized model checkpoint"
    )
    parser.add_argument(
        "--baseline_results", 
        type=str, 
        default=None,
        help="Path to baseline model results for comparison"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/ppo",
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
    
    return parser.parse_args()

def main():
    """Run PPO model evaluation."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation configuration
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    # Override seed in configuration
    eval_config["seed"] = args.seed
    
    # Create evaluator
    evaluator = MathEvaluator(config=eval_config)
    
    # Load model for evaluation
    logger.info(f"Loading PPO-optimized model from {args.model_path}")
    evaluator.load_model("ppo", args.model_path, model_type="ppo")
    
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
        logger.info(f"Evaluating PPO model on {dataset_name}")
        
        metrics = evaluator.evaluate_model(
            model_name="ppo",
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
    
    # Compare with baseline if baseline results provided
    if args.baseline_results and os.path.exists(args.baseline_results):
        try:
            # Load baseline model
            baseline_model_path = None
            baseline_config_path = os.path.join(args.baseline_results, "eval_config.yaml")
            if os.path.exists(baseline_config_path):
                with open(baseline_config_path, 'r') as f:
                    baseline_config = yaml.safe_load(f)
                    baseline_model_path = baseline_config.get("model_path", "huggyllama/finemath-llama-3b")
            
            if baseline_model_path:
                logger.info(f"Loading baseline model from {baseline_model_path} for comparison")
                evaluator.load_model("baseline", baseline_model_path, model_type="baseline")
                
                # Compare models on each dataset
                for dataset_name in dataset_names:
                    logger.info(f"Comparing models on {dataset_name}")
                    comparison = evaluator.compare_models(
                        model_names=["baseline", "ppo"],
                        dataset_name=dataset_name,
                        batch_size=args.batch_size
                    )
                    
                    # Generate qualitative examples
                    logger.info(f"Generating qualitative examples for {dataset_name}")
                    evaluator.generate_qualitative_examples(
                        model_names=["baseline", "ppo"],
                        dataset_name=dataset_name,
                        num_examples=5
                    )
        except Exception as e:
            logger.error(f"Error in baseline comparison: {e}")
            
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
    main() 