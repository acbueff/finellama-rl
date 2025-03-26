#!/usr/bin/env python
"""
Script to compare the results of different RL training methods (PPO, GRPO, baseline).
"""

import os
import argparse
import logging
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare RL methods results")
    
    parser.add_argument(
        "--baseline_results", 
        type=str, 
        default=None,
        help="Path to baseline evaluation results"
    )
    parser.add_argument(
        "--ppo_results", 
        type=str, 
        required=True,
        help="Path to PPO evaluation results"
    )
    parser.add_argument(
        "--grpo_results", 
        type=str, 
        required=True,
        help="Path to GRPO evaluation results"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/comparison",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gsm8k",
        help="Dataset name to focus analysis on"
    )
    
    return parser.parse_args()

def load_results(result_path: str) -> Optional[Dict[str, Any]]:
    """Load evaluation results from a JSON file."""
    if not os.path.exists(result_path):
        logger.warning(f"Results file not found: {result_path}")
        return None
        
    with open(result_path, 'r') as f:
        try:
            results = json.load(f)
            return results
        except json.JSONDecodeError:
            logger.error(f"Error parsing JSON from {result_path}")
            return None

def generate_comparison_tables(results: Dict[str, Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """Generate comparison tables for different metrics."""
    comparison = {}
    
    # Check if dataset exists in all results
    for method, method_results in results.items():
        if dataset not in method_results:
            logger.warning(f"Dataset {dataset} not found in {method} results")
            return {}
    
    # Extract metrics for the dataset
    metrics = {}
    for method, method_results in results.items():
        metrics[method] = method_results[dataset]
    
    # Create comparative metric tables
    comparison["accuracy"] = {
        method: metrics[method].get("final_answer_accuracy", 0.0)
        for method in metrics
    }
    
    comparison["step_accuracy"] = {
        method: metrics[method].get("step_by_step_accuracy", 0.0)
        for method in metrics
    }
    
    comparison["mathematical_validity"] = {
        method: metrics[method].get("mathematical_accuracy", 0.0)
        for method in metrics
    }
    
    comparison["rewards"] = {
        method: metrics[method].get("mean_reward", 0.0)
        for method in metrics
    }
    
    # Add percentage improvement relative to baseline (if available)
    if "baseline" in metrics:
        for metric_name, metric_values in comparison.items():
            baseline_value = metric_values.get("baseline", 0.0)
            if baseline_value > 0:
                comparison[f"{metric_name}_improvement"] = {
                    method: 100 * (metric_values[method] - baseline_value) / baseline_value
                    for method in metric_values if method != "baseline"
                }
    
    return comparison

def generate_comparison_plots(comparison: Dict[str, Dict[str, float]], output_dir: str, dataset: str):
    """Generate comparison plots and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('ggplot')
    
    # Iterate through metric groups
    for metric_name, metric_values in comparison.items():
        if "improvement" in metric_name:
            continue  # Skip improvement metrics for plotting
            
        # Create bar chart
        plt.figure(figsize=(10, 6))
        methods = list(metric_values.keys())
        values = [metric_values[method] for method in methods]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(methods)]
        
        plt.bar(methods, values, color=colors)
        plt.title(f"{metric_name.replace('_', ' ').title()} on {dataset.upper()}")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, value in enumerate(values):
            plt.text(i, value + 0.01, f"{value:.4f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_{metric_name}.png"), dpi=300)
        plt.close()
    
    # Create combined comparison chart
    plt.figure(figsize=(12, 8))
    metrics = [m for m in comparison.keys() if "improvement" not in m]
    methods = list(comparison[metrics[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        values = [comparison[metric][method] for metric in metrics]
        plt.bar(x + i * width - width * len(methods) / 2 + width / 2, 
                values, width, label=method)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Comparison of Methods on {dataset.upper()}')
    plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{dataset}_combined_comparison.png"), dpi=300)
    plt.close()

def generate_qualitative_comparison(results_dirs: Dict[str, str], dataset: str, output_dir: str):
    """Compare qualitative examples across methods."""
    examples = {}
    
    # Load qualitative examples for each method
    for method, result_dir in results_dirs.items():
        example_path = os.path.join(result_dir, f"{dataset}_examples.json")
        if os.path.exists(example_path):
            with open(example_path, 'r') as f:
                try:
                    method_examples = json.load(f)
                    examples[method] = method_examples
                except json.JSONDecodeError:
                    logger.warning(f"Error loading examples from {example_path}")
    
    # Create markdown comparison
    if examples:
        comparison_md = f"# Qualitative Comparison on {dataset.upper()}\n\n"
        
        # Find common examples across methods
        common_ids = set()
        for method, method_examples in examples.items():
            method_ids = {ex.get("id") for ex in method_examples}
            if not common_ids:
                common_ids = method_ids
            else:
                common_ids &= method_ids
        
        # Generate comparison for each common example
        for example_id in sorted(common_ids):
            comparison_md += f"## Example {example_id}\n\n"
            
            # Get the example from the first method to get the prompt
            first_method = list(examples.keys())[0]
            example = next(ex for ex in examples[first_method] if ex.get("id") == example_id)
            
            comparison_md += f"### Problem\n\n```\n{example.get('prompt', '')}\n```\n\n"
            
            if "reference" in example:
                comparison_md += f"### Reference Solution\n\n```\n{example.get('reference', '')}\n```\n\n"
            
            # Add responses for each method
            for method in examples:
                method_example = next(ex for ex in examples[method] if ex.get("id") == example_id)
                comparison_md += f"### {method.upper()} Solution\n\n```\n{method_example.get('sampled_response', '')}\n```\n\n"
            
            comparison_md += "---\n\n"
        
        # Save to file
        with open(os.path.join(output_dir, f"{dataset}_qualitative_comparison.md"), 'w') as f:
            f.write(comparison_md)

def main():
    """Run the comparison of methods."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results for each method
    results = {}
    
    if args.baseline_results:
        baseline_results = load_results(args.baseline_results)
        if baseline_results:
            results["baseline"] = baseline_results
    
    ppo_results = load_results(args.ppo_results)
    if ppo_results:
        results["ppo"] = ppo_results
    
    grpo_results = load_results(args.grpo_results)
    if grpo_results:
        results["grpo"] = grpo_results
    
    if not results:
        logger.error("No valid results found. Exiting.")
        return
    
    # Generate comparison tables
    logger.info(f"Generating comparison tables for {args.dataset}")
    comparison = generate_comparison_tables(results, args.dataset)
    
    # Save comparison tables
    with open(os.path.join(args.output_dir, f"{args.dataset}_comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Generate comparison plots
    logger.info("Generating comparison plots")
    generate_comparison_plots(comparison, args.output_dir, args.dataset)
    
    # Generate qualitative comparison
    logger.info("Generating qualitative comparison")
    result_dirs = {
        "baseline": os.path.dirname(args.baseline_results) if args.baseline_results else None,
        "ppo": os.path.dirname(args.ppo_results),
        "grpo": os.path.dirname(args.grpo_results)
    }
    result_dirs = {k: v for k, v in result_dirs.items() if v is not None}
    generate_qualitative_comparison(result_dirs, args.dataset, args.output_dir)
    
    logger.info(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 