#!/usr/bin/env python
"""
Script to compare the results of baseline, PPO, and GRPO methods.
This script loads evaluation results from multiple models and generates comparison charts and tables.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import math
from typing import Dict, List, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare baseline, PPO, and GRPO results")
    
    parser.add_argument(
        "--baseline_results", 
        type=str, 
        required=True,
        help="Path to baseline evaluation results directory"
    )
    parser.add_argument(
        "--ppo_results", 
        type=str, 
        required=True,
        help="Path to PPO evaluation results directory"
    )
    parser.add_argument(
        "--grpo_results", 
        type=str, 
        required=True,
        help="Path to GRPO evaluation results directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/comparison",
        help="Directory to save comparison results"
    )
    parser.add_argument(
        "--datasets", 
        type=str, 
        default=None,
        help="Comma-separated list of datasets to compare (None for all available)"
    )
    parser.add_argument(
        "--metrics", 
        type=str, 
        default="exact_match,f1_score,mathematical_accuracy,step_by_step_accuracy",
        help="Comma-separated list of metrics to compare"
    )
    
    return parser.parse_args()

def load_results(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation results from a directory.
    
    Args:
        results_dir: Directory containing evaluation results
        
    Returns:
        Dictionary mapping dataset names to metrics
    """
    results = {}
    
    # Try to load summary file first
    summary_path = os.path.join(results_dir, "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results = json.load(f)
    else:
        # Look for individual dataset metrics files
        for file in os.listdir(results_dir):
            if file.endswith("_metrics.json"):
                dataset_name = file.replace("_metrics.json", "")
                file_path = os.path.join(results_dir, file)
                
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    
                results[dataset_name] = metrics
    
    return results

def extract_common_datasets_and_metrics(
    baseline_results: Dict[str, Dict[str, Any]],
    ppo_results: Dict[str, Dict[str, Any]],
    grpo_results: Dict[str, Dict[str, Any]],
    dataset_filter: List[str] = None,
    metrics_filter: List[str] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract results for datasets and metrics common to all methods.
    
    Args:
        baseline_results: Baseline evaluation results
        ppo_results: PPO evaluation results
        grpo_results: GRPO evaluation results
        dataset_filter: List of datasets to include (None for all common datasets)
        metrics_filter: List of metrics to include (None for all common metrics)
        
    Returns:
        Dictionary mapping dataset names to metrics to method results
    """
    # Find common datasets
    baseline_datasets = set(baseline_results.keys())
    ppo_datasets = set(ppo_results.keys())
    grpo_datasets = set(grpo_results.keys())
    
    common_datasets = baseline_datasets.intersection(ppo_datasets).intersection(grpo_datasets)
    
    if dataset_filter:
        common_datasets = common_datasets.intersection(set(dataset_filter))
    
    # Extract results
    comparison = {}
    
    for dataset in common_datasets:
        # Skip dataset if it doesn't have metrics
        if not all(dataset in results for results in [baseline_results, ppo_results, grpo_results]):
            continue
            
        # Get metrics for this dataset
        baseline_metrics = baseline_results[dataset]
        ppo_metrics = ppo_results[dataset]
        grpo_metrics = grpo_results[dataset]
        
        # Find common metrics
        baseline_metric_keys = set(baseline_metrics.keys())
        ppo_metric_keys = set(ppo_metrics.keys())
        grpo_metric_keys = set(grpo_metrics.keys())
        
        common_metrics = baseline_metric_keys.intersection(ppo_metric_keys).intersection(grpo_metric_keys)
        
        if metrics_filter:
            common_metrics = common_metrics.intersection(set(metrics_filter))
        
        # Only keep metrics that are numeric
        common_metrics = [m for m in common_metrics if isinstance(baseline_metrics.get(m), (int, float))]
        
        # Create comparison for this dataset
        dataset_comparison = {}
        
        for metric in common_metrics:
            dataset_comparison[metric] = {
                "baseline": baseline_metrics.get(metric, 0.0),
                "ppo": ppo_metrics.get(metric, 0.0),
                "grpo": grpo_metrics.get(metric, 0.0)
            }
            
        comparison[dataset] = dataset_comparison
    
    return comparison

def create_comparison_table(
    comparison: Dict[str, Dict[str, Dict[str, float]]],
    output_file: str
) -> pd.DataFrame:
    """
    Create a comparison table and save it to a CSV file.
    
    Args:
        comparison: Comparison data
        output_file: Path to output CSV file
        
    Returns:
        DataFrame containing the comparison table
    """
    # Create a list of rows for the table
    rows = []
    
    for dataset, metrics in comparison.items():
        for metric, values in metrics.items():
            row = {
                "Dataset": dataset,
                "Metric": metric,
                "Baseline": values["baseline"],
                "PPO": values["ppo"],
                "GRPO": values["grpo"],
                "PPO vs Baseline": values["ppo"] - values["baseline"],
                "GRPO vs Baseline": values["grpo"] - values["baseline"],
                "GRPO vs PPO": values["grpo"] - values["ppo"],
                "PPO vs Baseline (%)": (values["ppo"] / values["baseline"] - 1) * 100 if values["baseline"] else 0,
                "GRPO vs Baseline (%)": (values["grpo"] / values["baseline"] - 1) * 100 if values["baseline"] else 0,
                "GRPO vs PPO (%)": (values["grpo"] / values["ppo"] - 1) * 100 if values["ppo"] else 0
            }
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    return df

def create_comparison_charts(
    comparison: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str
):
    """
    Create comparison charts and save them to files.
    
    Args:
        comparison: Comparison data
        output_dir: Directory to save charts
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})
    
    # For each dataset, create a chart comparing metrics across methods
    for dataset, metrics in comparison.items():
        # Create DataFrame for plotting
        plot_data = []
        
        for metric, values in metrics.items():
            for method, value in values.items():
                plot_data.append({
                    "Metric": metric,
                    "Method": method.capitalize(),
                    "Score": value
                })
                
        plot_df = pd.DataFrame(plot_data)
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(
            data=plot_df,
            x="Metric",
            y="Score",
            hue="Method",
            palette={"Baseline": "blue", "Ppo": "green", "Grpo": "red"}
        )
        
        chart.set_title(f"Performance Comparison on {dataset}")
        chart.set_xlabel("Metric")
        chart.set_ylabel("Score")
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"{dataset}_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    # Create a summary chart combining results across datasets
    summary_data = []
    
    for dataset, metrics in comparison.items():
        for metric, values in metrics.items():
            for method, value in values.items():
                summary_data.append({
                    "Dataset": dataset,
                    "Metric": metric,
                    "Method": method.capitalize(),
                    "Score": value
                })
                
    summary_df = pd.DataFrame(summary_data)
    
    # Create grouped bar chart for each metric
    metrics_list = summary_df["Metric"].unique()
    
    for metric in metrics_list:
        metric_df = summary_df[summary_df["Metric"] == metric]
        
        plt.figure(figsize=(12, 6))
        chart = sns.barplot(
            data=metric_df,
            x="Dataset",
            y="Score",
            hue="Method",
            palette={"Baseline": "blue", "Ppo": "green", "Grpo": "red"}
        )
        
        chart.set_title(f"{metric} Comparison Across Datasets")
        chart.set_xlabel("Dataset")
        chart.set_ylabel("Score")
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
    # Create a heatmap of improvements over baseline
    improvement_data = []
    
    for dataset, metrics in comparison.items():
        for metric, values in metrics.items():
            ppo_improvement = values["ppo"] - values["baseline"]
            grpo_improvement = values["grpo"] - values["baseline"]
            
            improvement_data.append({
                "Dataset": dataset,
                "Metric": metric,
                "PPO Improvement": ppo_improvement,
                "GRPO Improvement": grpo_improvement,
                "GRPO vs PPO": grpo_improvement - ppo_improvement
            })
            
    improvement_df = pd.DataFrame(improvement_data)
    
    # Pivot for heatmap
    ppo_pivot = improvement_df.pivot(index="Dataset", columns="Metric", values="PPO Improvement")
    grpo_pivot = improvement_df.pivot(index="Dataset", columns="Metric", values="GRPO Improvement")
    diff_pivot = improvement_df.pivot(index="Dataset", columns="Metric", values="GRPO vs PPO")
    
    # Create heatmaps
    for name, data in [("PPO_vs_Baseline", ppo_pivot), ("GRPO_vs_Baseline", grpo_pivot), ("GRPO_vs_PPO", diff_pivot)]:
        plt.figure(figsize=(12, 8))
        
        # Create a diverging colormap centered at zero
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        
        sns.heatmap(
            data,
            cmap=cmap,
            center=0,
            annot=True,
            fmt=".3f",
            linewidths=.5
        )
        
        plt.title(f"{name.replace('_', ' ')} Improvement")
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f"{name}_heatmap.png")
        plt.savefig(output_file, dpi=300)
        plt.close()

def main():
    """Run comparison analysis."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    logger.info("Loading baseline results")
    baseline_results = load_results(args.baseline_results)
    
    logger.info("Loading PPO results")
    ppo_results = load_results(args.ppo_results)
    
    logger.info("Loading GRPO results")
    grpo_results = load_results(args.grpo_results)
    
    # Parse dataset and metric filters
    dataset_filter = args.datasets.split(",") if args.datasets else None
    metrics_filter = args.metrics.split(",") if args.metrics else None
    
    # Extract common datasets and metrics
    logger.info("Extracting common datasets and metrics")
    comparison = extract_common_datasets_and_metrics(
        baseline_results,
        ppo_results,
        grpo_results,
        dataset_filter,
        metrics_filter
    )
    
    if not comparison:
        logger.error("No common datasets or metrics found")
        sys.exit(1)
    
    # Create comparison table
    logger.info("Creating comparison table")
    table_file = os.path.join(args.output_dir, "comparison_table.csv")
    table = create_comparison_table(comparison, table_file)
    
    # Print summary statistics
    logger.info("Summary Statistics:")
    avg_ppo_improvement = table["PPO vs Baseline"].mean()
    avg_grpo_improvement = table["GRPO vs Baseline"].mean()
    avg_grpo_vs_ppo = table["GRPO vs PPO"].mean()
    
    logger.info(f"Average PPO improvement over baseline: {avg_ppo_improvement:.4f}")
    logger.info(f"Average GRPO improvement over baseline: {avg_grpo_improvement:.4f}")
    logger.info(f"Average GRPO improvement over PPO: {avg_grpo_vs_ppo:.4f}")
    
    # Save summary statistics
    summary_stats = {
        "avg_ppo_improvement": avg_ppo_improvement,
        "avg_grpo_improvement": avg_grpo_improvement,
        "avg_grpo_vs_ppo": avg_grpo_vs_ppo,
        "num_datasets": len(comparison),
        "num_metrics": sum(len(metrics) for metrics in comparison.values()),
        "datasets": list(comparison.keys()),
        "metrics": list(set().union(*[metrics.keys() for metrics in comparison.values()]))
    }
    
    with open(os.path.join(args.output_dir, "summary_stats.json"), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create comparison charts
    logger.info("Creating comparison charts")
    charts_dir = os.path.join(args.output_dir, "charts")
    create_comparison_charts(comparison, charts_dir)
    
    logger.info(f"Comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 