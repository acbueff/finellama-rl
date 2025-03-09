#!/usr/bin/env python
"""
Script to run hyperparameter sweeps for PPO and GRPO optimization methods.
This script uses Weights & Biases to track experiments and perform hyperparameter optimization.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import subprocess
from typing import Dict, Any, List, Union

import torch
import numpy as np
from transformers import set_seed

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps for PPO and GRPO")
    
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["ppo", "grpo", "both"],
        default="both",
        help="Which method to optimize (ppo, grpo, or both)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="huggyllama/finemath-llama-3b",
        help="Path to pretrained model (or Hugging Face model ID)"
    )
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--eval_data_path", 
        type=str, 
        required=True,
        help="Path to evaluation data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="hyperparams",
        help="Directory to save sweep results and configs"
    )
    parser.add_argument(
        "--num_trials", 
        type=int, 
        default=10,
        help="Number of trials to run for each method"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="finellama-rl",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None,
        help="W&B entity name"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()

def get_ppo_sweep_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for PPO hyperparameter sweep.
    
    Args:
        base_config: Base configuration to extend
        
    Returns:
        W&B sweep configuration
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval_mean_reward",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {
                "min": 1e-6,
                "max": 5e-5,
                "distribution": "log_uniform"
            },
            "ppo_epochs": {
                "values": [1, 2, 3, 4, 5]
            },
            "clip_range": {
                "min": 0.05,
                "max": 0.3,
                "distribution": "uniform"
            },
            "value_clip_range": {
                "min": 0.05,
                "max": 0.3,
                "distribution": "uniform"
            },
            "gamma": {
                "min": 0.9,
                "max": 0.999,
                "distribution": "uniform"
            },
            "lam": {
                "min": 0.9,
                "max": 0.999,
                "distribution": "uniform"
            },
            "mini_batch_size": {
                "values": [4, 8, 16]
            },
            "entropy_coef": {
                "min": 0.0,
                "max": 0.05,
                "distribution": "uniform"
            },
            "vf_coef": {
                "min": 0.1,
                "max": 1.0,
                "distribution": "uniform"
            },
            "kl_penalty_weight": {
                "min": 0.0,
                "max": 0.2,
                "distribution": "uniform"
            },
            "temperature": {
                "min": 0.5,
                "max": 1.5,
                "distribution": "uniform"
            }
        }
    }

def get_grpo_sweep_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get configuration for GRPO hyperparameter sweep.
    
    Args:
        base_config: Base configuration to extend
        
    Returns:
        W&B sweep configuration
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "eval_mean_reward",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {
                "min": 1e-6,
                "max": 5e-5,
                "distribution": "log_uniform"
            },
            "grpo_epochs": {
                "values": [1, 2, 3, 4, 5]
            },
            "preference_threshold": {
                "min": 0.05,
                "max": 0.3,
                "distribution": "uniform"
            },
            "beta": {
                "min": 0.01,
                "max": 0.5,
                "distribution": "log_uniform"
            },
            "mini_batch_size": {
                "values": [4, 8, 16]
            },
            "kl_penalty_weight": {
                "min": 0.0,
                "max": 0.2,
                "distribution": "uniform"
            },
            "paired_temperature": {
                "min": 0.7,
                "max": 1.5,
                "distribution": "uniform"
            },
            "preference_margin": {
                "min": 0.01,
                "max": 0.2,
                "distribution": "uniform"
            },
            "diversity_coefficient": {
                "min": 0.0,
                "max": 0.3,
                "distribution": "uniform"
            }
        }
    }

def update_config_with_params(
    base_config: Dict[str, Any], 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update base configuration with sweep parameters.
    
    Args:
        base_config: Base configuration to update
        params: Parameters from W&B sweep
        
    Returns:
        Updated configuration
    """
    # Create a deep copy of the base config
    config = {**base_config}
    
    # Update config with sweep parameters
    for key, value in params.items():
        config[key] = value
        
    return config

def run_ppo_trial(
    config: Dict[str, Any],
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    trial_id: str
) -> Dict[str, Any]:
    """
    Run a single PPO trial with the given hyperparameters.
    
    Args:
        config: Configuration with hyperparameters
        model_path: Path to pretrained model
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        output_dir: Base output directory
        trial_id: Unique ID for this trial
        
    Returns:
        Dictionary with trial results
    """
    # Create trial-specific output directory
    trial_dir = os.path.join(output_dir, "ppo", f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(trial_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    # Create command to run training
    cmd = [
        "python", "../scripts/run_ppo_training.py",
        "--config", config_path,
        "--model_path", model_path,
        "--train_data_path", train_data_path,
        "--eval_data_path", eval_data_path,
        "--output_dir", trial_dir,
    ]
    
    # Run command
    logger.info(f"Running PPO trial {trial_id} with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Capture output
    stdout, stderr = process.communicate()
    
    # Check if training succeeded
    if process.returncode != 0:
        logger.error(f"PPO trial {trial_id} failed: {stderr}")
        return {"success": False, "error": stderr}
    
    # Load and return results
    results_path = os.path.join(trial_dir, "training_stats.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return {"success": True, "results": results}
    else:
        return {"success": False, "error": "No results file found"}

def run_grpo_trial(
    config: Dict[str, Any],
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    trial_id: str
) -> Dict[str, Any]:
    """
    Run a single GRPO trial with the given hyperparameters.
    
    Args:
        config: Configuration with hyperparameters
        model_path: Path to pretrained model
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        output_dir: Base output directory
        trial_id: Unique ID for this trial
        
    Returns:
        Dictionary with trial results
    """
    # Create trial-specific output directory
    trial_dir = os.path.join(output_dir, "grpo", f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(trial_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    # Create command to run training
    cmd = [
        "python", "../scripts/run_grpo_training.py",
        "--config", config_path,
        "--model_path", model_path,
        "--train_data_path", train_data_path,
        "--eval_data_path", eval_data_path,
        "--output_dir", trial_dir,
    ]
    
    # Run command
    logger.info(f"Running GRPO trial {trial_id} with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Capture output
    stdout, stderr = process.communicate()
    
    # Check if training succeeded
    if process.returncode != 0:
        logger.error(f"GRPO trial {trial_id} failed: {stderr}")
        return {"success": False, "error": stderr}
    
    # Load and return results
    results_path = os.path.join(trial_dir, "training_stats.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return {"success": True, "results": results}
    else:
        return {"success": False, "error": "No results file found"}

def run_ppo_sweep(
    base_config: Dict[str, Any],
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    wandb_project: str,
    wandb_entity: str,
    num_trials: int
):
    """
    Run a hyperparameter sweep for PPO.
    
    Args:
        base_config: Base configuration
        model_path: Path to pretrained model
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        output_dir: Base output directory
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        num_trials: Number of trials to run
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Please install it with 'pip install wandb'")
        return
    
    # Create sweep configuration
    sweep_config = get_ppo_sweep_config(base_config)
    
    # Initialize W&B
    wandb.init(project=wandb_project, entity=wandb_entity)
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=wandb_project,
        entity=wandb_entity
    )
    
    # Define the sweep function
    def sweep_function():
        # Initialize W&B run
        run = wandb.init()
        
        # Get trial parameters
        params = dict(run.config)
        
        # Update config with parameters
        config = update_config_with_params(base_config, params)
        
        # Run trial
        trial_results = run_ppo_trial(
            config=config,
            model_path=model_path,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            output_dir=output_dir,
            trial_id=run.id
        )
        
        # Log results to W&B
        if trial_results["success"]:
            results = trial_results["results"]
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    wandb.log({key: value})
        else:
            wandb.log({"error": True})
            
        # Finish run
        run.finish()
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_function, count=num_trials)

def run_grpo_sweep(
    base_config: Dict[str, Any],
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    wandb_project: str,
    wandb_entity: str,
    num_trials: int
):
    """
    Run a hyperparameter sweep for GRPO.
    
    Args:
        base_config: Base configuration
        model_path: Path to pretrained model
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        output_dir: Base output directory
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        num_trials: Number of trials to run
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed. Please install it with 'pip install wandb'")
        return
    
    # Create sweep configuration
    sweep_config = get_grpo_sweep_config(base_config)
    
    # Initialize W&B
    wandb.init(project=wandb_project, entity=wandb_entity)
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=wandb_project,
        entity=wandb_entity
    )
    
    # Define the sweep function
    def sweep_function():
        # Initialize W&B run
        run = wandb.init()
        
        # Get trial parameters
        params = dict(run.config)
        
        # Update config with parameters
        config = update_config_with_params(base_config, params)
        
        # Run trial
        trial_results = run_grpo_trial(
            config=config,
            model_path=model_path,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            output_dir=output_dir,
            trial_id=run.id
        )
        
        # Log results to W&B
        if trial_results["success"]:
            results = trial_results["results"]
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    wandb.log({key: value})
        else:
            wandb.log({"error": True})
            
        # Finish run
        run.finish()
    
    # Run sweep
    wandb.agent(sweep_id, function=sweep_function, count=num_trials)

def main():
    """Run hyperparameter sweeps."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Run sweeps
    if args.method in ["ppo", "both"]:
        logger.info("Running PPO hyperparameter sweep")
        run_ppo_sweep(
            base_config=base_config,
            model_path=args.model_path,
            train_data_path=args.train_data_path,
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            num_trials=args.num_trials
        )
    
    if args.method in ["grpo", "both"]:
        logger.info("Running GRPO hyperparameter sweep")
        run_grpo_sweep(
            base_config=base_config,
            model_path=args.model_path,
            train_data_path=args.train_data_path,
            eval_data_path=args.eval_data_path,
            output_dir=args.output_dir,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            num_trials=args.num_trials
        )
    
    logger.info("Hyperparameter sweeps complete")

if __name__ == "__main__":
    main() 