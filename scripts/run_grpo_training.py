#!/usr/bin/env python
"""
Script to run Gradient-based Reinforcement with Paired Optimization (GRPO) training on FineMath-Llama-3B.
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
from datasets import load_dataset

from src.models.finemathllama import FineMathLlamaModel
from src.rl.reward_model import MathRewardModel, create_reward_model
from src.rl.grpo_trainer import GRPOTrainer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training on FineMath-Llama-3B")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to GRPO configuration file"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default=None,
        help="Path to model configuration file (if different from GRPO config)"
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
        help="Path to training data (processed for GRPO)"
    )
    parser.add_argument(
        "--eval_data_path", 
        type=str, 
        default=None,
        help="Path to evaluation data (optional)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save model checkpoints (overrides config if provided)"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--preference_model_path", 
        type=str, 
        default=None,
        help="Path to pretrained preference model (optional)"
    )
    
    return parser.parse_args()

def main():
    """Run GRPO training on FineMath-Llama-3B."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load GRPO configuration
    with open(args.config, 'r') as f:
        grpo_config = yaml.safe_load(f)
    
    # Load model configuration (if provided separately)
    model_config = None
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
    
    # Override output directory if provided as argument
    if args.output_dir:
        grpo_config["output_dir"] = args.output_dir
    
    # Override preference model path if provided
    if args.preference_model_path:
        grpo_config["preference_model_path"] = args.preference_model_path
        grpo_config["use_preference_model"] = True
    
    # Create output directory
    output_dir = grpo_config.get("output_dir", "checkpoints/grpo_model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset for training
    logger.info(f"Loading training data from {args.train_data_path}")
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    logger.info(f"Loaded {len(train_dataset)} training examples")
    
    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_data_path:
        logger.info(f"Loading evaluation data from {args.eval_data_path}")
        eval_dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
        logger.info(f"Loaded {len(eval_dataset)} evaluation examples")
    
    # Initialize the model
    logger.info(f"Loading model from {args.model_path}")
    if args.resume_from:
        # Resume from checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model_wrapper = FineMathLlamaModel.from_pretrained(args.resume_from)
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
    else:
        # Start training from scratch
        if model_config:
            model_wrapper = FineMathLlamaModel(config=model_config)
        else:
            # Create default model wrapper
            model_wrapper = FineMathLlamaModel(config={
                "model_name_or_path": args.model_path,
                "use_peft": True,
                "load_in_4bit": True
            })
            
        model_wrapper.load_model()
        model = model_wrapper.model
        tokenizer = model_wrapper.tokenizer
    
    # Create reward model
    reward_model = create_reward_model(
        tokenizer=tokenizer,
        reference_model_path=args.model_path,  # Use same model as reference
        config=grpo_config
    )
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config=grpo_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Save combined configuration
    combined_config = {
        "grpo_config": grpo_config,
        "model_config": model_config or model_wrapper.config,
        "args": vars(args)
    }
    
    with open(os.path.join(output_dir, "training_config.yaml"), 'w') as f:
        yaml.dump(combined_config, f)
    
    # Run training
    logger.info("Starting GRPO training")
    training_stats = trainer.train(
        dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Save final training stats
    with open(os.path.join(output_dir, "training_stats.json"), 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main() 