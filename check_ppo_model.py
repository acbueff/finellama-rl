#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to check the PPO-trained LoRA adapter
"""

import os
import json
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def check_adapter_config(adapter_path):
    """Check the adapter configuration in the given path."""
    
    # List all files in the adapter path
    logger.info(f"Listing all files in {adapter_path}:")
    for root, dirs, files in os.walk(adapter_path):
        for file in files:
            logger.info(f" - {os.path.join(root, file)}")
    
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        logger.error(f"Adapter config not found at {config_path}")
        return None
    
    # Load and display the adapter config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f"Adapter configuration: {json.dumps(config, indent=2)}")
    
    return config

def get_model_stats(adapter_path):
    """Get statistics about the model."""
    # Check if there are checkpoint directories
    checkpoints = [d for d in os.listdir(adapter_path) if d.startswith("checkpoint-")]
    if checkpoints:
        logger.info(f"Found {len(checkpoints)} checkpoints: {', '.join(checkpoints)}")
    
    # Check training args if available
    train_args_path = os.path.join(adapter_path, "training_args.json")
    if os.path.exists(train_args_path):
        with open(train_args_path, 'r', encoding='utf-8') as f:
            train_args = json.load(f)
        
        logger.info("Training arguments summary:")
        for key in ["lr_scheduler_type", "learning_rate", "num_train_epochs", 
                    "per_device_train_batch_size", "max_steps"]:
            if key in train_args:
                logger.info(f" - {key}: {train_args[key]}")

def main():
    parser = argparse.ArgumentParser(description="Check PPO-trained LoRA adapter")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the LoRA adapter")
    
    args = parser.parse_args()
    
    logger.info(f"Checking LoRA adapter at {args.adapter_path}")
    
    if not os.path.exists(args.adapter_path):
        logger.error(f"Adapter path does not exist: {args.adapter_path}")
        return
    
    # Check adapter config
    config = check_adapter_config(args.adapter_path)
    
    # Get model stats
    get_model_stats(args.adapter_path)
    
    logger.info("Adapter check completed")

if __name__ == "__main__":
    main() 