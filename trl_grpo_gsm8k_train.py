#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TRL-based GRPO training for Qwen2.5 3B on GSM8K dataset.
This implementation uses Hugging Face's TRL library for GRPO.
"""

import os
import sys
import yaml
import json
import argparse
import logging
import random
import numpy as np
import time
import re
import torch
from typing import Dict, List, Any, Tuple, Optional, Union
from datasets import Dataset

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# TRL imports
from trl import GRPOConfig, GRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class GSM8KDataset:
    """Dataset loader for GSM8K GRPO training."""
    
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 512):
        """Load GSM8K dataset from JSONL file."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data from JSONL
        logger.info(f"Loading data from {data_path}")
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples")
        
    def to_hf_dataset(self) -> Dataset:
        """Convert to Hugging Face Dataset format."""
        formatted_data = []
        for item in self.data:
            formatted_data.append({
                "prompt": self.create_prompt(item["question"]),
                "target_answer": item["answer"]
            })
        
        return Dataset.from_list(formatted_data)
    
    @staticmethod
    def create_prompt(question: str) -> str:
        """Create a prompt for GSM8K problem solving."""
        return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from text."""
    
    # Try to find answer patterns like "therefore, the answer is X"
    patterns = [
        r"the answer is\s*(\d[\d\.,]*)\.?$",
        r"equals\s*(\d[\d\.,]*)\.?$",
        r"=\s*(\d[\d\.,]*)\.?$",
        r"(\d[\d\.,]*)\s*(?:dollars|units|meters|kg|pounds)\.?$",
        r"(\d[\d\.,]*)\.?$"  # Last number in text as fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback to last number in text
    numbers = re.findall(r"(\d[\d\.,]*)", text)
    if numbers:
        return numbers[-1]
    
    return None

def normalize_answer(answer: str) -> str:
    """Normalize extracted answer for comparison."""
    if answer is None:
        return ""
    
    # Remove commas, spaces, and trailing zeros
    answer = answer.replace(",", "").strip()
    
    # Try to convert to float then back to string to normalize format
    try:
        num = float(answer)
        # Convert to int if no decimal part
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer

def has_step_by_step(text: str) -> bool:
    """Check if a response has step-by-step reasoning."""
    
    # Check for numbered steps or calculation patterns
    has_steps = re.search(r"step\s*\d|let\'s|first|second|third|finally|=", text, re.IGNORECASE)
    
    # Check for multiple calculation lines
    calc_lines = re.findall(r"(\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+)", text)
    
    return (has_steps is not None) or len(calc_lines) >= 2

def correctness_reward_func(prompts, completions, target_answer, **kwargs):
    """
    Reward function based on mathematical correctness.
    Checks if the answer extracted from a completion matches the target answer.
    
    Args:
        prompts: List of problem prompts
        completions: List of model completions
        target_answer: List of target answers
    
    Returns:
        List of rewards
    """
    rewards = []
    
    for i, (_, completion, answer) in enumerate(zip(prompts, completions, target_answer)):
        # Extract answer from completion
        extracted = extract_answer(completion)
        
        # Normalize answers
        normalized_extracted = normalize_answer(extracted) if extracted is not None else ""
        normalized_target = normalize_answer(answer)
        
        # Check correctness
        correct = normalized_extracted == normalized_target
        
        # Log sample for debugging
        if i == 0 or random.random() < 0.05:  # Log first example and ~5% of others
            logger.info(f"Example answer extraction:")
            logger.info(f"  Target: {answer} (normalized: {normalized_target})")
            logger.info(f"  Extracted: {extracted} (normalized: {normalized_extracted})")
            logger.info(f"  Correct: {correct}")
        
        # Assign reward (1.0 if correct, 0.0 if incorrect)
        rewards.append(1.0 if correct else 0.0)
    
    return rewards

def reasoning_reward_func(prompts, completions, **kwargs):
    """
    Reward function based on step-by-step reasoning.
    Rewards completions that include clear reasoning steps.
    
    Args:
        prompts: List of problem prompts
        completions: List of model completions
    
    Returns:
        List of rewards
    """
    rewards = []
    
    for completion in completions:
        # Check for step-by-step reasoning
        has_steps = has_step_by_step(completion)
        
        # Length-based component - penalize extremely short or long explanations
        length = len(completion.split())
        length_component = min(1.0, max(0.0, (length / 100) if length < 100 else (200 / length) if length > 200 else 1.0))
        
        # Combine reasoning steps and length
        reward = 0.5 if has_steps else 0.0
        reward += 0.2 * length_component
        
        rewards.append(reward)
    
    return rewards

def brevity_reward_func(prompts, completions, **kwargs):
    """
    Reward function for brevity.
    Moderately rewards shorter completions.
    
    Args:
        prompts: List of problem prompts
        completions: List of model completions
    
    Returns:
        List of rewards
    """
    rewards = []
    
    for completion in completions:
        # Count tokens roughly by splitting on whitespace
        length = len(completion.split())
        
        # Reward brevity but don't penalize extremely short responses
        # Typical good responses are 50-150 tokens
        if length < 30:
            reward = 0.1  # Too short, minimal reward
        elif length <= 150:
            reward = 0.3 * (1 - length / 200)  # Linear scaling, max at shortest
        else:
            reward = max(0.0, 0.3 * (1 - length / 400))  # More aggressive penalty for very long responses
        
        rewards.append(reward)
    
    return rewards

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TRL GRPO training for Qwen on GSM8K")
    
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="configs/qwen2.5_3b_config.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--grpo_config", 
        type=str, 
        default="configs/qwen_gsm8k_grpo_config.yaml",
        help="Path to GRPO configuration file"
    )
    
    parser.add_argument(
        "--train_data", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--eval_data", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json",
        help="Path to evaluation data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for TRL GRPO training."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "progress"), exist_ok=True)
    
    # Create start marker file
    start_info = {
        "status": "started", 
        "timestamp": time.time(), 
        "args": vars(args)
    }
    with open(os.path.join(args.output_dir, "progress", "training_started.json"), "w") as f:
        json.dump(start_info, f)
    
    # Load configurations
    logger.info(f"Loading model config from {args.model_config}")
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    
    logger.info(f"Loading GRPO config from {args.grpo_config}")
    with open(args.grpo_config, "r") as f:
        grpo_config = yaml.safe_load(f)
    
    # Load tokenizer
    model_path = model_config.get("model_name_or_path", "Qwen/Qwen2.5-3B")
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=model_config.get("use_fast_tokenizer", True),
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    
    # Make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset_loader = GSM8KDataset(args.train_data, tokenizer)
    train_dataset = train_dataset_loader.to_hf_dataset()
    
    # Set up GRPO configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=grpo_config.get("num_epochs", 2),
        per_device_train_batch_size=grpo_config.get("batch_size", 2),
        per_device_eval_batch_size=grpo_config.get("batch_size", 2),
        gradient_accumulation_steps=grpo_config.get("gradient_accumulation_steps", 16),
        learning_rate=grpo_config.get("learning_rate", 2e-5),
        weight_decay=grpo_config.get("weight_decay", 0.01),
        max_grad_norm=grpo_config.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
        max_prompt_length=grpo_config.get("prompt_max_length", 256),
        max_completion_length=grpo_config.get("response_max_length", 128),
        num_generations=grpo_config.get("num_sample_pairs", 4),
        logging_steps=grpo_config.get("log_interval", 5),
        save_steps=grpo_config.get("save_interval", 50),
        evaluation_strategy="steps",
        eval_steps=grpo_config.get("eval_interval", 200),
        warmup_steps=grpo_config.get("warmup_steps", 50),
        lr_scheduler_type=grpo_config.get("lr_scheduler_type", "cosine"),
        bf16=True,  # Use bfloat16 for faster training
        bf16_full_eval=True,
        optim="adamw_torch",
        seed=args.seed,
        fp16=False,  # Don't use fp16 when bf16 is enabled
        dataloader_drop_last=True,
        dataloader_num_workers=grpo_config.get("num_workers", 2),
        disable_dropout=True,  # Disable dropout for more stable training
        temperature=grpo_config.get("temperature", 0.7),
        top_p=grpo_config.get("top_p", 0.9),
        top_k=grpo_config.get("top_k", 20),
        epsilon=0.2,  # PPO clipping parameter
        beta=grpo_config.get("kl_penalty_weight", 0.1),  # KL coefficient
        scale_rewards=not grpo_config.get("disable_reward_scaling", False),
        mask_truncated_completions=True,  # Better training stability
        log_completions=True,  # Log a sample of completions for debugging
    )
    
    # Setup 4-bit quantization if enabled
    if model_config.get("load_in_4bit", True):
        quatization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quatization_config = None
    
    # Setup LoRA config
    lora_config = LoraConfig(
        r=model_config.get("peft_config", {}).get("r", 16),
        lora_alpha=model_config.get("peft_config", {}).get("lora_alpha", 32),
        lora_dropout=model_config.get("peft_config", {}).get("lora_dropout", 0.05),
        target_modules=model_config.get("peft_config", {}).get("target_modules", 
                                                             ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=model_config.get("peft_config", {}).get("bias", "none"),
        task_type="CAUSAL_LM"
    )
    
    # Print training configuration summary
    logger.info(f"Training configuration summary:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Use LoRA: {model_config.get('use_peft', True)}")
    logger.info(f"  Load in 4-bit: {model_config.get('load_in_4bit', True)}")
    logger.info(f"  Training data: {args.train_data}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Number of generations: {training_args.num_generations}")
    logger.info(f"  Training epochs: {training_args.num_train_epochs}")
    
    # Create GRPO Trainer
    trainer = GRPOTrainer(
        model=model_path,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config if model_config.get("use_peft", True) else None,
        model_init_kwargs={
            "quantization_config": quatization_config,
            "trust_remote_code": model_config.get("trust_remote_code", True),
            "attn_implementation": model_config.get("attn_implementation", "flash_attention_2"),
            "device_map": "auto",
        },
        reward_funcs=[
            # Correctness reward (most important)
            correctness_reward_func,
            # Reasoning process reward 
            reasoning_reward_func,
            # Brevity reward (mild preference for concise answers)
            brevity_reward_func
        ]
    )
    
    # Start training
    logger.info("Starting GRPO training")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    
    # Update progress file
    with open(os.path.join(args.output_dir, "progress", "training_progress.json"), "w") as f:
        json.dump({"status": "completed", "timestamp": time.time()}, f)
    
    logger.info("GRPO training completed")

if __name__ == "__main__":
    main() 