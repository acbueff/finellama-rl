#!/usr/bin/env python
"""
Simplified PPO training script for Qwen2.5 model on GSM8K dataset.
This script tries to avoid compatibility issues with the tokenizer.
"""

import os
import argparse
import logging
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified PPO training for Qwen2.5")
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="SunnyLin/Qwen2.5-7B-DPO-VP",
        help="Hugging Face model ID for Qwen2.5"
    )
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        required=True,
        help="Path to training data (processed for PPO)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/ppo_finetuned",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout rate"
    )
    
    return parser.parse_args()

def load_model_and_tokenizer(model_id: str):
    """
    Load model and tokenizer with proper settings for Qwen2.5.
    """
    logger.info(f"Loading model and tokenizer from {model_id}")
    
    try:
        # First try loading with use_fast=False to avoid tokenizer issues
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,  # Use slow tokenizer to avoid compatibility issues
            padding_side="left",
            trust_remote_code=True
        )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configure 4-bit quantization
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def apply_peft(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Apply LoRA adapter for efficient fine-tuning.
    """
    logger.info("Applying LoRA adapter")
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Apply LoRA adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_model(model, tokenizer, train_dataset, args):
    """
    Fine-tune the model using a simplified version of PPO.
    """
    logger.info("Starting simplified PPO training")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )
    
    # Prepare the dataset for training
    def tokenize_function(examples):
        # Just use the prompts for supervised fine-tuning as a starting point
        prompts = examples["prompt"]
        responses = examples.get("reference", [""] * len(prompts))
        
        # Format as instruction text
        texts = [prompt + response for prompt, response in zip(prompts, responses)]
        
        result = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create the labels (same as input_ids, but -100 for prompt tokens)
        labels = result["input_ids"].clone()
        
        # Mark prompt tokens with -100 so they don't contribute to loss
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer(
                prompt, 
                padding=False, 
                truncation=False, 
                return_tensors="pt"
            )
            prompt_length = prompt_tokens["input_ids"].size(1)
            labels[i, :prompt_length] = -100
            
        result["labels"] = labels
        return result
    
    # Process the dataset
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return model

def main():
    """Main entry point."""
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load the dataset
    logger.info(f"Loading training data from {args.train_data_path}")
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")
    logger.info(f"Loaded {len(train_dataset)} training examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    
    # Apply PEFT for efficient fine-tuning
    model = apply_peft(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Train the model
    model = train_model(model, tokenizer, train_dataset, args)
    
    logger.info(f"Training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 