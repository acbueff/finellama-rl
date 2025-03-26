#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified PPO training script for Qwen2.5 3B on GSM8K dataset.
"""

import os
import sys
import yaml
import json
import argparse
import logging
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler,
    set_seed,
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class GSM8KDataset(Dataset):
    """Custom dataset for GSM8K PPO training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as a dict with 'prompt' and 'reference'
        return {
            "prompt": item["question"],
            "reference": item["answer"]
        }

def create_prompt(question: str) -> str:
    """Create a prompt for GSM8K problem solving."""
    return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def compute_rewards(
    model, 
    tokenizer, 
    ref_model, 
    prompts: List[str], 
    responses: List[str], 
    references: List[str] = None
) -> torch.Tensor:
    """
    Compute rewards for PPO training based on mathematical correctness and KL divergence.
    
    Args:
        model: Current policy model
        tokenizer: Tokenizer
        ref_model: Reference model for KL penalty
        prompts: List of problem prompts
        responses: List of generated responses
        references: List of reference answers
        
    Returns:
        Tensor of rewards for each response
    """
    batch_size = len(prompts)
    rewards = torch.zeros(batch_size, device=model.device)
    
    # 1. Compute correctness reward based on final answer extraction
    if references:
        for i, (response, reference) in enumerate(zip(responses, references)):
            # Extract final answers (simplified)
            pred_answer = extract_answer(response)
            ref_answer = extract_answer(reference)
            
            # Award points for correct answer
            if pred_answer is not None and ref_answer is not None:
                if normalize_answer(pred_answer) == normalize_answer(ref_answer):
                    rewards[i] += 1.0
    
    # 2. Process reward based on step-by-step reasoning
    for i, response in enumerate(responses):
        # Check if response has step-by-step reasoning
        if has_step_by_step(response):
            rewards[i] += 0.5
    
    # 3. Apply KL penalty to avoid diverging too much from reference model
    kl_penalty_weight = 0.05
    kl_penalties = compute_kl_divergence(model, tokenizer, ref_model, prompts, responses)
    rewards = rewards - kl_penalty_weight * kl_penalties
    
    return rewards

def compute_kl_divergence(
    model, 
    tokenizer, 
    ref_model, 
    prompts: List[str], 
    responses: List[str]
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference model.
    
    Args:
        model: Current policy model
        tokenizer: Tokenizer
        ref_model: Reference model
        prompts: List of problem prompts
        responses: List of generated responses
        
    Returns:
        Tensor of KL divergences
    """
    batch_size = len(prompts)
    kl_divs = torch.zeros(batch_size, device=model.device)
    
    # Create full sequences (prompt + response)
    sequences = [create_prompt(p) + " " + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    inputs = tokenizer(
        sequences, 
        padding=True, 
        truncation=True,
        max_length=512, 
        return_tensors="pt"
    ).to(model.device)
    
    # Get logits from both models
    with torch.no_grad():
        outputs_ref = ref_model(**inputs)
        outputs_policy = model(**inputs)
        
        ref_logits = outputs_ref.logits[:, :-1, :]  # Shift right for prediction
        policy_logits = outputs_policy.logits[:, :-1, :]
        
        # Get KL divergence per token
        ref_probs = F.softmax(ref_logits, dim=-1)
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        
        kl = torch.sum(ref_probs * (torch.log(ref_probs + 1e-10) - policy_log_probs), dim=-1)
        
        # Get prompt lengths to extract only response KL
        prompt_tokens = tokenizer(
            [create_prompt(p) for p in prompts], 
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        prompt_lens = torch.sum(prompt_tokens.attention_mask, dim=1)
        
        # Average KL only for response tokens
        for i in range(batch_size):
            resp_kl = kl[i, prompt_lens[i]:]
            kl_divs[i] = torch.mean(resp_kl[resp_kl > 0])
    
    return kl_divs

def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from text."""
    import re
    
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
    import re
    
    # Check for numbered steps or calculation patterns
    has_steps = re.search(r"step\s*\d|let\'s|first|second|third|finally|=", text, re.IGNORECASE)
    
    # Check for multiple calculation lines
    calc_lines = re.findall(r"(\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+)", text)
    
    return (has_steps is not None) or len(calc_lines) >= 2

def prepare_batch_for_training(
    prompts: List[str], 
    responses: List[str], 
    rewards: torch.Tensor,
    tokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of data for PPO training.
    
    Args:
        prompts: List of problem prompts
        responses: List of generated responses
        rewards: Tensor of rewards
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dict with input tensors for training
    """
    # Create full sequences (prompt + response)
    sequences = [create_prompt(p) + " " + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    inputs = tokenizer(
        sequences, 
        padding=True, 
        truncation=True,
        max_length=max_length, 
        return_tensors="pt"
    )
    
    # Get prompt lengths to identify response tokens
    prompt_tokens = tokenizer(
        [create_prompt(p) for p in prompts], 
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    prompt_lens = torch.sum(prompt_tokens.attention_mask, dim=1)
    
    # Create a mask for response tokens (1 for response tokens, 0 for prompt tokens)
    response_mask = torch.zeros_like(inputs.attention_mask)
    for i in range(len(prompts)):
        response_mask[i, prompt_lens[i]:] = inputs.attention_mask[i, prompt_lens[i]:]
    
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "response_mask": response_mask,
        "rewards": rewards
    }

def generate_responses(
    model, 
    tokenizer, 
    prompts: List[str], 
    temperature: float = 1.0,
    max_new_tokens: int = 256,
    do_sample: bool = True
) -> List[str]:
    """
    Generate responses for a batch of prompts.
    
    Args:
        model: Model to generate with
        tokenizer: Tokenizer
        prompts: List of problem prompts
        temperature: Sampling temperature
        max_new_tokens: Maximum number of tokens to generate
        do_sample: Whether to sample or use greedy decoding
        
    Returns:
        List of generated response strings
    """
    # Format prompts
    formatted_prompts = [create_prompt(p) for p in prompts]
    
    # Generate responses
    responses = []
    
    # Process in batches of 1 to avoid OOM
    for prompt in formatted_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Extract only the generated response (not the prompt)
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        responses.append(response)
    
    return responses

def train_ppo(
    model,
    ref_model,
    tokenizer,
    dataset,
    optimizer,
    lr_scheduler,
    config: Dict[str, Any],
    accelerator: Accelerator,
) -> Dict[str, List[float]]:
    """
    Train model with PPO.
    
    Args:
        model: Model to train
        ref_model: Reference model for KL penalty
        tokenizer: Tokenizer
        dataset: Training dataset
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        config: Training configuration
        accelerator: Accelerator for distributed training
        
    Returns:
        Dict with training statistics
    """
    # Training stats
    stats = {
        "rewards": [],
        "kl_divs": [],
        "policy_loss": [],
        "value_loss": [],
        "total_loss": []
    }
    
    # Training parameters
    ppo_epochs = config.get("ppo_epochs", 4)
    num_rollouts = config.get("num_rollouts", 32)
    mini_batch_size = config.get("mini_batch_size", 8)
    batch_size = config.get("batch_size", 64)
    clip_range = config.get("clip_range", 0.2)
    vf_coef = config.get("vf_coef", 0.5)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    max_steps = config.get("max_steps", 5000)
    save_interval = config.get("save_interval", 200)
    log_interval = config.get("log_interval", 10)
    
    # PPO training loop
    global_step = 0
    epoch = 0
    
    # Setup progress bar
    progress_bar = tqdm(total=max_steps, desc=f"Training epoch {epoch}")
    
    # Main training loop
    while global_step < max_steps:
        # Sample rollouts from dataset
        indices = random.sample(range(len(dataset)), min(num_rollouts, len(dataset)))
        rollouts = [dataset[i] for i in indices]
        
        # Extract prompts and references
        prompts = [item["prompt"] for item in rollouts]
        references = [item["reference"] for item in rollouts]
        
        # Generate responses
        with accelerator.unwrap_model(model).disable_adapter() if hasattr(accelerator.unwrap_model(model), "disable_adapter") else nullcontext():
            responses = generate_responses(
                accelerator.unwrap_model(model),
                tokenizer,
                prompts,
                temperature=config.get("temperature", 1.0),
                max_new_tokens=config.get("response_max_length", 256),
                do_sample=config.get("do_sample", True)
            )
        
        # Compute rewards
        with torch.no_grad():
            rewards = compute_rewards(
                accelerator.unwrap_model(model),
                tokenizer,
                accelerator.unwrap_model(ref_model),
                prompts,
                responses,
                references
            )
        
        # Log stats
        avg_reward = rewards.mean().item()
        stats["rewards"].append(avg_reward)
        
        if global_step % log_interval == 0:
            logger.info(f"Step {global_step}: Average reward: {avg_reward:.4f}")
        
        # PPO update loop
        for _ in range(ppo_epochs):
            # Prepare minibatches
            batch_indices = list(range(len(prompts)))
            random.shuffle(batch_indices)
            
            for i in range(0, len(batch_indices), mini_batch_size):
                mini_batch_indices = batch_indices[i:i + mini_batch_size]
                
                mini_batch_prompts = [prompts[j] for j in mini_batch_indices]
                mini_batch_responses = [responses[j] for j in mini_batch_indices]
                mini_batch_rewards = rewards[mini_batch_indices]
                
                # Prepare inputs
                train_inputs = prepare_batch_for_training(
                    mini_batch_prompts,
                    mini_batch_responses,
                    mini_batch_rewards,
                    tokenizer,
                    max_length=config.get("max_length", 512)
                )
                
                # Move inputs to device
                train_inputs = {k: v.to(accelerator.device) for k, v in train_inputs.items()}
                
                # Forward pass with current policy
                outputs = model(
                    input_ids=train_inputs["input_ids"],
                    attention_mask=train_inputs["attention_mask"],
                    labels=train_inputs["input_ids"],
                    output_hidden_states=True
                )
                
                # Compute loss (simplified PPO objective)
                logits = outputs.logits[:, :-1, :]  # shift left
                labels = train_inputs["input_ids"][:, 1:]  # shift right
                
                # Get log probs for actions
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Gather log probs of chosen tokens
                chosen_log_probs = torch.gather(
                    log_probs, 
                    2, 
                    labels.unsqueeze(-1)
                ).squeeze(-1)
                
                # Get log probs from reference model
                with torch.no_grad():
                    ref_outputs = ref_model(
                        input_ids=train_inputs["input_ids"],
                        attention_mask=train_inputs["attention_mask"]
                    )
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_chosen_log_probs = torch.gather(
                        ref_log_probs, 
                        2, 
                        labels.unsqueeze(-1)
                    ).squeeze(-1)
                
                # Compute probabilities and ratio
                ratio = torch.exp(chosen_log_probs - ref_chosen_log_probs)
                
                # Only consider response tokens
                valid_tokens = train_inputs["response_mask"][:, 1:]
                ratio = ratio * valid_tokens
                
                # Compute advantages (use rewards directly as advantage)
                advantages = train_inputs["rewards"].unsqueeze(-1).expand_as(ratio)
                advantages = advantages * valid_tokens
                
                # Compute PPO policy loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = torch.sum(policy_loss) / torch.sum(valid_tokens)
                
                # Simplified value function loss
                value_loss = outputs.loss  # Use language modeling loss
                
                # Total loss
                loss = policy_loss + vf_coef * value_loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()
                
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                # Update stats
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["total_loss"].append(loss.item())
                
                # Log every log_interval steps
                if global_step % log_interval == 0:
                    logger.info(f"Step {global_step}: Loss: {loss.item():.4f}, "
                               f"Policy loss: {policy_loss.item():.4f}, "
                               f"Value loss: {value_loss.item():.4f}")
                
                # Save checkpoint every save_interval steps
                if global_step % save_interval == 0 and global_step > 0:
                    save_checkpoint(
                        accelerator, 
                        model, 
                        tokenizer, 
                        optimizer, 
                        lr_scheduler,
                        epoch, 
                        global_step, 
                        config
                    )
                
                # Increment step counter
                global_step += 1
                progress_bar.update(1)
                
                # Check if we reached max steps
                if global_step >= max_steps:
                    break
            
            # Check if we reached max steps
            if global_step >= max_steps:
                break
        
        # Increment epoch counter
        epoch += 1
        progress_bar.set_description(f"Training epoch {epoch}")
    
    # Final save
    save_checkpoint(
        accelerator, 
        model, 
        tokenizer, 
        optimizer, 
        lr_scheduler,
        epoch, 
        global_step, 
        config
    )
    
    return stats

def evaluate(
    model,
    tokenizer,
    eval_dataset,
    accelerator: Accelerator,
    num_examples: int = 50
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K data.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        accelerator: Accelerator for distributed training
        num_examples: Number of examples to evaluate
        
    Returns:
        Dict with evaluation metrics
    """
    logger.info(f"Evaluating model on {num_examples} examples")
    
    # Sample evaluation examples
    indices = random.sample(range(len(eval_dataset)), min(num_examples, len(eval_dataset)))
    eval_samples = [eval_dataset[i] for i in indices]
    
    # Extract prompts and references
    prompts = [item["prompt"] for item in eval_samples]
    references = [item["reference"] for item in eval_samples]
    
    # Generate responses (use greedy decoding for evaluation)
    with torch.no_grad():
        responses = generate_responses(
            accelerator.unwrap_model(model),
            tokenizer,
            prompts,
            temperature=0.0,  # Greedy decoding
            max_new_tokens=256,
            do_sample=False
        )
    
    # Compute accuracy
    correct = 0
    for response, reference in zip(responses, references):
        pred_answer = extract_answer(response)
        ref_answer = extract_answer(reference)
        
        if pred_answer is not None and ref_answer is not None:
            if normalize_answer(pred_answer) == normalize_answer(ref_answer):
                correct += 1
    
    accuracy = correct / len(prompts)
    logger.info(f"Evaluation accuracy: {accuracy:.4f} ({correct}/{len(prompts)})")
    
    # Save predictions
    eval_results = []
    for i, (prompt, response, reference) in enumerate(zip(prompts, responses, references)):
        pred_answer = extract_answer(response)
        ref_answer = extract_answer(reference)
        is_correct = False
        
        if pred_answer is not None and ref_answer is not None:
            is_correct = normalize_answer(pred_answer) == normalize_answer(ref_answer)
        
        eval_results.append({
            "id": i,
            "prompt": prompt,
            "response": response,
            "reference": reference,
            "predicted_answer": pred_answer,
            "reference_answer": ref_answer,
            "is_correct": is_correct
        })
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(prompts),
        "results": eval_results
    }

def save_checkpoint(
    accelerator,
    model,
    tokenizer,
    optimizer,
    lr_scheduler,
    epoch: int,
    step: int,
    config: Dict[str, Any]
) -> None:
    """
    Save model checkpoint.
    
    Args:
        accelerator: Accelerator for distributed training
        model: Model to save
        tokenizer: Tokenizer to save
        optimizer: Optimizer state to save
        lr_scheduler: Learning rate scheduler state to save
        epoch: Current epoch
        step: Current step
        config: Training configuration
    """
    output_dir = config.get("output_dir", "checkpoints")
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and tokenizer
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save optimizer and scheduler
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save training state
    accelerator.save_state(checkpoint_dir)
    
    # Save config
    with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PPO training for Qwen2.5 3B on GSM8K")
    
    parser.add_argument("--model_config", type=str, default="configs/qwen2.5_3b_config.yaml",
                      help="Path to model configuration file")
    parser.add_argument("--ppo_config", type=str, default="configs/qwen_gsm8k_ppo_config.yaml",
                      help="Path to PPO training configuration file")
    parser.add_argument("--train_data", type=str, default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json",
                      help="Path to GSM8K training data")
    parser.add_argument("--eval_data", type=str, default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json",
                      help="Path to GSM8K validation data")
    parser.add_argument("--output_dir", type=str, default="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/ppo_finetuned",
                      help="Directory to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results",
                      help="Directory to save evaluation results")
    parser.add_argument("--eval_steps", type=int, default=100,
                      help="Evaluate model every eval_steps steps")
    parser.add_argument("--eval_examples", type=int, default=50,
                      help="Number of examples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    return parser.parse_args()

def main():
    """Run PPO training for Qwen2.5 3B on GSM8K."""
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load configurations
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.ppo_config, 'r') as f:
        ppo_config = yaml.safe_load(f)
    
    # Update config with command line arguments
    ppo_config["output_dir"] = args.output_dir
    ppo_config["seed"] = args.seed
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps=ppo_config.get("gradient_accumulation_steps", 8))
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_config['model_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_name_or_path"],
        use_fast=model_config.get("use_fast_tokenizer", True),
        trust_remote_code=model_config.get("trust_remote_code", True)
    )
    
    # Ensure the tokenizer has pad_token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model with quantization
    logger.info(f"Loading policy model from {model_config['model_name_or_path']}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=model_config.get("peft_config", {}).get("r", 16),
        lora_alpha=model_config.get("peft_config", {}).get("lora_alpha", 32),
        lora_dropout=model_config.get("peft_config", {}).get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
        target_modules=model_config.get("peft_config", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=model_config.get("peft_config", {}).get("bias", "none")
    )
    
    # For 4-bit quantization
    if model_config.get("load_in_4bit", True):
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            quantization_config=quantization_config,
            trust_remote_code=model_config.get("trust_remote_code", True),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        policy_model = prepare_model_for_kbit_training(policy_model)
    else:
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name_or_path"],
            trust_remote_code=model_config.get("trust_remote_code", True),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    # Apply LoRA
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()
    
    # Load reference model (for KL penalty)
    logger.info(f"Loading reference model from {model_config['model_name_or_path']}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Load datasets
    train_dataset = GSM8KDataset(args.train_data, tokenizer)
    eval_dataset = GSM8KDataset(args.eval_data, tokenizer)
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
    
    # Create optimizer
    optimizer = AdamW(
        policy_model.parameters(),
        lr=ppo_config.get("learning_rate", 1e-5),
        weight_decay=ppo_config.get("weight_decay", 0.01)
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name=ppo_config.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=ppo_config.get("warmup_steps", 100),
        num_training_steps=ppo_config.get("max_steps", 5000)
    )
    
    # Prepare for distributed training
    policy_model, ref_model, optimizer, lr_scheduler = accelerator.prepare(
        policy_model, ref_model, optimizer, lr_scheduler
    )
    
    # Train
    logger.info("Starting PPO training")
    training_stats = train_ppo(
        policy_model,
        ref_model,
        tokenizer,
        train_dataset,
        optimizer,
        lr_scheduler,
        ppo_config,
        accelerator
    )
    
    # Final evaluation
    logger.info("Running final evaluation")
    eval_results = evaluate(
        policy_model,
        tokenizer,
        eval_dataset,
        accelerator,
        num_examples=args.eval_examples
    )
    
    # Save evaluation results
    with open(os.path.join(args.results_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Save training stats
    with open(os.path.join(args.results_dir, "training_stats.json"), "w") as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("Training and evaluation completed")

if __name__ == "__main__":
    from contextlib import nullcontext
    main() 