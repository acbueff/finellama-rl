#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified GRPO training script for Qwen2.5 3B on GSM8K dataset.
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
import signal
from typing import Dict, List, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
    AutoModel
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
    """Custom dataset for GSM8K GRPO training."""
    
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

def generate_response_pairs(
    model,
    tokenizer,
    prompt: str,
    target_answer: str,
    config: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Generate pairs of responses for a given prompt with timeout protection.
    
    Args:
        model: Model to generate responses
        tokenizer: Tokenizer
        prompt: Problem prompt
        target_answer: Reference answer
        config: Configuration dict
        
    Returns:
        List of (response_a, response_b) tuples
    """
    num_pairs = config.get("num_sample_pairs", 8)  # Use reduced sample pairs from config
    max_new_tokens = config.get("response_max_length", 256)  # Use reduced response length
    temperature = config.get("paired_temperature", 0.8)
    top_p = config.get("paired_top_p", 0.92)
    top_k = config.get("paired_top_k", 25)
    do_sample = config.get("do_sample", True)
    
    # Check for ultra-fast sampling configuration
    if config.get("generation_optimization", {}).get("ultra_fast_sampling_params", False):
        logger.info("Using ultra-fast sampling parameters")
        temperature = 0.5
        top_p = 0.7
        top_k = 10
        max_new_tokens = min(max_new_tokens, 128)  # Limit response length
    
    # Check for very short generation configuration
    if config.get("generation_optimization", {}).get("very_short_generation", False):
        logger.info("Using very short generation")
        max_new_tokens = min(max_new_tokens, 64)  # Drastically limit response length
    
    # Get generation timeout from config or environment variable
    generation_timeout = config.get("generation_timeout", 30)
    if "GENERATION_TIMEOUT_SECONDS" in os.environ:
        try:
            generation_timeout = int(os.environ["GENERATION_TIMEOUT_SECONDS"])
            logger.info(f"Using generation timeout of {generation_timeout}s from environment variable")
        except ValueError:
            logger.warning(f"Invalid GENERATION_TIMEOUT_SECONDS value: {os.environ['GENERATION_TIMEOUT_SECONDS']}")
    
    max_attempts = config.get("max_generation_attempts", 3)
    
    # Generate full prompt
    full_prompt = create_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Function for timeout handling
    def generate_with_timeout(batch_size=4, sampling=True, temp=temperature):
        try:
            # Use a smaller number of sequences per call to prevent memory issues
            curr_batch_size = min(batch_size, 2*num_pairs)
            
            # Clone inputs for this batch to avoid reference issues
            batch_inputs = {
                "input_ids": inputs.input_ids.repeat(curr_batch_size, 1),
                "attention_mask": inputs.attention_mask.repeat(curr_batch_size, 1)
            }
            
            start_time = time.time()
            logger.info(f"Starting generation with batch_size={curr_batch_size}, sampling={sampling}, temp={temp}")
            
            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=sampling,
                    temperature=temp,
                    top_p=top_p if sampling else 1.0,
                    top_k=top_k if sampling else 0,
                    num_return_sequences=curr_batch_size,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f}s")
            
            # Process outputs
            responses = []
            for output in outputs:
                try:
                    response_ids = output[inputs.input_ids.shape[1]:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    responses.append(response_text.strip())
                except Exception as e:
                    logger.error(f"Error decoding response: {str(e)}")
            
            return responses
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return []
    
    # Use ThreadPoolExecutor for timeout handling
    with ThreadPoolExecutor(max_workers=1) as executor:
        responses = []
        attempts = 0
        timeout_count = 0
        
        while len(responses) < num_pairs * 2 and attempts < max_attempts:
            attempts += 1
            logger.info(f"Generation attempt {attempts}/{max_attempts}")
            
            # Decide if we need to switch to simpler generation
            use_simple = (timeout_count >= 1 or 
                         config.get("generation_optimization", {}).get("simple_greedy_if_timeout", False))
            
            try:
                # First try: use normal sampling
                if not use_simple:
                    # Submit generation task with timeout
                    future = executor.submit(generate_with_timeout, 
                                           batch_size=4, 
                                           sampling=True,
                                           temp=temperature)
                    batch_responses = future.result(timeout=generation_timeout)
                    responses.extend(batch_responses)
                    
                # If we didn't get enough responses, try greedy generation
                if len(responses) < num_pairs * 2:
                    # Fallback to greedy generation
                    logger.info("Using greedy generation as fallback")
                    future = executor.submit(generate_with_timeout, 
                                          batch_size=2, 
                                          sampling=False,
                                          temp=1.0)  # Temp doesn't matter for greedy
                    batch_responses = future.result(timeout=generation_timeout)
                    responses.extend(batch_responses)
                    
                # If we still don't have enough, add some simple responses
                if len(responses) < num_pairs * 2:
                    logger.warning("Adding basic responses to complete set")
                    responses.append(f"The answer is {target_answer}.")
                    responses.append("I'll solve this step by step.")
                    responses.append(f"After calculating, I get {target_answer}.")
                    responses.append("Let me think through this problem carefully.")
                
            except TimeoutError:
                logger.warning(f"Generation timed out after {generation_timeout}s")
                timeout_count += 1
                
                # Try with simpler parameters on next attempt
                temperature = max(0.1, temperature * 0.7)
                top_p = max(0.5, top_p * 0.8)
                max_new_tokens = max(32, max_new_tokens // 2)
                
                # Add some basic responses so we don't get stuck
                if timeout_count >= 2:
                    logger.warning("Adding basic responses after timeout")
                    responses.append(f"The answer is {target_answer}.")
                    responses.append("I'll solve this step by step.")
                    responses.append(f"After calculating, I get {target_answer}.")
                    responses.append("Let me think through this problem carefully.")
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                # Add basic responses on error
                responses.append(f"The answer is {target_answer}.")
                responses.append("I'll solve this step by step.")
    
    # Deduplicate responses and handle edge cases
    unique_responses = []
    for resp in responses:
        if resp and resp not in unique_responses:
            unique_responses.append(resp)
    
    logger.info(f"Generated {len(unique_responses)} unique responses after deduplication")
    
    # If we have fewer than 2 responses, create basic responses
    if len(unique_responses) < 2:
        logger.warning("Insufficient responses generated, creating fallback responses")
        unique_responses.append("I'll solve this step by step.")
        unique_responses.append(f"The answer is {target_answer}.")
    
    # Form pairs - ensuring we don't exceed the number of available responses
    pairs = []
    for i in range(min(num_pairs, len(unique_responses) // 2)):
        pair = (unique_responses[2*i], unique_responses[2*i+1])
        pairs.append(pair)
    
    # If we don't have enough pairs, but have some responses, create additional pairs
    while len(pairs) < num_pairs and len(unique_responses) >= 2:
        # Randomly sample two different responses
        idxs = random.sample(range(len(unique_responses)), 2)
        pair = (unique_responses[idxs[0]], unique_responses[idxs[1]])
        pairs.append(pair)
    
    # Return what we have, even if fewer than requested
    logger.info(f"Returning {len(pairs)} response pairs")
    return pairs

def compute_preferences(
    response_a: str,
    response_b: str,
    target_answer: str,
    config: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Compute preference between two responses based on answer correctness.
    
    Args:
        response_a: First response
        response_b: Second response
        target_answer: Reference target answer
        config: Configuration dict
        
    Returns:
        Tuple of (preference, confidence) where preference is 1.0 if response_a is preferred,
        0.0 if response_b is preferred, and 0.5 if there's a tie. Confidence is a value between 0 and 1.
    """
    # Extract answers from responses
    extracted_a = extract_answer(response_a)
    extracted_b = extract_answer(response_b)
    
    # Normalize answers
    normalized_a = normalize_answer(extracted_a) if extracted_a is not None else None
    normalized_b = normalize_answer(extracted_b) if extracted_b is not None else None
    normalized_target = normalize_answer(target_answer)
    
    # Check for step-by-step reasoning
    has_steps_a = has_step_by_step(response_a)
    has_steps_b = has_step_by_step(response_b)
    
    # Determine correctness
    correct_a = normalized_a == normalized_target if normalized_a is not None else False
    correct_b = normalized_b == normalized_target if normalized_b is not None else False
    
    # Base confidence value
    base_confidence = config.get("base_confidence", 0.9)
    step_bonus = config.get("step_bonus", 0.1)
    
    # Compute preferences
    if correct_a and correct_b:
        # Both correct - prefer the one with step-by-step reasoning
        if has_steps_a and not has_steps_b:
            preference = 1.0
            confidence = base_confidence
        elif has_steps_b and not has_steps_a:
            preference = 0.0
            confidence = base_confidence
        else:
            # Tie - both correct with same reasoning status
            preference = 0.5
            confidence = base_confidence / 2  # Lower confidence for ties
    elif correct_a:
        # Only A is correct
        preference = 1.0
        confidence = base_confidence + (step_bonus if has_steps_a else 0)
    elif correct_b:
        # Only B is correct
        preference = 0.0
        confidence = base_confidence + (step_bonus if has_steps_b else 0)
    else:
        # Both wrong
        if has_steps_a and not has_steps_b:
            preference = 1.0
            confidence = 0.6  # Lower confidence since both are wrong
        elif has_steps_b and not has_steps_a:
            preference = 0.0
            confidence = 0.6  # Lower confidence since both are wrong
        else:
            # Tie - both wrong with same reasoning status
            preference = 0.5
            confidence = 0.5  # Lowest confidence
    
    return preference, confidence

def compute_kl_divergence(
    model,
    ref_model,
    input_ids,
    attention_mask,
    response_mask
) -> torch.Tensor:
    """
    Compute KL divergence between current model and reference model.
    
    Args:
        model: Current policy model
        ref_model: Reference model
        input_ids: Input token IDs
        attention_mask: Attention mask
        response_mask: Mask for response tokens
        
    Returns:
        KL divergence tensor
    """
    if ref_model is None:
        return torch.tensor(0.0, device=input_ids.device)
    
    batch_size = input_ids.size(0)
    
    # Get logits from current model (already computed)
    with torch.no_grad():
        policy_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        ref_outputs = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Get log probs
    policy_logits = policy_outputs.logits[:, :-1]  # Shift left
    ref_logits = ref_outputs.logits[:, :-1]  # Shift left
    
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    
    # Get next tokens
    next_tokens = input_ids[:, 1:]  # Shift right
    response_mask_shifted = response_mask[:, 1:]  # Shift right
    
    # Compute KL divergence only for response tokens
    kl_div = torch.zeros(batch_size, device=input_ids.device)
    
    for i in range(batch_size):
        # Extract positions where response_mask is 1
        mask_positions = response_mask_shifted[i].nonzero().squeeze(-1)
        
        if len(mask_positions) == 0:
            continue
        
        # Get policy and ref log probs for these positions
        policy_log_probs_i = torch.gather(
            policy_log_probs[i, mask_positions], 
            1, 
            next_tokens[i, mask_positions].unsqueeze(-1)
        ).squeeze(-1)
        
        ref_log_probs_i = torch.gather(
            ref_log_probs[i, mask_positions], 
            1, 
            next_tokens[i, mask_positions].unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute KL (reverse KL: policy / ref)
        kl_div[i] = (policy_log_probs_i - ref_log_probs_i).mean()
    
    return kl_div

def prepare_batch_for_grpo(
    prompts: List[str], 
    response_pairs: List[Tuple[str, str]],
    preferences: torch.Tensor,
    confidence: torch.Tensor,
    tokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of data for GRPO training.
    
    Args:
        prompts: List of problem prompts
        response_pairs: List of (response_a, response_b) tuples
        preferences: Tensor of preferences (1 for A, 0 for B, 0.5 for tie)
        confidence: Tensor of confidence in preferences
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dict with input tensors for training
    """
    # Extract responses from pairs
    responses_a = [pair[0] for pair in response_pairs]
    responses_b = [pair[1] for pair in response_pairs]
    
    # Create full sequences (prompt + response)
    sequences_a = [create_prompt(p) + " " + r for p, r in zip(prompts, responses_a)]
    sequences_b = [create_prompt(p) + " " + r for p, r in zip(prompts, responses_b)]
    
    # Tokenize
    inputs_a = tokenizer(
        sequences_a, 
        padding=True, 
        truncation=True,
        max_length=max_length, 
        return_tensors="pt"
    )
    
    inputs_b = tokenizer(
        sequences_b, 
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
    
    # Create masks for response tokens
    response_mask_a = torch.zeros_like(inputs_a.attention_mask)
    response_mask_b = torch.zeros_like(inputs_b.attention_mask)
    
    for i in range(len(prompts)):
        response_mask_a[i, prompt_lens[i]:] = inputs_a.attention_mask[i, prompt_lens[i]:]
        response_mask_b[i, prompt_lens[i]:] = inputs_b.attention_mask[i, prompt_lens[i]:]
    
    # Reshape preferences and confidence for batch processing
    preferences_tensor = preferences.view(-1, 1)
    confidence_tensor = confidence.view(-1, 1)
    
    return {
        "input_ids_a": inputs_a.input_ids,
        "attention_mask_a": inputs_a.attention_mask,
        "response_mask_a": response_mask_a,
        "input_ids_b": inputs_b.input_ids,
        "attention_mask_b": inputs_b.attention_mask,
        "response_mask_b": response_mask_b,
        "preferences": preferences_tensor,
        "confidence": confidence_tensor,
        "prompt_lens": prompt_lens
    }

def train_grpo(
    model,
    ref_model,
    tokenizer,
    dataset,
    optimizer,
    lr_scheduler,
    config,
    accelerator
):
    """Train model with GRPO (Generative Reward Policy Optimization)."""
    
    model.train()
    ref_model.eval()
    
    batch_size = config.get("batch_size", 2)  # Reduced batch size
    epochs = config.get("num_epochs", 2)  # Reduced epochs
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 32)  # Increased grad accum
    eval_steps = config.get("eval_steps", 200)
    kl_coef = config.get("kl_penalty_weight", 0.05)
    save_steps = config.get("save_interval", 50)  # More frequent checkpoints
    confidence_scale = config.get("confidence_scale", 1.0)
    max_steps = config.get("max_steps", 2000)  # Reduced max steps
    log_interval = config.get("log_interval", 5)  # More frequent logging
    
    # Additional settings for progressive saving
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    progress_dir = config.get("progress_dir", "./progress")
    
    # Save zero checkpoint before training
    if config.get("save_zero_checkpoint", True):
        logger.info("Saving initial checkpoint before training")
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=0,
            step=0,
            config=config
        )
    
    step = 0
    global_step = 0
    
    all_train_stats = []
    progress_markers = set()
    
    # Create progress tracking file
    progress_file = os.path.join(progress_dir, "training_progress.json")
    os.makedirs(progress_dir, exist_ok=True)
    
    try:
        with open(progress_file, "w") as f:
            json.dump({"status": "started", "step": 0, "epoch": 0}, f)
    except Exception as e:
        logger.warning(f"Could not create progress file: {str(e)}")
    
    # Main training loop with progress tracking
    for epoch in range(epochs):
        epoch_stats = []
        
        # Save epoch start in progress file
        try:
            with open(progress_file, "w") as f:
                json.dump({"status": "running", "step": global_step, "epoch": epoch}, f)
        except Exception as e:
            logger.warning(f"Could not update progress file: {str(e)}")
        
        # Break if we've reached max steps
        if global_step >= max_steps:
            logger.info(f"Reached max steps {max_steps}, stopping training")
            break
        
        for _ in range(0, len(dataset), batch_size):
            # Sample a batch of problems from dataset - with error handling
            try:
                problem_indices = torch.randperm(len(dataset))[:batch_size]
                prompts = [dataset[i]["prompt"] for i in problem_indices]
                target_answers = [dataset[i]["reference"] for i in problem_indices]
                
                # Log the start of response generation to help diagnose any hanging
                generation_start_time = time.time()
                logger.info(f"Starting response generation for batch at step {global_step}")
                
                # Generate response pairs for each problem
                response_pairs_list = []
                for i, (prompt, target) in enumerate(zip(prompts, target_answers)):
                    logger.info(f"Generating responses for problem {i+1}/{len(prompts)}")
                    
                    # Generate with timeout protection
                    start_time = time.time()
                    pairs = generate_response_pairs(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        target_answer=target,
                        config=config
                    )
                    logger.info(f"Generated {len(pairs)} pairs in {time.time() - start_time:.2f}s")
                    response_pairs_list.append(pairs)
                
                logger.info(f"Response generation completed in {time.time() - generation_start_time:.2f}s")
                
                # Compute preferences for each response pair
                all_preferences = []
                all_confidence = []
                
                # Flatten the response pairs for batch processing
                flat_prompts = []
                flat_response_pairs = []
                
                for i, (prompt, pairs) in enumerate(zip(prompts, response_pairs_list)):
                    for pair in pairs:
                        flat_prompts.append(prompt)
                        flat_response_pairs.append(pair)
                        
                        # Compute preference for this pair
                        response_a, response_b = pair
                        preference, confidence = compute_preferences(
                            response_a=response_a,
                            response_b=response_b,
                            target_answer=target_answers[i],
                            config=config
                        )
                        
                        all_preferences.append(preference)
                        all_confidence.append(confidence * confidence_scale)
                
                # Skip batch if we have no response pairs
                if len(flat_response_pairs) == 0:
                    logger.warning("No valid response pairs generated, skipping batch")
                    continue
                
                preferences_tensor = torch.tensor(all_preferences, dtype=torch.float)
                confidence_tensor = torch.tensor(all_confidence, dtype=torch.float)
                
                # Prepare mini-batch for training
                mini_batch = prepare_batch_for_grpo(
                    prompts=flat_prompts,
                    response_pairs=flat_response_pairs,
                    preferences=preferences_tensor,
                    confidence=confidence_tensor,
                    tokenizer=tokenizer,
                    max_length=config.get("max_length", 512)
                )
                
                for key in mini_batch:
                    mini_batch[key] = mini_batch[key].to(accelerator.device)
                
                # Forward pass and compute loss
                # Get loss for response A
                outputs_a = model(
                    input_ids=mini_batch["input_ids_a"],
                    attention_mask=mini_batch["attention_mask_a"]
                )
                logprobs_a = torch.log_softmax(outputs_a.logits, dim=-1)
                
                # Get loss for response B
                outputs_b = model(
                    input_ids=mini_batch["input_ids_b"],
                    attention_mask=mini_batch["attention_mask_b"]
                )
                logprobs_b = torch.log_softmax(outputs_b.logits, dim=-1)
                
                # Compute KL divergence for regularization
                kl_div_a = compute_kl_divergence(
                    model=model,
                    ref_model=ref_model,
                    input_ids=mini_batch["input_ids_a"],
                    attention_mask=mini_batch["attention_mask_a"],
                    response_mask=mini_batch["response_mask_a"]
                )
                
                kl_div_b = compute_kl_divergence(
                    model=model,
                    ref_model=ref_model,
                    input_ids=mini_batch["input_ids_b"],
                    attention_mask=mini_batch["attention_mask_b"],
                    response_mask=mini_batch["response_mask_b"]
                )
                
                kl_div = (kl_div_a + kl_div_b) / 2.0
                
                # Compute policy loss based on preferences with error handling
                log_ratio = torch.zeros_like(mini_batch["preferences"])
                for i in range(mini_batch["input_ids_a"].size(0)):
                    try:
                        response_len_a = mini_batch["response_mask_a"][i].sum().item()
                        response_len_b = mini_batch["response_mask_b"][i].sum().item()
                        
                        # Skip if either response is empty
                        if response_len_a == 0 or response_len_b == 0:
                            continue
                        
                        # Get logprobs for each token in response A
                        resp_logprobs_a = torch.zeros(response_len_a, device=accelerator.device)
                        for t in range(response_len_a):
                            token_pos = mini_batch["prompt_lens"][i] + t
                            if token_pos < mini_batch["input_ids_a"].size(1):
                                next_token = mini_batch["input_ids_a"][i, token_pos]
                                resp_logprobs_a[t] = logprobs_a[i, token_pos-1, next_token]
                        
                        # Get logprobs for each token in response B
                        resp_logprobs_b = torch.zeros(response_len_b, device=accelerator.device)
                        for t in range(response_len_b):
                            token_pos = mini_batch["prompt_lens"][i] + t
                            if token_pos < mini_batch["input_ids_b"].size(1):
                                next_token = mini_batch["input_ids_b"][i, token_pos]
                                resp_logprobs_b[t] = logprobs_b[i, token_pos-1, next_token]
                        
                        # Compute log ratio for pairwise comparison
                        log_ratio[i] = resp_logprobs_a.sum() - resp_logprobs_b.sum()
                    except Exception as e:
                        logger.error(f"Error computing log ratios: {str(e)}")
                        # Set to small neutral value to avoid NaNs
                        log_ratio[i] = torch.tensor(1e-6, device=accelerator.device)
                
                # GRPO loss with preference weighting
                policy_loss = -torch.mean(
                    mini_batch["confidence"] * (2 * mini_batch["preferences"] - 1) * log_ratio
                )
                
                # Total loss with KL regularization
                loss = policy_loss + kl_coef * kl_div
                
                # Backward pass and optimization
                loss = loss / gradient_accumulation_steps
                
                # Ensure loss is a scalar for backward pass
                if loss.dim() > 0:
                    loss = loss.mean()
                    
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss value: {loss.item()}, skipping backward pass")
                    loss = torch.tensor(0.0, device=accelerator.device)
                else:
                    accelerator.backward(loss)
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    accelerator.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Log training stats and update progress file
                    if global_step % log_interval == 0:
                        avg_pref = torch.mean(mini_batch["preferences"]).item()
                        avg_confidence = torch.mean(mini_batch["confidence"]).item()
                        
                        # Ensure kl_div is a scalar
                        kl_div_scalar = kl_div.mean().item() if kl_div.dim() > 0 else kl_div.item()
                        
                        stats = {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": loss.item() * gradient_accumulation_steps,
                            "policy_loss": policy_loss.item(),
                            "kl_div": kl_div_scalar,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "avg_preference": avg_pref,
                            "avg_confidence": avg_confidence
                        }
                        
                        epoch_stats.append(stats)
                        all_train_stats.append(stats)
                        
                        print(f"Step {global_step}: loss={stats['loss']:.4f}, "
                            f"policy_loss={stats['policy_loss']:.4f}, "
                            f"kl_div={stats['kl_div']:.4f}, "
                            f"avg_pref={avg_pref:.4f}, "
                            f"avg_conf={avg_confidence:.4f}")
                        
                        # Update progress file
                        try:
                            with open(progress_file, "w") as f:
                                json.dump({
                                    "status": "running", 
                                    "step": global_step, 
                                    "epoch": epoch,
                                    "stats": stats
                                }, f)
                        except Exception as e:
                            logger.warning(f"Could not update progress file: {str(e)}")
                    
                    # Track progress at fixed intervals (10%, 20%, etc.)
                    progress_pct = int((global_step / max_steps) * 100)
                    for marker in [10, 25, 50, 75]:
                        if progress_pct >= marker and marker not in progress_markers:
                            progress_markers.add(marker)
                            logger.info(f"Training {marker}% complete")
                            # Save intermediate checkpoint at progress markers
                            save_checkpoint(
                                accelerator=accelerator,
                                model=model,
                                tokenizer=tokenizer,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                epoch=epoch,
                                step=global_step,
                                config=config
                            )
                    
                    # Evaluate model periodically
                    if global_step % eval_steps == 0:
                        logger.info(f"Running evaluation at step {global_step}")
                        model.eval()  # Set to eval mode
                        eval_results = evaluate(
                            model=model,
                            tokenizer=tokenizer,
                            eval_dataset=dataset,
                            accelerator=accelerator,
                            num_examples=config.get("eval_examples", 25)
                        )
                        model.train()  # Set back to train mode
                    
                    # Save checkpoint periodically
                    if global_step % save_steps == 0:
                        save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            epoch=epoch,
                            step=global_step,
                            config=config
                        )
                    
                    # Check if we've reached max steps
                    if global_step >= max_steps:
                        logger.info(f"Reached max steps {max_steps}, stopping training")
                        break
                
                step += 1
                
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
                logger.error(f"Traceback: {sys.exc_info()[2]}")
                # Continue to next batch
                continue
        
        # Break early if we've reached max steps
        if global_step >= max_steps:
            break
    
    # Final save at the end of training
    try:
        logger.info("Saving final model checkpoint")
        save_checkpoint(
            accelerator=accelerator,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epochs,
            step=global_step,
            config=config
        )
        
        # Update progress file with completion status
        with open(progress_file, "w") as f:
            json.dump({
                "status": "completed", 
                "step": global_step, 
                "epochs_completed": epochs
            }, f)
    except Exception as e:
        logger.error(f"Error saving final checkpoint: {str(e)}")
    
    # Return all training statistics
    return all_train_stats

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
        # Generate single response for each prompt
        formatted_prompts = [create_prompt(p) for p in prompts]
        responses = []
        
        for prompt in formatted_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract only the generated response (not the prompt)
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
    
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
    parser = argparse.ArgumentParser(description="GRPO training for Qwen on GSM8K")
    
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
        default="data/gsm8k_train.json",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--eval_data", 
        type=str, 
        default="data/gsm8k_val.json",
        help="Path to evaluation data"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="checkpoints/qwen_gsm8k_grpo",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results/qwen_gsm8k_grpo",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=200,
        help="Number of steps between evaluations"
    )
    
    parser.add_argument(
        "--eval_examples", 
        type=int, 
        default=50,
        help="Number of examples to evaluate on"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Load configurations
        logger.info(f"Loading model config from {args.model_config}")
        with open(args.model_config, "r") as f:
            model_config = yaml.safe_load(f)
        
        logger.info(f"Loading GRPO config from {args.grpo_config}")
        with open(args.grpo_config, "r") as f:
            grpo_config = yaml.safe_load(f)
        
        # Merge configurations and override with command line arguments
        config = {**model_config, **grpo_config}
        
        config["train_data"] = args.train_data
        config["eval_data"] = args.eval_data
        config["output_dir"] = args.output_dir
        config["results_dir"] = args.results_dir
        config["eval_steps"] = args.eval_steps
        config["eval_examples"] = args.eval_examples
        config["seed"] = args.seed
        
        # Create progress directory for tracking
        progress_dir = os.path.join(args.output_dir, "progress")
        os.makedirs(progress_dir, exist_ok=True)
        
        # Create a start marker file
        with open(os.path.join(progress_dir, "training_started.json"), "w") as f:
            json.dump({"status": "started", "timestamp": time.time(), "args": vars(args)}, f)
        
        # Initialize accelerator
        logger.info(f"Initializing accelerator with: {config.get('accelerator_config', {})}")
        accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 32),
            mixed_precision=config.get("mixed_precision", "bf16"),
            log_with=config.get("log_with", "wandb") if config.get("wandb_logging", False) else None
        )
        
        # Load tokenizer
        model_path = model_config.get("model_name_or_path", "Qwen/Qwen2.5-3B")
        logger.info(f"Loading tokenizer from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=model_config.get("use_fast_tokenizer", True),
                trust_remote_code=model_config.get("trust_remote_code", True)
            )
            
            # Ensure the tokenizer has pad_token set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token since it was None")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
        
        # Load policy model with quantization
        logger.info(f"Loading policy model from {model_config['model_name_or_path']}")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=model_config.get("peft_config", {}).get("r", 8),
            lora_alpha=model_config.get("peft_config", {}).get("lora_alpha", 16),
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
            
            try:
                # Explicitly use Qwen2.5 3B model to match PPO training
                model_path = model_config.get("model_name_or_path", "Qwen/Qwen2.5-3B")
                logger.info(f"Loading 4-bit quantized model from {model_path}")
                
                policy_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    trust_remote_code=model_config.get("trust_remote_code", True),
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                
                # Prepare model for k-bit training
                policy_model = prepare_model_for_kbit_training(policy_model)
            except Exception as e:
                logger.error(f"Error loading policy model: {str(e)}")
                raise RuntimeError(f"Failed to load 4-bit quantized model: {str(e)}")
        else:
            try:
                # Explicitly use Qwen2.5 3B model to match PPO training
                model_path = model_config.get("model_name_or_path", "Qwen/Qwen2.5-3B")
                logger.info(f"Loading model from {model_path} without quantization")
                
                policy_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=model_config.get("trust_remote_code", True),
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            except Exception as e:
                logger.error(f"Error loading policy model: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Apply LoRA
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.print_trainable_parameters()
        
        # Load reference model (for KL penalty)
        logger.info(f"Loading reference model from {model_config['model_name_or_path']}")
        try:
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_path,  # Use the same path as policy model
                trust_remote_code=model_config.get("trust_remote_code", True),
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            # Freeze reference model
            for param in ref_model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning("Will proceed without reference model for KL penalty")
            ref_model = None
        
        # Load datasets
        train_dataset = GSM8KDataset(args.train_data, tokenizer)
        eval_dataset = GSM8KDataset(args.eval_data, tokenizer)
        
        logger.info(f"Loaded {len(train_dataset)} training examples and {len(eval_dataset)} evaluation examples")
        
        # Create optimizer
        optimizer = AdamW(
            policy_model.parameters(),
            lr=grpo_config.get("learning_rate", 2e-5),
            weight_decay=grpo_config.get("weight_decay", 0.01)
        )
        
        # Create learning rate scheduler
        lr_scheduler = get_scheduler(
            name=grpo_config.get("lr_scheduler_type", "cosine"),
            optimizer=optimizer,
            num_warmup_steps=grpo_config.get("warmup_steps", 50),
            num_training_steps=grpo_config.get("max_steps", 2000)
        )
        
        # Prepare for distributed training
        policy_model, ref_model, optimizer, lr_scheduler = accelerator.prepare(
            policy_model, ref_model, optimizer, lr_scheduler
        )
        
        # Train
        logger.info("Starting GRPO training")
        training_stats = train_grpo(
            policy_model,
            ref_model,
            tokenizer,
            train_dataset,
            optimizer,
            lr_scheduler,
            config,
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
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Save training stats
        with open(os.path.join(args.results_dir, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=2)
        
        # Create completion marker
        with open(os.path.join(progress_dir, "training_completed.json"), "w") as f:
            json.dump({
                "status": "completed", 
                "timestamp": time.time(), 
                "stats": {
                    "final_accuracy": eval_results["accuracy"],
                    "training_steps": len(training_stats)
                }
            }, f)
        
        logger.info("Training and evaluation completed")
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Traceback: {sys.exc_info()[2]}")
        
        # Save error information to file
        try:
            error_dir = os.path.join(args.output_dir, "errors")
            os.makedirs(error_dir, exist_ok=True)
            
            error_file = os.path.join(error_dir, f"error_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(error_file, "w") as f:
                import traceback
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time()
                }
                json.dump(error_info, f, indent=2)
                
            logger.info(f"Error details saved to {error_file}")
        except Exception as save_error:
            logger.error(f"Failed to save error details: {str(save_error)}")
        
        # Re-raise the exception
        raise

if __name__ == "__main__":
    from contextlib import nullcontext
    main() 