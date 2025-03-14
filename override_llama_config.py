#!/usr/bin/env python
"""
Override script to handle Llama3's rope_scaling issue for the GSM8K evaluation.
This script directly loads and evaluates the model on GSM8K, bypassing the problematic validation.
"""

import os
import sys
import json
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GSM8K Evaluation with rope_scaling fix")
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="HuggingFaceTB/FineMath-Llama-3B",
        help="Path to model or Hugging Face model ID"
    )
    parser.add_argument(
        "--gsm8k_path", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json",
        help="Path to GSM8K test data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=-1,
        help="Number of examples to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Whether to use 4-bit quantization"
    )
    
    return parser.parse_args()

def load_gsm8k_dataset(data_path, num_examples=-1):
    """Load the GSM8K dataset from a JSON file."""
    logger.info(f"Loading GSM8K dataset from {data_path}")
    
    # Load the dataset based on file path
    if os.path.exists(data_path):
        dataset = load_dataset('json', data_files=data_path, split='train')
    else:
        # Try loading from HuggingFace datasets
        dataset = load_dataset('gsm8k', 'main', split='test')
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Limit number of examples if specified
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))
        logger.info(f"Using {num_examples} examples for evaluation")
    
    return dataset

def load_model_with_fixed_rope_scaling(model_id, use_4bit=True):
    """
    Load the model with a fixed rope_scaling configuration.
    """
    logger.info(f"Loading model {model_id} with fixed rope_scaling")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model config
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    # Fix rope_scaling
    if hasattr(config, "rope_scaling"):
        logger.info(f"Original rope_scaling: {config.rope_scaling}")
        
        # Replace rope_scaling with a simple compatible version
        config.rope_scaling = {"type": "linear", "factor": 1.0}
        
        logger.info(f"Fixed rope_scaling: {config.rope_scaling}")
    
    # Create 4-bit configuration if needed
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )
    else:
        # Load with full precision
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True
        )
    
    logger.info("Model loaded successfully")
    return model, tokenizer

def create_gsm8k_prompt(problem):
    """Create a prompt for the GSM8K problem."""
    prompt_template = """Solve the following grade school math problem step by step:
{problem}

Solution:"""
    
    return prompt_template.format(problem=problem)

def evaluate_gsm8k(model, tokenizer, dataset, batch_size=4, max_tokens=512):
    """
    Evaluate the model on GSM8K.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        dataset: The GSM8K dataset
        batch_size: Batch size
        max_tokens: Maximum tokens to generate
        
    Returns:
        List of predictions, metrics
    """
    logger.info(f"Evaluating model on {len(dataset)} GSM8K problems")
    
    problems = dataset["question"]
    references = dataset["answer"]
    
    # Create prompts
    prompts = [create_gsm8k_prompt(problem) for problem in problems]
    
    # Generation configuration
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate responses in batches
    all_responses = []
    model.eval()
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode responses
        for j, output in enumerate(outputs):
            full_response = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract just the generated part (remove the prompt)
            prompt_len = len(batch_prompts[j])
            if len(full_response) > prompt_len:
                response = full_response[prompt_len:].strip()
            else:
                response = full_response
                
            all_responses.append(response)
            
        logger.info(f"Processed {min(i+batch_size, len(prompts))}/{len(prompts)} examples")
    
    # Evaluate results
    metrics = compute_metrics(all_responses, references, problems)
    
    return all_responses, metrics

def compute_metrics(predictions, references, problems):
    """
    Compute evaluation metrics for GSM8K.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        problems: List of problems
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing metrics")
    
    # Extract final answers from predictions and references
    def extract_final_answer(text):
        if "####" in text:
            # Extract number after ####
            parts = text.split("####")
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Clean up the answer
                answer = ''.join(c for c in answer if c.isdigit() or c == '.' or c == '-')
                return answer
        return None
    
    pred_answers = [extract_final_answer(pred) for pred in predictions]
    ref_answers = [extract_final_answer(ref) for ref in references]
    
    # Count correct predictions
    correct = 0
    for pred, ref in zip(pred_answers, ref_answers):
        if pred is not None and ref is not None and pred == ref:
            correct += 1
    
    # Calculate accuracy
    accuracy = correct / len(references) if len(references) > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(references),
    }
    
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{len(references)})")
    
    return metrics

def save_results(predictions, metrics, output_dir, problems, references):
    """
    Save evaluation results to disk.
    
    Args:
        predictions: List of model predictions
        metrics: Dictionary of metrics
        output_dir: Output directory
        problems: List of problems
        references: List of reference answers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "gsm8k_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "gsm8k_predictions.jsonl")
    with open(predictions_path, 'w') as f:
        for i, (problem, pred, ref) in enumerate(zip(problems, predictions, references)):
            example = {
                "id": i,
                "problem": problem,
                "prediction": pred,
                "reference": ref
            }
            f.write(json.dumps(example) + "\n")
    
    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        summary = {
            "dataset": "gsm8k",
            "num_examples": len(problems),
            "metrics": metrics,
            "predictions_path": predictions_path
        }
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load GSM8K dataset
    dataset = load_gsm8k_dataset(args.gsm8k_path, args.num_examples)
    
    # Load model with fixed rope_scaling
    model, tokenizer = load_model_with_fixed_rope_scaling(
        args.model_id, 
        use_4bit=args.use_4bit
    )
    
    # Evaluate model on GSM8K
    predictions, metrics = evaluate_gsm8k(
        model, 
        tokenizer, 
        dataset, 
        batch_size=args.batch_size,
        max_tokens=args.max_tokens
    )
    
    # Save results
    save_results(
        predictions, 
        metrics, 
        args.output_dir, 
        dataset["question"], 
        dataset["answer"]
    )
    
    logger.info("Evaluation complete")

if __name__ == "__main__":
    main() 