#!/usr/bin/env python
"""
Standalone script to evaluate FineMath-Llama-3B on GSM8K.
This script completely bypasses the rope_scaling validation by directly modifying
the model's configuration before loading.
"""

import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
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
        "--num_examples", 
        type=int, 
        default=-1,
        help="Number of examples to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Whether to use 4-bit quantization"
    )
    
    return parser.parse_args()

def load_gsm8k_dataset(data_path: str, num_examples: int = -1):
    """Load the GSM8K dataset."""
    logger.info(f"Loading GSM8K dataset from {data_path}")
    
    if os.path.exists(data_path):
        # Load from local file
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

def load_model_with_fixed_config(model_id: str, use_4bit: bool = True):
    """
    Load the model with a fixed configuration that bypasses rope_scaling validation.
    """
    logger.info(f"Loading model {model_id} with fixed configuration")
    
    # First, load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load the config
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    # Fix rope_scaling by completely replacing it
    if hasattr(config, "rope_scaling"):
        logger.info(f"Original rope_scaling: {config.rope_scaling}")
        # Replace with a simple compatible version
        config.rope_scaling = {"type": "linear", "factor": 1.0}
        logger.info(f"Fixed rope_scaling: {config.rope_scaling}")
    
    # Monkey patch the validation method to be a no-op
    from transformers.models.llama.configuration_llama import LlamaConfig
    original_validation = LlamaConfig._rope_scaling_validation
    LlamaConfig._rope_scaling_validation = lambda self: None
    logger.info("Disabled rope_scaling validation")
    
    # Load the model with our fixed config
    try:
        if use_4bit:
            # Use 4-bit quantization to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
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
        
        # Restore the original validation method
        LlamaConfig._rope_scaling_validation = original_validation
        
        return model, tokenizer
        
    except Exception as e:
        # Restore the original validation method
        LlamaConfig._rope_scaling_validation = original_validation
        logger.error(f"Failed to load model: {str(e)}")
        raise

def create_prompt(question: str) -> str:
    """Create a prompt for GSM8K evaluation."""
    return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def generate_answers(
    model, 
    tokenizer, 
    questions: List[str], 
    batch_size: int = 8, 
    max_tokens: int = 256
) -> List[str]:
    """Generate answers for the given questions."""
    logger.info(f"Generating answers for {len(questions)} questions")
    
    # Create prompts
    prompts = [create_prompt(q) for q in questions]
    
    # Generate in batches
    all_outputs = []
    
    # Put model in evaluation mode
    model.eval()
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            # Get only the generated part (not the prompt)
            prompt_length = len(tokenizer.encode(batch[j], add_special_tokens=False))
            generated_tokens = output[prompt_length:]
            
            # Decode
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_outputs.append(text)
        
        logger.info(f"Processed {min(i+batch_size, len(prompts))}/{len(prompts)} examples")
    
    return all_outputs

def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from the generated text."""
    if "####" in text:
        parts = text.split("####")
        if len(parts) > 1:
            answer = parts[-1].strip()
            # Clean up the answer (keep only digits, decimal points, and minus signs)
            answer = ''.join(c for c in answer if c.isdigit() or c == '.' or c == '-')
            return answer
    return None

def evaluate_answers(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """Evaluate the predictions against the references."""
    logger.info("Evaluating predictions")
    
    # Extract answers
    pred_answers = [extract_answer(pred) for pred in predictions]
    ref_answers = [extract_answer(ref) for ref in references]
    
    # Count correct answers
    correct = 0
    for pred, ref in zip(pred_answers, ref_answers):
        if pred is not None and ref is not None and pred == ref:
            correct += 1
    
    # Calculate accuracy
    total = len(references)
    accuracy = correct / total if total > 0 else 0
    
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def save_results(
    output_dir: str,
    questions: List[str],
    predictions: List[str],
    references: List[str],
    metrics: Dict[str, Any]
):
    """Save the evaluation results."""
    logger.info(f"Saving results to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "gsm8k_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "gsm8k_predictions.jsonl")
    with open(predictions_path, 'w') as f:
        for i, (q, p, r) in enumerate(zip(questions, predictions, references)):
            example = {
                "id": i,
                "question": q,
                "prediction": p,
                "reference": r
            }
            f.write(json.dumps(example) + "\n")
    
    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        summary = {
            "dataset": "gsm8k",
            "num_examples": len(questions),
            "metrics": metrics
        }
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load dataset
        dataset = load_gsm8k_dataset(args.gsm8k_path, args.num_examples)
        
        # Load model with fixed config
        model, tokenizer = load_model_with_fixed_config(args.model_id, args.use_4bit)
        
        # Generate answers
        questions = dataset["question"]
        references = dataset["answer"]
        predictions = generate_answers(
            model, 
            tokenizer, 
            questions, 
            batch_size=args.batch_size,
            max_tokens=args.max_tokens
        )
        
        # Evaluate answers
        metrics = evaluate_answers(predictions, references)
        
        # Save results
        save_results(
            args.output_dir,
            questions,
            predictions,
            references,
            metrics
        )
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 