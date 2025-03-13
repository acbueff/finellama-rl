#!/usr/bin/env python
"""
Fixed evaluation script for GSM8K dataset.
This script correctly handles the JSONL format and includes the rope_scaling patch.
"""

import os
import sys
import json
import argparse
import logging
import torch
import csv
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Apply the rope_scaling patch
try:
    from patch_llama_config import patch_llama_config
    patch_success = patch_llama_config()
    logger.info(f"rope_scaling patch applied: {patch_success}")
except ImportError:
    logger.warning("patch_llama_config not found, rope_scaling patch not applied")
    patch_success = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the GSM8K dataset")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="HuggingFaceTB/FineMath-Llama-3B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k",
        help="Directory to save results"
    )
    parser.add_argument(
        "--gsm8k_path", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json",
        help="Path to GSM8K test data (in JSONL format)"
    )
    parser.add_argument(
        "--num_examples", 
        type=int, 
        default=10,
        help="Number of examples to evaluate (default: 10, -1 for all)"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true",
        help="Use 4-bit quantization to save memory"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=256,
        help="Maximum tokens to generate for each prompt"
    )
    return parser.parse_args()

def load_model_and_tokenizer(model_id, use_4bit=True):
    """Load model and tokenizer with fixed rope_scaling."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Load config
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use 4-bit quantization to save memory
    if use_4bit:
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
    
    return model, tokenizer

def load_gsm8k_data(gsm8k_path, num_examples=-1):
    """Load GSM8K data from JSONL file."""
    examples = []
    
    # Read examples line by line (JSONL format)
    with open(gsm8k_path, 'r') as f:
        for i, line in enumerate(f):
            if num_examples > 0 and i >= num_examples:
                break
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON on line {i+1}: {e}")
    
    logger.info(f"Loaded {len(examples)} examples from {gsm8k_path}")
    return examples

def create_prompt(question):
    """Create a prompt for a GSM8K question."""
    return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def extract_answer(text):
    """Extract the numerical answer from the generated text."""
    # Look for the pattern #### X
    import re
    match = re.search(r'####\s*([0-9\.]+)', text)
    if match:
        return match.group(1).strip()
    return None

def evaluate_model(model, tokenizer, examples, max_tokens=256):
    """Evaluate the model on the examples and return results."""
    results = []
    correct = 0
    
    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        question = example.get("question", "")
        reference_answer = example.get("answer", "")
        
        # Get the ground truth answer
        ground_truth = extract_answer(reference_answer)
        if not ground_truth:
            logger.warning(f"Could not extract answer from example {i}")
            continue
        
        # Create prompt
        prompt = create_prompt(question)
        
        # Generate - Fix issue by using proper input batching and ensuring tensors have correct shapes
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            try:
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,  # Changed to True as temperature is set
                    pad_token_id=tokenizer.pad_token_id
                )
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    logger.error(f"Matrix shape mismatch error for example {i}. Trying with smaller batch...")
                    # Try with a different approach - single token at a time generation
                    generated_ids = input_ids.clone()
                    for _ in range(max_tokens):
                        with torch.no_grad():
                            outputs = model(generated_ids, attention_mask=torch.ones_like(generated_ids).to(model.device))
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                    output = generated_ids
                else:
                    # Re-raise for unexpected errors
                    raise
        
        # Decode
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract answer
        generated_answer = extract_answer(generated_text)
        
        # Check if correct
        is_correct = False
        if generated_answer and ground_truth:
            is_correct = generated_answer.strip() == ground_truth.strip()
            if is_correct:
                correct += 1
        
        # Save result
        result = {
            "question": question,
            "ground_truth_full": reference_answer,
            "ground_truth": ground_truth,
            "generated": generated_text,
            "generated_answer": generated_answer,
            "correct": is_correct
        }
        results.append(result)
        
        # Log progress
        if (i + 1) % 5 == 0:
            logger.info(f"Evaluated {i+1}/{len(examples)} examples. Accuracy so far: {correct/(i+1):.4f}")
    
    # Calculate accuracy
    accuracy = correct / len(examples) if examples else 0
    logger.info(f"Final accuracy: {accuracy:.4f} ({correct}/{len(examples)})")
    
    return results, accuracy

def save_results(results, accuracy, output_dir):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, "gsm8k_predictions.jsonl")
    with open(predictions_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "gsm8k_metrics.json")
    metrics = {
        "accuracy": accuracy,
        "num_examples": len(results),
        "num_correct": sum(1 for r in results if r["correct"])
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    summary = {
        "gsm8k": metrics
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    return metrics

def main():
    args = parse_args()
    
    # Set environment variables
    os.environ["HF_DATASETS_CACHE"] = "/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
    os.environ["HF_HOME"] = "/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id, use_4bit=args.use_4bit)
    
    # Load GSM8K data
    logger.info(f"Loading GSM8K data from {args.gsm8k_path}")
    examples = load_gsm8k_data(args.gsm8k_path, num_examples=args.num_examples)
    
    # Evaluate model
    logger.info("Evaluating model")
    results, accuracy = evaluate_model(model, tokenizer, examples, max_tokens=args.max_tokens)
    
    # Save results
    metrics = save_results(results, accuracy, args.output_dir)
    
    logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main() 