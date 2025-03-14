#!/usr/bin/env python
"""
Standalone script to evaluate Qwen2.5-7B-DPO-VP on GSM8K.
This script completely bypasses any validation issues
by directly implementing the evaluation logic.
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
import re

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone GSM8K Evaluation")
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="SunnyLin/Qwen2.5-7B-DPO-VP",
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
        default=50,
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
    
    # Load the dataset - handle both JSON and JSONL formats
    dataset = []
    
    try:
        # First try loading as a single JSON object
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # If it's a list, use it directly
        if isinstance(data, list):
            dataset = data
        # If it's a dictionary with a 'data' field, use that
        elif isinstance(data, dict) and 'data' in data:
            dataset = data['data']
        # If it's a dictionary with examples, convert to list
        else:
            dataset = [data]
    except json.JSONDecodeError:
        # If that fails, try loading as JSONL (line-by-line JSON objects)
        logger.info("JSON loading failed, trying JSONL format")
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        example = json.loads(line)
                        dataset.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line: {line[:50]}... - {str(e)}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    # Limit number of examples if specified
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset[:num_examples]
        logger.info(f"Using {num_examples} examples for evaluation")
    
    return dataset

def load_model_directly(model_id: str, use_4bit: bool = True):
    """
    Load the model directly.
    """
    logger.info(f"Loading model {model_id} directly")
    
    # Import here to avoid validation at module level
    import transformers
    from transformers import AutoTokenizer
    
    # First, load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
        padding_side='left'  # Set padding to left side for decoder-only models
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    try:
        # Load the model with 4-bit quantization if requested
        if use_4bit:
            from transformers import BitsAndBytesConfig, AutoModelForCausalLM
            
            # Configure 4-bit quantization
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
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
        else:
            from transformers import AutoModelForCausalLM
            
            # Load the model without quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=True
            )
        
        logger.info("Model loaded successfully")
        
        return model, tokenizer
    
    except Exception as e:
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
                do_sample=False,  # Use greedy decoding
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

def extract_answer(text):
    """Extract the answer from the text."""
    # First check for the standard GSM8K format with ####
    if '####' in text:
        match = re.search(r'####\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
    
    # Look for "Therefore" or "Thus" statements with numbers at the end
    therefore_pattern = re.search(r'(?:Therefore|Thus|In conclusion|The answer is)[^.]*?(\d+(?:,\d+)*(?:\.\d+)?)[^.]*\.', text)
    if therefore_pattern:
        return therefore_pattern.group(1).replace(',', '')
    
    # Look for the last equation with an equal sign
    equations = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*=\s*(\$?\s*\d+(?:,\d+)*(?:\.\d+)?)', text)
    if equations:
        # Get the last equation result, remove $ if present
        last_result = equations[-1][1].replace('$', '').replace(',', '').strip()
        return last_result
    
    # Look for dollar amounts
    money_pattern = re.search(r'\$\s*(\d+(?:,\d+)*(?:\.\d+)?)', text)
    if money_pattern:
        return money_pattern.group(1).replace(',', '')
    
    # Split by sentences and find the last sentence with a number
    sentences = re.split(r'[.!?]+', text)
    for sentence in reversed(sentences):
        number_pattern = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', sentence)
        if number_pattern:
            return number_pattern.group(1).replace(',', '')
    
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
        
        # Extract questions and references
        questions = []
        references = []
        
        for example in dataset:
            questions.append(example.get("question", ""))
            references.append(example.get("answer", ""))
        
        logger.info(f"Extracted {len(questions)} questions and {len(references)} references")
        
        # Load model with direct approach
        model, tokenizer = load_model_directly(args.model_id, args.use_4bit)
        
        # Generate answers
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