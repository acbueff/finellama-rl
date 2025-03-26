#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct evaluation script for PPO-finetuned model on GSM8K.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import torch
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_gsm8k_dataset(data_path: str, num_examples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K dataset from a JSON file."""
    logger.info(f"Loading GSM8K dataset from {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except json.JSONDecodeError:
        logger.info("JSON loading failed, trying JSONL format")
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    if not isinstance(dataset, list):
        # If dataset is a dict with a 'data' key (common format)
        if isinstance(dataset, dict) and 'data' in dataset:
            dataset = dataset['data']
        else:
            raise ValueError(f"Unsupported dataset format: {type(dataset)}")
    
    logger.info(f"Loaded {len(dataset)} examples")
    
    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset[:num_examples]
        logger.info(f"Using {len(dataset)} examples for evaluation")
    
    return dataset

def extract_answer(text: str) -> str:
    """Extract the final answer from the generated text."""
    # Look for patterns like "The answer is X" or "X is the answer"
    patterns = [
        r"(?:the\s+)?(?:final\s+)?(?:answer\s+is\s+)[^0-9]*([0-9,]+(?:\.[0-9]+)?)",
        r"(?:=\s*)([0-9,]+(?:\.[0-9]+)?)",
        r"(?:we\s+get\s+)[^0-9]*([0-9,]+(?:\.[0-9]+)?)",
        r"(?:result\s+is\s+)[^0-9]*([0-9,]+(?:\.[0-9]+)?)",
        r"(?:^|[\s\n])([0-9,]+(?:\.[0-9]+)?)(?:\s*$)",  # Only numbers at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "")
    
    # If no patterns match, try to find the last number in the text
    numbers = re.findall(r"[0-9,]+(?:\.[0-9]+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return ""

def normalize_answer(answer: str) -> str:
    """Normalize an answer to facilitate comparison."""
    # Remove commas
    answer = answer.replace(",", "")
    
    # Handle dollar signs
    if "$" in answer:
        answer = answer.replace("$", "")
    
    # Handle percentages
    if "%" in answer:
        answer = answer.replace("%", "")
    
    # Try to convert to float
    try:
        answer_float = float(answer)
        # Handle special cases
        if answer_float == int(answer_float):
            return str(int(answer_float))
        return str(answer_float)
    except ValueError:
        return answer.strip()

def evaluate_answers(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Evaluate model predictions against reference answers."""
    if len(predictions) != len(references):
        raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match references ({len(references)})")
    
    correct = 0
    
    # For each prediction and reference
    for pred, ref in zip(predictions, references):
        # Extract answers
        pred_answer = extract_answer(pred)
        ref_answer = extract_answer(ref)
        
        # Normalize answers
        pred_answer = normalize_answer(pred_answer)
        ref_answer = normalize_answer(ref_answer)
        
        # Compare
        if pred_answer == ref_answer:
            correct += 1
    
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(predictions)
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    return metrics

def save_results(
    model_id: str,
    metrics: Dict[str, float],
    questions: List[str],
    references: List[str],
    predictions: List[str],
    output_dir: str
) -> None:
    """Save evaluation results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a name for the results file based on the model name
    model_name = os.path.basename(model_id)
    if not model_name:
        model_name = model_id.replace("/", "_")
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
    
    # Prepare results object
    results = {
        "model_id": model_id,
        "timestamp": timestamp,
        "metrics": metrics,
        "examples": []
    }
    
    # Add individual examples
    for q, r, p in zip(questions, references, predictions):
        pred_answer = extract_answer(p)
        ref_answer = extract_answer(r)
        is_correct = normalize_answer(pred_answer) == normalize_answer(ref_answer)
        
        results["examples"].append({
            "question": q,
            "reference": r,
            "prediction": p,
            "extracted_reference": ref_answer,
            "extracted_prediction": pred_answer,
            "is_correct": is_correct
        })
    
    # Save to file
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")

def create_prompt(question: str) -> str:
    """Create a prompt for GSM8K evaluation."""
    return f"Solve the following grade school math problem step by step:\n{question}\n\nSolution:"

def load_model_and_tokenizer(base_model_id: str, adapter_path: str, use_4bit: bool = False):
    """Load the base model with the LoRA adapter."""
    logger.info(f"Loading base model {base_model_id} and adapter {adapter_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        # Load model with quantization if requested
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
        else:
            # Load base model without quantization
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        
        logger.info("Base model loaded successfully")
        
        # Apply LoRA adapter
        try:
            from peft import get_peft_model_state_dict
            model = PeftModel.from_pretrained(model, adapter_path)
            logger.info("LoRA adapter applied successfully")
        except Exception as e:
            logger.error(f"Error applying LoRA adapter: {str(e)}")
            raise
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_answers(
    model, 
    tokenizer, 
    questions: List[str], 
    max_tokens: int = 256,
    batch_size: int = 1
) -> List[str]:
    """Generate answers for the given questions."""
    logger.info(f"Generating answers for {len(questions)} questions")
    
    # Create prompts
    prompts = [create_prompt(q) for q in questions]
    
    responses = []
    
    # Process questions one by one (for simplicity)
    for prompt in tqdm(prompts, desc="Generating answers"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K dataset")
    
    parser.add_argument("--base_model_id", type=str, default="SunnyLin/Qwen2.5-7B-DPO-VP",
                        help="Base model ID from HuggingFace")
    parser.add_argument("--adapter_path", type=str, required=True, 
                        help="Path to the LoRA adapter")
    parser.add_argument("--gsm8k_path", type=str, required=True,
                        help="Path to the GSM8K dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--num_examples", type=int, default=50,
                        help="Number of examples to evaluate on")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
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
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.base_model_id, args.adapter_path, args.use_4bit)
        
        # Generate answers
        predictions = generate_answers(
            model, 
            tokenizer, 
            questions,
            max_tokens=args.max_tokens
        )
        
        # Evaluate answers
        metrics = evaluate_answers(predictions, references)
        
        # Save results
        save_results(
            args.adapter_path,
            metrics,
            questions,
            references,
            predictions,
            args.output_dir
        )
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 