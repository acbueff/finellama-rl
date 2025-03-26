#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom GSM8K evaluation script for PPO-finetuned models.
This version is designed to handle PEFT compatibility issues.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class GSM8KDataset(Dataset):
    """Custom dataset for GSM8K."""
    
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

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

def run_evaluation_pipeline(
    model_id: str,
    gsm8k_path: str,
    output_dir: str,
    num_examples: int = 100,
    batch_size: int = 8,
    max_tokens: int = 256,
    use_4bit: bool = False
):
    """Run the complete evaluation pipeline."""
    logger.info(f"Starting evaluation pipeline for {model_id}")
    
    try:
        # Load dataset
        dataset = load_gsm8k_dataset(gsm8k_path, num_examples)
        
        # Extract questions and references
        questions = []
        references = []
        
        for example in dataset:
            questions.append(example.get("question", ""))
            references.append(example.get("answer", ""))
        
        logger.info(f"Extracted {len(questions)} questions and {len(references)} references")
        
        # Setup for model inference
        logger.info(f"Setting up a command to run the actual inference")
        
        # Create inference script
        inference_script = """
import os
import json
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def create_prompt(question: str) -> str:
    return f"Solve the following grade school math problem step by step:\\n{question}\\n\\nSolution:"

def main():
    # Parse arguments
    model_id = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    max_tokens = int(sys.argv[4])
    use_4bit = sys.argv[5].lower() == 'true'
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Load base model and tokenizer
    base_model_id = "SunnyLin/Qwen2.5-7B-DPO-VP"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # Load model
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        adapter_path = model_id
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        adapter_path = model_id
        model = PeftModel.from_pretrained(model, adapter_path)
    
    # Generate predictions
    predictions = []
    for question in questions:
        prompt = create_prompt(question)
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
        predictions.append(response)
    
    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
"""
        
        # Save the inference script
        script_path = "ppo_model_inference.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(inference_script)
        
        # Save questions to a temporary file
        temp_questions_file = "temp_questions.json"
        with open(temp_questions_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False)
        
        # Define output file for predictions
        temp_predictions_file = "temp_predictions.json"
        
        # Run the inference command
        use_4bit_str = "true" if use_4bit else "false"
        cmd = f"apptainer exec --nv --env HF_TOKEN=$HF_TOKEN /proj/berzelius-aiics-real/users/x_anbue/env.sif python {script_path} {model_id} {temp_questions_file} {temp_predictions_file} {max_tokens} {use_4bit_str}"
        
        logger.info(f"Running command: {cmd}")
        
        # Execute the command
        os.system(cmd)
        
        # Load predictions
        try:
            with open(temp_predictions_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            logger.info(f"Loaded {len(predictions)} predictions")
            
            # Evaluate predictions
            metrics = evaluate_answers(predictions, references)
            
            # Save results
            save_results(model_id, metrics, questions, references, predictions, output_dir)
            
            # Cleanup temporary files
            try:
                os.remove(temp_questions_file)
                os.remove(temp_predictions_file)
                os.remove(script_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K dataset")
    
    parser.add_argument("--model_id", type=str, required=True, 
                        help="Path to the model or model_id from HuggingFace")
    parser.add_argument("--gsm8k_path", type=str, required=True,
                        help="Path to the GSM8K dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Number of examples to evaluate on")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        model_id=args.model_id,
        gsm8k_path=args.gsm8k_path,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        use_4bit=args.use_4bit
    )

if __name__ == "__main__":
    main() 