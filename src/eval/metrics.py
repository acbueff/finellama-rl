"""
Metrics for evaluating mathematical reasoning capabilities.
This module provides various metrics for assessing model performance on mathematical tasks.
"""

import re
import string
import logging
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any, Set

# Try importing optional dependencies
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

def normalize_answer(text: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Remove whitespace and convert to lowercase
    text = re.sub(r'\s+', '', text.lower())
    
    # Remove punctuation except decimal points in numbers
    text = re.sub(r'[^\w\s\d.]', '', text)
    
    # Remove units and common text patterns
    text = re.sub(r'(?:units|unit|dollars|km|kg|m|s|\$|£|€|=|is)', '', text)
    
    # Replace common fractions with decimal form
    fraction_map = {
        '1/2': '0.5', '1/3': '0.333', '2/3': '0.667',
        '1/4': '0.25', '3/4': '0.75', '1/5': '0.2'
    }
    for frac, dec in fraction_map.items():
        text = text.replace(frac, dec)
    
    # Handle percentages
    text = re.sub(r'(\d+)%', r'\1/100', text)
    
    return text

def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from a solution text.
    
    Args:
        text: Solution text
        
    Returns:
        Extracted final answer
    """
    # Try different patterns for extracting final answers
    patterns = [
        r"(?:final answer|answer|result)(?:\s+is)?(?:\s*:)?\s*([^\.]+)",
        r"(?:therefore|thus|so|hence)(?:\s+,)?\s+([^\.]+)",
        r"(?:=\s*)([^\.]+?)(?:\s*$|\s*\.)"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return the last match
    
    # If no match found, try to get the last line or statement
    lines = text.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if non_empty_lines:
        return non_empty_lines[-1]
    
    # If all else fails, return the last 50 characters
    return text[-50:].strip() if text else ""

def exact_match_score(prediction: str, reference: str) -> float:
    """
    Compute exact match score between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_answer = normalize_answer(extract_final_answer(prediction))
    ref_answer = normalize_answer(extract_final_answer(reference))
    
    return float(pred_answer == ref_answer)

def f1_score(prediction: str, reference: str) -> float:
    """
    Compute F1 score between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = set(normalize_answer(extract_final_answer(prediction)).split())
    ref_tokens = set(normalize_answer(extract_final_answer(reference)).split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0 if pred_tokens or ref_tokens else 1.0
    
    common_tokens = pred_tokens.intersection(ref_tokens)
    
    # Compute precision and recall
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    # Compute F1
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * (precision * recall) / (precision + recall)

def normalized_edit_distance(prediction: str, reference: str) -> float:
    """
    Compute normalized Levenshtein distance between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        Normalized edit distance between 0.0 (identical) and 1.0 (completely different)
    """
    pred_answer = normalize_answer(extract_final_answer(prediction))
    ref_answer = normalize_answer(extract_final_answer(reference))
    
    # Compute Levenshtein distance
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        
        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Normalize by the length of the longer string
    max_len = max(len(pred_answer), len(ref_answer))
    if max_len == 0:
        return 0.0
    
    distance = levenshtein(pred_answer, ref_answer)
    normalized_distance = distance / max_len
    
    # Return similarity rather than distance (1.0 is best)
    return 1.0 - normalized_distance

def mathematical_equivalence(prediction: str, reference: str) -> float:
    """
    Check if the predicted answer is mathematically equivalent to the reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        1.0 if equivalent, 0.0 otherwise
    """
    if not SYMPY_AVAILABLE:
        logger.warning("SymPy not available, mathematical equivalence check skipped")
        return float(normalize_answer(extract_final_answer(prediction)) == 
                     normalize_answer(extract_final_answer(reference)))
    
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)
    
    # Try to parse the answers as mathematical expressions
    try:
        # Extract any numbers from the text
        pred_nums = re.findall(r'[-+]?\d*\.?\d+', pred_answer)
        ref_nums = re.findall(r'[-+]?\d*\.?\d+', ref_answer)
        
        # Check numeric equivalence first
        if pred_nums and ref_nums:
            try:
                # Compare the first number found in each
                pred_val = float(pred_nums[0])
                ref_val = float(ref_nums[0])
                if abs(pred_val - ref_val) < 1e-6:
                    return 1.0
            except (ValueError, TypeError):
                pass
                
        # Try parsing as symbolic expressions
        pred_expr = parse_expr(pred_answer.replace('=', ''), evaluate=True)
        ref_expr = parse_expr(ref_answer.replace('=', ''), evaluate=True)
        
        # Check equivalence
        if pred_expr == ref_expr:
            return 1.0
            
        # Try simplifying the difference
        diff = sympy.simplify(pred_expr - ref_expr)
        if diff == 0:
            return 1.0
            
    except (SympifyError, TypeError, ValueError, Exception):
        # Fall back to normalized string comparison
        pass
        
    # Fall back to normalized string comparison
    return float(normalize_answer(pred_answer) == normalize_answer(ref_answer))

def rouge_score(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE score between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        Dictionary of ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        logger.warning("Rouge not available, rouge_score skipped")
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }
    
    rouge = Rouge()
    
    try:
        # Ensure non-empty strings
        pred = prediction.strip() or " "
        ref = reference.strip() or " "
        
        scores = rouge.get_scores(pred, ref)[0]
        
        # Extract F1 scores
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }
    except Exception as e:
        logger.warning(f"Error computing ROUGE score: {e}")
        return {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }

def extract_numbers(text: str) -> List[float]:
    """
    Extract numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    number_pattern = r'[-+]?\d*\.?\d+'
    number_matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in number_matches:
        try:
            numbers.append(float(match))
        except (ValueError, TypeError):
            pass
            
    return numbers

def calculate_numeric_error(prediction: str, reference: str) -> Optional[float]:
    """
    Calculate the numeric error between prediction and reference.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        Numeric error, or None if not applicable
    """
    # Extract the final answers
    pred_answer = extract_final_answer(prediction)
    ref_answer = extract_final_answer(reference)
    
    # Extract numbers
    pred_numbers = extract_numbers(pred_answer)
    ref_numbers = extract_numbers(ref_answer)
    
    if not pred_numbers or not ref_numbers:
        return None
        
    # Use the first number as the answer
    pred_val = pred_numbers[0]
    ref_val = ref_numbers[0]
    
    # Calculate relative error
    if abs(ref_val) < 1e-10:
        return abs(pred_val - ref_val)  # Absolute error for values near zero
    else:
        return abs((pred_val - ref_val) / ref_val)  # Relative error

def mathematical_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute the accuracy of answers that are mathematically equivalent to references.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if len(predictions) != len(references):
        logger.warning(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        
    correct = sum(mathematical_equivalence(pred, ref) 
                 for pred, ref in zip(predictions, references))
    
    return correct / len(predictions) if predictions else 0.0

def numeric_relative_error(predictions: List[str], references: List[str]) -> Optional[float]:
    """
    Compute the average relative error for numerical answers.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        
    Returns:
        Average relative error, or None if no numeric pairs found
    """
    errors = []
    
    for pred, ref in zip(predictions, references):
        error = calculate_numeric_error(pred, ref)
        if error is not None:
            errors.append(error)
            
    return sum(errors) / len(errors) if errors else None

def step_by_step_reasoning_score(prediction: str, reference: str) -> float:
    """
    Score the step-by-step reasoning quality.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
        
    Returns:
        Reasoning quality score between 0.0 and 1.0
    """
    # Extract reasoning steps
    def extract_steps(text: str) -> List[str]:
        # Split by numbered steps
        step_pattern = r'(?:^|\n)(?:\d+\.\s*)([^\n]+)'
        steps = re.findall(step_pattern, text)
        
        # If no numbered steps, try splitting by lines
        if not steps:
            steps = [line.strip() for line in text.split('\n') if line.strip()]
            
        return steps
        
    pred_steps = extract_steps(prediction)
    ref_steps = extract_steps(reference)
    
    if not pred_steps or not ref_steps:
        return 0.0
        
    # Compute overlap between steps
    pred_step_text = ' '.join(pred_steps)
    ref_step_text = ' '.join(ref_steps)
    
    # Use various similarity metrics
    # 1. Number of steps similarity
    steps_ratio = min(len(pred_steps) / len(ref_steps), 1.0)
    
    # 2. Word overlap similarity
    pred_words = set(pred_step_text.lower().split())
    ref_words = set(ref_step_text.lower().split())
    
    if not pred_words or not ref_words:
        word_overlap = 0.0
    else:
        common_words = pred_words.intersection(ref_words)
        word_overlap = len(common_words) / len(ref_words)
    
    # 3. Keyword detection (math operators, etc.)
    math_keywords = {'+', '-', '*', '/', '=', '<', '>', '≤', '≥', 'let', 'solve', 'equation'}
    
    pred_keywords = set()
    for step in pred_steps:
        for keyword in math_keywords:
            if keyword in step:
                pred_keywords.add(keyword)
                
    ref_keywords = set()
    for step in ref_steps:
        for keyword in math_keywords:
            if keyword in step:
                ref_keywords.add(keyword)
                
    if not ref_keywords:
        keyword_overlap = 0.0
    else:
        common_keywords = pred_keywords.intersection(ref_keywords)
        keyword_overlap = len(common_keywords) / len(ref_keywords)
    
    # Combine scores with weighted average
    reasoning_score = (0.2 * steps_ratio) + (0.5 * word_overlap) + (0.3 * keyword_overlap)
    
    return reasoning_score

def count_reasoning_steps(prediction: str) -> int:
    """
    Count the number of identifiable reasoning steps in the prediction.
    
    Args:
        prediction: Predicted answer
        
    Returns:
        Number of reasoning steps
    """
    # Check for numbered steps
    numbered_steps = re.findall(r'(?:^|\n)(?:\d+\.\s*)', prediction)
    if numbered_steps:
        return len(numbered_steps)
    
    # Check for equations with equal signs
    equations = re.findall(r'[^=]+=[^=]+', prediction)
    
    # Check for line breaks that might indicate steps
    lines = [line.strip() for line in prediction.split('\n') if line.strip()]
    
    # Return the max of the equation count and line count
    return max(len(equations), len(lines))

def compute_metrics(
    predictions: List[str],
    references: Optional[List[str]] = None,
    problems: Optional[List[str]] = None,
    metrics_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute all specified metrics for the predictions.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers (optional)
        problems: List of problem statements (optional)
        metrics_list: List of metric names to compute
        
    Returns:
        Dictionary of metric names to scores
    """
    if metrics_list is None:
        # Default metrics
        metrics_list = ["exact_match", "f1_score", "normalized_edit_distance"]
        
    # Add metrics that don't require references
    results = {
        "num_samples": len(predictions),
        "avg_response_length": sum(len(p) for p in predictions) / len(predictions),
        "avg_reasoning_steps": sum(count_reasoning_steps(p) for p in predictions) / len(predictions)
    }
    
    # Handle empty predictions case
    if not predictions:
        logger.warning("Empty predictions list")
        for metric in metrics_list:
            results[metric] = 0.0
        return results
    
    # Add metrics that require references, if available
    if references:
        if len(predictions) != len(references):
            logger.warning(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        # Compute all requested metrics
        for metric in metrics_list:
            if metric == "exact_match":
                results["exact_match"] = sum(exact_match_score(p, r) for p, r in zip(predictions, references)) / len(predictions)
            elif metric == "f1_score":
                results["f1_score"] = sum(f1_score(p, r) for p, r in zip(predictions, references)) / len(predictions)
            elif metric == "normalized_edit_distance":
                results["normalized_edit_distance"] = sum(normalized_edit_distance(p, r) for p, r in zip(predictions, references)) / len(predictions)
            elif metric == "rouge":
                rouge_scores = [rouge_score(p, r) for p, r in zip(predictions, references)]
                results["rouge-1"] = sum(score["rouge-1"] for score in rouge_scores) / len(predictions)
                results["rouge-2"] = sum(score["rouge-2"] for score in rouge_scores) / len(predictions)
                results["rouge-l"] = sum(score["rouge-l"] for score in rouge_scores) / len(predictions)
            elif metric == "mathematical_accuracy":
                results["mathematical_accuracy"] = mathematical_accuracy(predictions, references)
            elif metric == "numeric_error":
                error = numeric_relative_error(predictions, references)
                if error is not None:
                    results["numeric_relative_error"] = error
            elif metric == "step_by_step_accuracy":
                results["step_by_step_accuracy"] = sum(step_by_step_reasoning_score(p, r) for p, r in zip(predictions, references)) / len(predictions)
            elif metric == "final_answer_accuracy":
                results["final_answer_accuracy"] = sum(mathematical_equivalence(p, r) for p, r in zip(predictions, references)) / len(predictions)
    
    # Compute metrics that use problem statements, if available
    if problems:
        if len(predictions) != len(problems):
            logger.warning(f"Length mismatch: {len(predictions)} predictions vs {len(problems)} problems")
            min_len = min(len(predictions), len(problems))
            predictions = predictions[:min_len]
            problems = problems[:min_len]
            
        # Check for appropriate metrics
        for metric in metrics_list:
            if metric == "problem_relevance":
                # Simple heuristic: Check for number overlap between problem and prediction
                relevances = []
                for problem, pred in zip(problems, predictions):
                    prob_nums = extract_numbers(problem)
                    pred_nums = extract_numbers(pred)
                    overlap = len(set(prob_nums).intersection(set(pred_nums)))
                    total = len(set(prob_nums))
                    relevances.append(overlap / total if total > 0 else 0.0)
                results["problem_relevance"] = sum(relevances) / len(problems)
    
    return results 