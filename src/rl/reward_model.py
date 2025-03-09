"""
Reward model implementation for evaluating mathematical responses.
This module provides reward functions for both PPO and GRPO training.
"""

import re
import logging
import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.core.sympify import SympifyError

logger = logging.getLogger(__name__)

class MathRewardModel:
    """
    A class that provides reward functions for mathematical reasoning.
    This model evaluates the quality of mathematical solutions by considering:
    1. Correctness of the final answer
    2. Step-by-step reasoning quality
    3. Mathematical validity
    4. Brevity/conciseness
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        reference_model_path: Optional[str] = None,
        device: str = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the reward model.
        
        Args:
            tokenizer: Tokenizer for text processing
            reference_model_path: Path to a reference model for KL penalty calculation
            device: Device to run the model on ('cuda' or 'cpu')
            config: Configuration dictionary with reward weights and parameters
        """
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reward weights from config
        self.math_correctness_weight = self.config.get("math_correctness_weight", 1.0)
        self.process_weight = self.config.get("process_weight", 0.5)
        self.brevity_weight = self.config.get("brevity_weight", 0.1)
        self.kl_penalty_weight = self.config.get("kl_penalty_weight", 0.05)
        self.max_length_penalty = self.config.get("max_length_penalty", 0.001)
        
        # Initialize reference model if provided
        self.reference_model = None
        if reference_model_path and self.config.get("use_reference_kl", True):
            try:
                # Load just for KL divergence calculation - quantize aggressively
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                from transformers import AutoModelForCausalLM
                logger.info(f"Loading reference model from {reference_model_path}")
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    reference_model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                logger.info("Reference model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reference model: {e}")
                self.reference_model = None
        
        # Load learned reward model if specified
        self.learned_reward_model = None
        if self.config.get("use_learned_reward", False):
            reward_model_path = self.config.get("learned_reward_model_path", None)
            if reward_model_path:
                try:
                    logger.info(f"Loading learned reward model from {reward_model_path}")
                    self.learned_reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
                    self.learned_reward_model = AutoModelForSequenceClassification.from_pretrained(
                        reward_model_path
                    ).to(self.device)
                    logger.info("Learned reward model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load learned reward model: {e}")
                    self.learned_reward_model = None
    
    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rewards for a batch of responses.
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            references: Optional list of reference answers
            
        Returns:
            Tuple containing (total_rewards, reward_components)
        """
        batch_size = len(prompts)
        
        # Initialize reward components
        reward_dict = {
            "correctness_reward": torch.zeros(batch_size, device=self.device),
            "process_reward": torch.zeros(batch_size, device=self.device),
            "brevity_reward": torch.zeros(batch_size, device=self.device),
            "kl_penalty": torch.zeros(batch_size, device=self.device),
            "learned_reward": torch.zeros(batch_size, device=self.device),
        }
        
        # Extract problem statements from prompts
        problems = [self._extract_problem_from_prompt(p) for p in prompts]
        
        # Compute correctness rewards
        if references:
            reward_dict["correctness_reward"] = self._compute_correctness_rewards(
                problems, responses, references
            )
        else:
            # If no references provided, calculate accuracy-based rewards
            reward_dict["correctness_reward"] = self._compute_math_validity_rewards(problems, responses)
        
        # Compute reasoning process rewards
        reward_dict["process_reward"] = self._compute_process_rewards(responses)
        
        # Compute brevity rewards
        reward_dict["brevity_reward"] = self._compute_brevity_rewards(responses)
        
        # Compute KL penalties if reference model is available
        if self.reference_model and self.config.get("use_reference_kl", True):
            reward_dict["kl_penalty"] = self._compute_kl_penalties(prompts, responses)
        
        # Compute learned rewards if model is available
        if self.learned_reward_model and self.config.get("use_learned_reward", False):
            reward_dict["learned_reward"] = self._compute_learned_rewards(prompts, responses)
        
        # Calculate total weighted rewards
        total_rewards = (
            self.math_correctness_weight * reward_dict["correctness_reward"] +
            self.process_weight * reward_dict["process_reward"] +
            self.brevity_weight * reward_dict["brevity_reward"] -
            self.kl_penalty_weight * reward_dict["kl_penalty"]
        )
        
        # Add learned rewards if available
        if self.learned_reward_model and self.config.get("use_learned_reward", False):
            learned_weight = self.config.get("learned_reward_weight", 0.5)
            total_rewards += learned_weight * reward_dict["learned_reward"]
        
        return total_rewards, reward_dict
    
    def _extract_problem_from_prompt(self, prompt: str) -> str:
        """Extract the problem statement from a prompt."""
        # This is a simple extraction based on the prompt format
        # Might need to be adapted based on the actual prompt templates used
        if "Solution:" in prompt:
            return prompt.split("Solution:")[0].strip()
        else:
            return prompt.strip()
    
    def _compute_correctness_rewards(
        self,
        problems: List[str],
        responses: List[str],
        references: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards based on the correctness of the final answers.
        
        Args:
            problems: List of problem statements
            responses: List of model responses
            references: List of reference answers
            
        Returns:
            Tensor of correctness rewards
        """
        batch_size = len(problems)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i, (problem, response, reference) in enumerate(zip(problems, responses, references)):
            # Extract final answers
            model_answer = self._extract_final_answer(response)
            reference_answer = self._extract_final_answer(reference)
            
            # Compute exact match reward
            if self._normalize_answer(model_answer) == self._normalize_answer(reference_answer):
                rewards[i] += 1.0
            else:
                # Try to parse mathematical expressions and check for equivalence
                try:
                    model_expr = self._parse_math_expression(model_answer)
                    ref_expr = self._parse_math_expression(reference_answer)
                    
                    if model_expr is not None and ref_expr is not None:
                        if self._check_expression_equivalence(model_expr, ref_expr):
                            rewards[i] += 0.8  # Partial reward for equivalent but not exact match
                    
                    # Numerical similarity for numeric answers
                    if model_expr is not None and ref_expr is not None:
                        try:
                            model_val = float(model_expr)
                            ref_val = float(ref_expr)
                            if abs(model_val - ref_val) < 1e-6:
                                rewards[i] += 0.9  # High reward for numerically equivalent answers
                            elif abs(model_val - ref_val) / max(abs(ref_val), 1e-10) < 0.01:
                                rewards[i] += 0.7  # Partial reward for close answers
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    logger.debug(f"Error parsing math expressions: {e}")
            
            # Partial credit for having correct steps even if final answer is wrong
            if rewards[i] < 0.5 and self._contains_correct_steps(response, reference):
                rewards[i] += 0.3  # Partial reward for correct reasoning steps
        
        return rewards
    
    def _compute_math_validity_rewards(
        self,
        problems: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards based on mathematical validity when references aren't available.
        
        Args:
            problems: List of problem statements
            responses: List of model responses
            
        Returns:
            Tensor of validity rewards
        """
        batch_size = len(problems)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i, (problem, response) in enumerate(zip(problems, responses)):
            # Check for presence of a final answer
            if self._has_final_answer(response):
                rewards[i] += 0.3
                
                # Try to check mathematical consistency
                if self._check_mathematical_consistency(problem, response):
                    rewards[i] += 0.4
                
                # Check for calculator-like expressions followed by correct results
                if self._check_calculation_correctness(response):
                    rewards[i] += 0.3
            
            # Check for step-by-step reasoning
            if len(self._extract_reasoning_steps(response)) >= 2:
                rewards[i] += 0.2
                
            # Check for variable definitions
            if self._uses_variables(response):
                rewards[i] += 0.1
        
        # Cap rewards at 1.0
        rewards = torch.clamp(rewards, max=1.0)
        return rewards
    
    def _compute_process_rewards(self, responses: List[str]) -> torch.Tensor:
        """
        Compute rewards based on the quality of the reasoning process.
        
        Args:
            responses: List of model responses
            
        Returns:
            Tensor of process rewards
        """
        batch_size = len(responses)
        rewards = torch.zeros(batch_size, device=self.device)
        
        for i, response in enumerate(responses):
            # Extract reasoning steps
            steps = self._extract_reasoning_steps(response)
            
            # Reward based on number of steps (more detailed reasoning)
            num_steps = len(steps)
            if num_steps >= 3:
                step_reward = min(0.5, 0.1 * num_steps)  # Cap at 0.5
                rewards[i] += step_reward
            
            # Check for logical progression between steps
            if self._check_logical_progression(steps):
                rewards[i] += 0.3
            
            # Check for use of variables and equations
            if self._uses_variables(response):
                rewards[i] += 0.2
            
            # Check for use of proper mathematical notation
            if self._uses_math_notation(response):
                rewards[i] += 0.2
            
            # Check for explanations alongside steps
            if self._has_explanations(response):
                rewards[i] += 0.2
        
        # Cap rewards at 1.0
        rewards = torch.clamp(rewards, max=1.0)
        return rewards
    
    def _compute_brevity_rewards(self, responses: List[str]) -> torch.Tensor:
        """
        Compute rewards that penalize overly verbose answers.
        
        Args:
            responses: List of model responses
            
        Returns:
            Tensor of brevity rewards
        """
        batch_size = len(responses)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # Optimal length ranges
        min_optimal_length = 100  # Characters
        max_optimal_length = 1000  # Characters
        
        for i, response in enumerate(responses):
            length = len(response)
            
            # Perfect length gets reward of 1.0
            if min_optimal_length <= length <= max_optimal_length:
                rewards[i] = 1.0
            elif length < min_optimal_length:
                # Too short - partial reward
                rewards[i] = length / min_optimal_length
            else:
                # Too long - decreasing reward
                excess_chars = length - max_optimal_length
                length_penalty = self.max_length_penalty * excess_chars
                rewards[i] = max(0.0, 1.0 - length_penalty)
        
        return rewards
    
    def _compute_kl_penalties(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute KL-divergence penalties using a reference model.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Tensor of KL penalties
        """
        if not self.reference_model:
            return torch.zeros(len(prompts), device=self.device)
        
        batch_size = len(prompts)
        kl_penalties = torch.zeros(batch_size, device=self.device)
        
        try:
            # Process in smaller batches to avoid OOM errors
            max_batch = 4
            for i in range(0, batch_size, max_batch):
                batch_prompts = prompts[i:i+max_batch]
                batch_responses = responses[i:i+max_batch]
                
                batch_kl = self._compute_batch_kl(batch_prompts, batch_responses)
                kl_penalties[i:i+max_batch] = batch_kl
        except Exception as e:
            logger.warning(f"Error computing KL penalties: {e}")
        
        return kl_penalties
    
    def _compute_batch_kl(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute KL divergence for a small batch."""
        batch_size = len(prompts)
        kl_divs = torch.zeros(batch_size, device=self.device)
        
        # Prepare inputs
        inputs = []
        for prompt, response in zip(prompts, responses):
            inputs.append(prompt + " " + response)
        
        # Tokenize
        with torch.no_grad():
            inputs_tokens = self.tokenizer(
                inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            # Get log probs from reference model
            outputs = self.reference_model(
                input_ids=inputs_tokens.input_ids,
                attention_mask=inputs_tokens.attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits
            log_probs_ref = torch.log_softmax(logits, dim=-1)
            
            # KL is approximated here as we don't have policy model probs
            # Using a simple heuristic based on token positions
            for i in range(batch_size):
                # Only consider response tokens for KL
                prompt_len = len(self.tokenizer.encode(prompts[i]))
                response_len = len(self.tokenizer.encode(responses[i])) - 1  # Exclude BOS token
                
                # Calculate average negative log-prob of response tokens
                start_idx = min(prompt_len, inputs_tokens.input_ids.size(1))
                end_idx = min(start_idx + response_len, inputs_tokens.input_ids.size(1))
                
                if start_idx < end_idx:
                    token_ids = inputs_tokens.input_ids[i, start_idx:end_idx]
                    token_probs = torch.gather(
                        log_probs_ref[i, start_idx-1:end_idx-1], 
                        1, 
                        token_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Higher negative log-prob means higher KL divergence
                    kl_divs[i] = -token_probs.mean()
        
        return kl_divs
    
    def _compute_learned_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards using a learned reward model.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Tensor of learned rewards
        """
        if not self.learned_reward_model:
            return torch.zeros(len(prompts), device=self.device)
        
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, device=self.device)
        
        try:
            # Process in smaller batches
            max_batch = 8
            for i in range(0, batch_size, max_batch):
                batch_prompts = prompts[i:i+max_batch]
                batch_responses = responses[i:i+max_batch]
                
                # Combine prompts and responses for the reward model
                batch_inputs = [p + " " + r for p, r in zip(batch_prompts, batch_responses)]
                
                # Tokenize
                inputs = self.learned_reward_tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get rewards
                with torch.no_grad():
                    outputs = self.learned_reward_model(**inputs)
                    batch_rewards = torch.sigmoid(outputs.logits.squeeze(-1))
                    rewards[i:i+len(batch_prompts)] = batch_rewards
        except Exception as e:
            logger.warning(f"Error computing learned rewards: {e}")
        
        return rewards
    
    def compute_preference_scores(
        self,
        prompts: List[str],
        response_pairs: List[Tuple[str, str]],
        references: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute preference scores for pairs of responses (for GRPO).
        
        Args:
            prompts: List of prompts
            response_pairs: List of tuples of responses (response_a, response_b)
            references: Optional list of reference answers
            
        Returns:
            Tuple containing (preference_scores, preference_margins)
        """
        batch_size = len(prompts)
        
        # Separate response pairs
        responses_a = [pair[0] for pair in response_pairs]
        responses_b = [pair[1] for pair in response_pairs]
        
        # Compute rewards for each response
        rewards_a, _ = self.compute_rewards(prompts, responses_a, references)
        rewards_b, _ = self.compute_rewards(prompts, responses_b, references)
        
        # Calculate preference scores (sigmoid of reward differences)
        reward_diffs = rewards_a - rewards_b
        preference_scores = torch.sigmoid(reward_diffs * 5.0)  # Scale for sharper preferences
        
        # Calculate preference margins (absolute difference in rewards)
        preference_margins = torch.abs(reward_diffs)
        
        return preference_scores, preference_margins
    
    def get_binary_preferences(
        self,
        prompts: List[str],
        response_pairs: List[Tuple[str, str]],
        references: Optional[List[str]] = None,
        threshold: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get binary preference labels for response pairs.
        
        Args:
            prompts: List of prompts
            response_pairs: List of tuples of responses (response_a, response_b)
            references: Optional list of reference answers
            threshold: Minimum reward difference to establish preference
            
        Returns:
            Tuple containing (binary_preferences, preference_strengths)
            binary_preferences: 1 if A > B, 0 if B > A, 0.5 if no clear preference
            preference_strengths: Confidence in the preference
        """
        preference_scores, preference_margins = self.compute_preference_scores(
            prompts, response_pairs, references
        )
        
        # Convert to binary preferences with threshold
        binary_prefs = torch.ones_like(preference_scores) * 0.5  # Default to no preference
        binary_prefs[preference_scores > 0.5 + threshold/2] = 1.0  # Prefer A
        binary_prefs[preference_scores < 0.5 - threshold/2] = 0.0  # Prefer B
        
        # Preference strength based on margin
        preference_strengths = torch.clamp(preference_margins / threshold, max=1.0)
        
        return binary_prefs, preference_strengths
    
    # Helper methods for reward calculation
    
    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from a solution text."""
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
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize an answer string for comparison."""
        # Remove whitespace and convert to lowercase
        text = re.sub(r'\s+', '', text.lower())
        
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
    
    def _parse_math_expression(self, text: str) -> Optional[sympy.Expr]:
        """Parse a mathematical expression using SymPy."""
        try:
            # Try direct parsing
            return parse_expr(text.replace('=', ''), evaluate=True)
        except (SympifyError, TypeError, ValueError):
            # Try extracting a number
            try:
                number_match = re.search(r'[-+]?\d*\.?\d+', text)
                if number_match:
                    return sympy.Number(float(number_match.group(0)))
            except (SympifyError, TypeError, ValueError):
                pass
                
            # Try parsing LaTeX
            try:
                if '$' in text or '\\' in text:
                    # Extract LaTeX between $ signs
                    latex_match = re.search(r'\$(.*?)\$', text)
                    if latex_match:
                        return parse_latex(latex_match.group(1))
            except Exception:
                pass
                
        return None
    
    def _check_expression_equivalence(
        self,
        expr1: sympy.Expr,
        expr2: sympy.Expr
    ) -> bool:
        """Check if two mathematical expressions are equivalent."""
        try:
            # Try direct equality
            if expr1 == expr2:
                return True
                
            # Try simplifying the difference
            diff = sympy.simplify(expr1 - expr2)
            if diff == 0:
                return True
                
            # For numerical expressions, check numerical closeness
            if expr1.is_Number and expr2.is_Number:
                return abs(float(expr1) - float(expr2)) < 1e-6
        except Exception:
            pass
            
        return False
    
    def _contains_correct_steps(self, response: str, reference: str) -> bool:
        """Check if the response contains steps similar to the reference solution."""
        # Extract steps from both texts
        response_steps = self._extract_reasoning_steps(response)
        reference_steps = self._extract_reasoning_steps(reference)
        
        if not response_steps or not reference_steps:
            return False
            
        # Look for key expressions from reference in the response
        key_expressions = []
        for step in reference_steps:
            # Extract mathematical expressions
            expressions = re.findall(r'(\d+\.?\d*|\w+)\s*=\s*(\d+\.?\d*)', step)
            key_expressions.extend(expressions)
            
            # Extract equation fragments
            eqs = re.findall(r'(\w+\s*[-+*/]\s*\w+)', step)
            key_expressions.extend(eqs)
        
        # Count how many key expressions appear in the response
        matches = 0
        for expr in key_expressions:
            expr_str = str(expr)
            if expr_str in response:
                matches += 1
                
        # If at least 30% of key expressions match, consider it contains correct steps
        if key_expressions and matches / len(key_expressions) >= 0.3:
            return True
            
        return False
    
    def _has_final_answer(self, text: str) -> bool:
        """Check if the text contains a clearly marked final answer."""
        patterns = [
            r"(?:final answer|answer|result)(?:\s+is)?(?:\s*:)",
            r"(?:therefore|thus|so|hence)(?:\s+,)?\s+.+",
            r"(?:=\s*).+?(?:\s*$|\s*\.)"
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def _check_mathematical_consistency(self, problem: str, response: str) -> bool:
        """Check for mathematical consistency between problem and response."""
        # Extract numbers from problem
        problem_numbers = re.findall(r'\d+\.?\d*', problem)
        
        # Check if some of these numbers appear in the response
        matches = 0
        for num in problem_numbers:
            if num in response:
                matches += 1
                
        return matches >= min(len(problem_numbers), 2)
    
    def _check_calculation_correctness(self, text: str) -> bool:
        """Check for calculator-like expressions followed by correct results."""
        # Find calculation patterns: expr = result
        calculations = re.findall(r'([\d\s+\-*/()\.]+)\s*=\s*(\d+\.?\d*)', text)
        
        correct_calcs = 0
        for calc_expr, result in calculations:
            try:
                expected = eval(calc_expr)
                actual = float(result)
                if abs(expected - actual) < 1e-6:
                    correct_calcs += 1
            except Exception:
                pass
                
        return correct_calcs > 0
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from a solution text."""
        # Split by newlines and certain patterns
        steps = []
        
        # First try splitting by numbered steps
        numbered_steps = re.findall(r'(?:^|\n)(?:\d+\.\s*)([^\n]+)', text)
        if numbered_steps:
            steps.extend(numbered_steps)
        else:
            # Try splitting by lines
            lines = text.split('\n')
            steps.extend([line.strip() for line in lines if line.strip()])
            
        # Filter out very short steps and headers
        steps = [step for step in steps if len(step) > 10]
        
        return steps
    
    def _check_logical_progression(self, steps: List[str]) -> bool:
        """Check if steps show logical progression."""
        if len(steps) < 2:
            return False
            
        # Check for variable reuse between steps
        variables_seen = set()
        for step in steps:
            # Extract variable assignments
            var_assignments = re.findall(r'(\w+)\s*=', step)
            vars_in_step = set(var_assignments)
            
            # Extract variables used in expressions
            vars_used = re.findall(r'[^=](\w+)(?:\s*[-+*/])', step)
            vars_used = [v for v in vars_used if v.isalpha() and not v.lower() in ('is', 'the', 'and', 'or')]
            
            # Check if this step uses variables from previous steps
            if variables_seen.intersection(vars_used):
                return True
                
            # Update seen variables
            variables_seen.update(vars_in_step)
            
        return False
    
    def _uses_variables(self, text: str) -> bool:
        """Check if the solution uses algebraic variables properly."""
        # Look for variable assignments
        var_assignments = re.findall(r'(?:let|set|define)\s+(\w+)\s*=', text.lower())
        if var_assignments:
            return True
            
        # Look for algebraic expressions
        algebra_patterns = re.findall(r'(\w+)(?:\s*[-+*/]\s*\w+)+\s*=', text)
        if algebra_patterns:
            return True
            
        return False
    
    def _uses_math_notation(self, text: str) -> bool:
        """Check if proper mathematical notation is used."""
        # Check for LaTeX
        if re.search(r'\$.*?\$', text):
            return True
            
        # Check for fractions, square roots, exponents
        notation_patterns = [
            r'\\frac{.*?}{.*?}',
            r'\\sqrt{.*?}',
            r'\^[0-9]',
            r'[0-9]\/[0-9]'
        ]
        
        for pattern in notation_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def _has_explanations(self, text: str) -> bool:
        """Check if the solution includes explanations with the steps."""
        explanation_markers = [
            'because', 'since', 'as', 'therefore', 'thus', 
            'this means', 'which means', 'so', 'now', 'we can'
        ]
        
        for marker in explanation_markers:
            if marker in text.lower():
                return True
                
        return False


# Factory function to create reward model based on configuration
def create_reward_model(
    tokenizer: PreTrainedTokenizer,
    reference_model_path: Optional[str] = None,
    config: Dict[str, Any] = None
) -> MathRewardModel:
    """
    Create a reward model based on the provided configuration.
    
    Args:
        tokenizer: Tokenizer for processing text
        reference_model_path: Path to reference model for KL penalty
        config: Configuration dictionary
        
    Returns:
        MathRewardModel instance
    """
    return MathRewardModel(
        tokenizer=tokenizer,
        reference_model_path=reference_model_path,
        config=config
    ) 