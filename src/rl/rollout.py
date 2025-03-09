"""
Rollout collection and environment interaction for RL training.
This module handles the generation of responses from the model and collecting feedback.
"""

import os
import logging
import random
import torch
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

logger = logging.getLogger(__name__)

class RolloutCollector:
    """
    Base class for collecting rollouts from a model during RL training.
    Handles common functionality for PPO and GRPO rollout collection.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_fn: Callable,
        config: Dict[str, Any] = None,
        device: str = None
    ):
        """
        Initialize the rollout collector.
        
        Args:
            model: Model to collect rollouts from
            tokenizer: Tokenizer for encoding/decoding text
            reward_fn: Function to compute rewards for generated responses
            config: Configuration parameters
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generation parameters
        self.do_sample = self.config.get("do_sample", True)
        self.temperature = self.config.get("temperature", 1.0)
        self.top_p = self.config.get("top_p", 1.0)
        self.top_k = self.config.get("top_k", 50)
        self.max_new_tokens = self.config.get("response_max_length", 512)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Set random seeds for reproducibility
        seed = self.config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None
    ) -> str:
        """
        Generate a response from the model for a given prompt.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated response string
        """
        # Use provided parameters or fall back to defaults
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        do_sample = do_sample if do_sample is not None else self.do_sample
        
        # Encode the prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.config.get("prompt_max_length", 512)
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get the generated tokens (excluding the prompt)
        generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
        
        # Decode the generated tokens
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response
    
    def generate_batch_responses(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            batch_size: Batch size for generation
            **generation_kwargs: Additional kwargs for generation
            
        Returns:
            List of generated response strings
        """
        responses = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Encode batch
            batch_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.get("prompt_max_length", 512)
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    max_new_tokens=generation_kwargs.get("max_new_tokens", self.max_new_tokens),
                    temperature=generation_kwargs.get("temperature", self.temperature),
                    top_p=generation_kwargs.get("top_p", self.top_p),
                    top_k=generation_kwargs.get("top_k", self.top_k),
                    do_sample=generation_kwargs.get("do_sample", self.do_sample),
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Process each item in the batch
            for j, (input_ids, output_ids) in enumerate(zip(
                batch_inputs.input_ids, batch_outputs.sequences
            )):
                # Get the number of tokens in the prompt
                prompt_length = input_ids.shape[0]
                
                # Extract only the generated part (response)
                response_ids = output_ids[prompt_length:]
                
                # Decode
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response)
                
        return responses
    
    def collect_experiences(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_rollouts: int = 64,
        prompt_key: str = "prompt",
        reference_key: Optional[str] = "reference"
    ) -> Dict[str, Any]:
        """
        Collect experiences by generating responses and computing rewards.
        
        Args:
            dataset: Dataset containing prompts and references
            batch_size: Batch size for generation
            num_rollouts: Number of rollouts to collect
            prompt_key: Key for prompts in the dataset
            reference_key: Key for reference answers in the dataset
            
        Returns:
            Dictionary containing collected experiences
        """
        # Sample from the dataset
        samples = dataset.select(
            random.sample(range(len(dataset)), min(num_rollouts, len(dataset)))
        )
        
        # Extract prompts and references
        prompts = samples[prompt_key]
        references = samples[reference_key] if reference_key in samples.features else None
        
        # Generate responses
        responses = self.generate_batch_responses(
            prompts=prompts,
            batch_size=batch_size
        )
        
        # Compute rewards
        rewards, reward_components = self.reward_fn(prompts, responses, references)
        
        # Prepare the experiences data
        experiences = {
            "prompts": prompts,
            "responses": responses,
            "rewards": rewards.cpu().numpy().tolist(),
            "reward_components": {k: v.cpu().numpy().tolist() for k, v in reward_components.items()},
        }
        
        if references:
            experiences["references"] = references
            
        return experiences
    

class PPORolloutCollector(RolloutCollector):
    """
    Specialized rollout collector for PPO training.
    """
    
    def collect_ppo_experiences(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_rollouts: int = 64,
        prompt_key: str = "prompt",
        reference_key: Optional[str] = "reference"
    ) -> Dict[str, Any]:
        """
        Collect experiences specifically for PPO training.
        
        Args:
            dataset: Dataset containing prompts and references
            batch_size: Batch size for generation
            num_rollouts: Number of rollouts to collect
            prompt_key: Key for prompts in the dataset
            reference_key: Key for reference answers in the dataset
            
        Returns:
            Dictionary containing collected experiences
        """
        experiences = self.collect_experiences(
            dataset=dataset,
            batch_size=batch_size,
            num_rollouts=num_rollouts,
            prompt_key=prompt_key,
            reference_key=reference_key
        )
        
        # For PPO, we need to collect the logprobs and values
        # We'll do this in a separate forward pass to avoid modifying the generation
        prompt_response_pairs = [
            p + " " + r for p, r in zip(experiences["prompts"], experiences["responses"])
        ]
        
        # Tokenize prompt-response pairs
        tokens = self.tokenizer(
            prompt_response_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 2048)
        ).to(self.device)
        
        # Extract lengths of prompts to separate prompts and responses
        prompt_lengths = [
            len(self.tokenizer.encode(p, add_special_tokens=False)) 
            for p in experiences["prompts"]
        ]
        
        # Forward pass to get logprobs
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                return_dict=True
            )
            
            logits = outputs.logits
            
            # Extract logprobs for generated tokens
            logprobs = torch.log_softmax(logits, dim=-1)
            
            # Get logprobs of actually generated tokens
            response_logprobs = []
            for i, (ids, logprob, p_len) in enumerate(zip(
                tokens.input_ids, logprobs, prompt_lengths
            )):
                # Skip the prompt tokens
                start_idx = min(p_len, ids.shape[0] - 1)
                
                # Get the logprobs of the chosen tokens (offset by 1 for next token prediction)
                token_logprobs = torch.gather(
                    logprob[start_idx-1:-1], 
                    1, 
                    ids[start_idx:].unsqueeze(-1)
                ).squeeze(-1)
                
                response_logprobs.append(token_logprobs)
        
        # Add logprobs to experiences
        experiences["response_logprobs"] = [lp.cpu().numpy().tolist() for lp in response_logprobs]
        
        return experiences
    

class GRPORolloutCollector(RolloutCollector):
    """
    Specialized rollout collector for GRPO training that generates pairs of responses.
    """
    
    def collect_grpo_experiences(
        self,
        dataset: Dataset,
        batch_size: int = 16,
        num_sample_pairs: int = 64,
        prompt_key: str = "prompt",
        reference_key: Optional[str] = "reference",
        temperature_range: Tuple[float, float] = (0.7, 1.3),
        diversity_coefficient: float = 0.1
    ) -> Dict[str, Any]:
        """
        Collect paired response experiences for GRPO training.
        
        Args:
            dataset: Dataset containing prompts and references
            batch_size: Batch size for generation
            num_sample_pairs: Number of response pairs to collect
            prompt_key: Key for prompts in the dataset
            reference_key: Key for reference answers in the dataset
            temperature_range: Range of temperatures for diverse sampling
            diversity_coefficient: How much to prioritize diversity in pairs
            
        Returns:
            Dictionary containing collected paired experiences
        """
        # Sample from the dataset
        samples = dataset.select(
            random.sample(range(len(dataset)), min(num_sample_pairs, len(dataset)))
        )
        
        # Extract prompts and references
        prompts = samples[prompt_key]
        references = samples[reference_key] if reference_key in samples.features else None
        
        # Generate first set of responses with base temperature
        responses_a = self.generate_batch_responses(
            prompts=prompts,
            batch_size=batch_size,
            temperature=self.temperature
        )
        
        # Generate second set of responses with varied temperature for diversity
        varied_temps = [
            random.uniform(temperature_range[0], temperature_range[1])
            for _ in range(len(prompts))
        ]
        
        responses_b = []
        for i, (prompt, temp) in enumerate(zip(prompts, varied_temps)):
            # Generate with varied temperature
            response = self.generate_response(
                prompt=prompt,
                temperature=temp,
                # Use higher top_k for more diversity
                top_k=self.top_k * 2
            )
            
            # If responses are too similar, try again with higher temperature
            max_attempts = 3
            attempts = 0
            
            while (self._compute_similarity(responses_a[i], response) > 0.8 and 
                  attempts < max_attempts):
                # Increase temperature to get more diversity
                temp = min(temp * 1.5, 2.0)
                response = self.generate_response(
                    prompt=prompt,
                    temperature=temp,
                    top_k=self.top_k * 2
                )
                attempts += 1
                
            responses_b.append(response)
        
        # Compute preference scores
        response_pairs = list(zip(responses_a, responses_b))
        preference_scores, preference_margins = self.reward_fn.compute_preference_scores(
            prompts, response_pairs, references
        )
        
        # Get binary preferences with a threshold
        threshold = self.config.get("preference_threshold", 0.1)
        binary_prefs, preference_strengths = self.reward_fn.get_binary_preferences(
            prompts, response_pairs, references, threshold
        )
        
        # Prepare the experiences data
        experiences = {
            "prompts": prompts,
            "response_pairs": response_pairs,
            "preference_scores": preference_scores.cpu().numpy().tolist(),
            "preference_margins": preference_margins.cpu().numpy().tolist(),
            "binary_preferences": binary_prefs.cpu().numpy().tolist(),
            "preference_strengths": preference_strengths.cpu().numpy().tolist(),
        }
        
        if references:
            experiences["references"] = references
            
        return experiences
    
    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute simple similarity between two texts.
        Returns a value between 0 (completely different) and 1 (identical).
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Similarity score
        """
        # Simple character-level Jaccard similarity
        a_chars = set(text_a)
        b_chars = set(text_b)
        
        if not a_chars and not b_chars:
            return 1.0
            
        intersection = len(a_chars.intersection(b_chars))
        union = len(a_chars.union(b_chars))
        
        return intersection / union
    
    def generate_diverse_pair(
        self,
        prompt: str,
        diversity_coefficient: float = 0.1,
        max_attempts: int = 5
    ) -> Tuple[str, str]:
        """
        Generate a diverse pair of responses for the same prompt.
        
        Args:
            prompt: Input prompt
            diversity_coefficient: How much to prioritize diversity
            max_attempts: Maximum number of attempts to generate diverse responses
            
        Returns:
            Tuple of (response_a, response_b)
        """
        # Generate first response
        response_a = self.generate_response(prompt)
        
        # Generate second response with higher temperature
        best_response_b = None
        lowest_similarity = 1.0
        
        for _ in range(max_attempts):
            # Increase temperature for diversity
            temp = min(self.temperature * 1.5, 2.0)
            
            response_b = self.generate_response(
                prompt=prompt,
                temperature=temp,
                top_k=self.top_k * 2
            )
            
            # Compute similarity
            similarity = self._compute_similarity(response_a, response_b)
            
            # Track the most diverse response
            if similarity < lowest_similarity:
                lowest_similarity = similarity
                best_response_b = response_b
                
            # If diverse enough, stop early
            if similarity < 0.5:
                break
                
        return response_a, best_response_b or response_b


def create_rollout_collector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    reward_fn: Callable,
    method: str = "ppo",
    config: Dict[str, Any] = None,
    device: str = None
) -> RolloutCollector:
    """
    Factory function to create the appropriate rollout collector.
    
    Args:
        model: Model to collect rollouts from
        tokenizer: Tokenizer for encoding/decoding text
        reward_fn: Function to compute rewards
        method: Reinforcement learning method ("ppo" or "grpo")
        config: Configuration parameters
        device: Device to run on
        
    Returns:
        Appropriate RolloutCollector instance
    """
    if method.lower() == "ppo":
        return PPORolloutCollector(model, tokenizer, reward_fn, config, device)
    elif method.lower() == "grpo":
        return GRPORolloutCollector(model, tokenizer, reward_fn, config, device)
    else:
        raise ValueError(f"Unknown RL method: {method}") 