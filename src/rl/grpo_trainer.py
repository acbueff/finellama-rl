"""
Gradient-based Reinforcement with Paired Optimization (GRPO) trainer for language models.
This module implements GRPO training for fine-tuning language models.
"""

import os
import logging
import math
import time
import random
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedTokenizer, PreTrainedModel

from datasets import Dataset
from tqdm.auto import tqdm

from ..models.finemathllama import FineMathLlamaModel
from .reward_model import MathRewardModel
from .rollout import RolloutCollector, GRPORolloutCollector
from .base_trainer import BaseRLTrainer

logger = logging.getLogger(__name__)

class GRPOTrainer(BaseRLTrainer):
    """
    Gradient-based Reinforcement with Paired Optimization (GRPO) trainer for language models.
    GRPO is a reinforcement learning algorithm that optimizes language models based on
    preference pairs rather than absolute rewards.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, FineMathLlamaModel],
        tokenizer: PreTrainedTokenizer,
        reward_model: MathRewardModel,
        config: Dict[str, Any],
        rollout_collector: Optional[GRPORolloutCollector] = None,
        device: Optional[str] = None,
        ref_model: Optional[PreTrainedModel] = None
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            reward_model: Reward model for computing rewards
            config: Training configuration
            rollout_collector: Rollout collector for gathering experiences
            device: Device to train on ('cuda' or 'cpu')
            ref_model: Reference model for KL penalty
        """
        super().__init__(model, tokenizer, reward_model, config, rollout_collector, device)
        
        # GRPO-specific parameters
        self.grpo_epochs = self.config.get("grpo_epochs", 3)
        self.preference_threshold = self.config.get("preference_threshold", 0.1)
        self.beta = self.config.get("beta", 0.1)  # Regularization coefficient
        
        # Reference model handling
        self.ref_model = ref_model
        if self.ref_model is None and self.config.get("use_reference_kl", True):
            self.ref_model = self._create_reference_model()
            
        # Preference model (if enabled)
        self.use_preference_model = self.config.get("use_preference_model", True)
        self.preference_model = None
        if self.use_preference_model:
            self._init_preference_model()
            
        # Initialize rollout collector if not provided
        if rollout_collector is None:
            from .rollout import create_rollout_collector
            self.rollout_collector = create_rollout_collector(
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_model,  # Pass whole reward model for preferences
                method="grpo",
                config=self.config,
                device=self.device
            )
        else:
            self.rollout_collector = rollout_collector
        
    def _create_reference_model(self) -> Optional[PreTrainedModel]:
        """Create a copy of the model to use as a reference for KL penalty."""
        try:
            logger.info("Creating reference model for KL penalty")
            
            # Create a copy of the model
            from transformers import AutoModelForCausalLM
            
            if self.model_wrapper:
                model_name = self.model_wrapper.config.get("model_name_or_path")
            else:
                model_name = self.model.config._name_or_path
                
            # Use 4-bit quantization for memory efficiency
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Freeze the reference model
            for param in ref_model.parameters():
                param.requires_grad = False
                
            return ref_model
        except Exception as e:
            logger.warning(f"Failed to create reference model: {e}")
            return None
            
    def _init_preference_model(self):
        """Initialize a preference model for learning preferences."""
        try:
            logger.info("Initializing preference model")
            
            # Get preference model configuration
            preference_model_type = self.config.get("preference_model_type", "bert-base")
            
            # Use a simple binary classifier
            from transformers import AutoModel, AutoTokenizer
            
            # Load the base model
            base_model = AutoModel.from_pretrained(preference_model_type)
            self.preference_tokenizer = AutoTokenizer.from_pretrained(preference_model_type)
            
            # Create a binary classifier on top of the base model
            class PreferenceModel(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model
                    self.dropout = nn.Dropout(0.1)
                    self.classifier = nn.Linear(base_model.config.hidden_size, 1)
                    
                def forward(self, input_ids, attention_mask):
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    # Use the [CLS] embedding as the representation
                    cls_output = outputs.last_hidden_state[:, 0]
                    cls_output = self.dropout(cls_output)
                    logits = self.classifier(cls_output)
                    return logits
                    
            self.preference_model = PreferenceModel(base_model).to(self.device)
            
            # Create optimizer for preference model
            self.preference_optimizer = AdamW(
                self.preference_model.parameters(),
                lr=self.config.get("preference_model_lr", 5e-5)
            )
            
        except Exception as e:
            logger.warning(f"Failed to initialize preference model: {e}")
            self.preference_model = None
            self.use_preference_model = False
            
    def _train_preference_model(
        self,
        prompts: List[str],
        response_pairs: List[Tuple[str, str]],
        preferences: torch.Tensor,
        confidence: torch.Tensor,
        num_steps: int = 10
    ) -> float:
        """
        Train the preference model on paired responses.
        
        Args:
            prompts: List of prompts
            response_pairs: List of (response_a, response_b) tuples
            preferences: Tensor of preferences (1 if A>B, 0 if B>A, 0.5 if tied)
            confidence: Confidence in the preferences
            num_steps: Number of training steps
            
        Returns:
            Average training loss
        """
        if not self.use_preference_model or self.preference_model is None:
            return 0.0
            
        logger.info("Training preference model")
        
        # Prepare training data
        inputs_a = [p + " " + r[0] for p, r in zip(prompts, response_pairs)]
        inputs_b = [p + " " + r[1] for p, r in zip(prompts, response_pairs)]
        
        # Use only samples with clear preferences
        clear_prefs = (preferences != 0.5).nonzero().squeeze(-1).cpu().numpy()
        if len(clear_prefs) == 0:
            logger.warning("No clear preferences found for training preference model")
            return 0.0
            
        # Filter data
        filtered_inputs_a = [inputs_a[i] for i in clear_prefs]
        filtered_inputs_b = [inputs_b[i] for i in clear_prefs]
        filtered_prefs = preferences[clear_prefs]
        filtered_conf = confidence[clear_prefs]
        
        num_samples = len(filtered_inputs_a)
        batch_size = min(self.config.get("preference_model_batch_size", 16), num_samples)
        
        all_losses = []
        
        self.preference_model.train()
        
        # Training loop
        for step in range(num_steps):
            # Sample batch indices
            batch_indices = random.sample(range(num_samples), min(batch_size, num_samples))
            
            # Prepare batch
            batch_inputs_a = [filtered_inputs_a[i] for i in batch_indices]
            batch_inputs_b = [filtered_inputs_b[i] for i in batch_indices]
            batch_prefs = filtered_prefs[batch_indices].to(self.device)
            batch_conf = filtered_conf[batch_indices].to(self.device)
            
            # Tokenize inputs
            encoded_a = self.preference_tokenizer(
                batch_inputs_a,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            encoded_b = self.preference_tokenizer(
                batch_inputs_b,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass for both inputs
            self.preference_optimizer.zero_grad()
            
            logits_a = self.preference_model(encoded_a.input_ids, encoded_a.attention_mask)
            logits_b = self.preference_model(encoded_b.input_ids, encoded_b.attention_mask)
            
            # Calculate preference loss
            # For pairs where A is preferred (pref=1), logits_a should be higher
            # For pairs where B is preferred (pref=0), logits_b should be higher
            logits_diff = logits_a - logits_b
            loss = F.binary_cross_entropy_with_logits(
                logits_diff, batch_prefs.view(-1, 1), weight=batch_conf.view(-1, 1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.preference_model.parameters(), 1.0)
            self.preference_optimizer.step()
            
            all_losses.append(loss.item())
            
        self.preference_model.eval()
        
        # Return average loss
        return sum(all_losses) / len(all_losses) if all_losses else 0.0
        
    def _predict_preferences(
        self,
        prompts: List[str],
        response_pairs: List[Tuple[str, str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict preferences using the preference model.
        
        Args:
            prompts: List of prompts
            response_pairs: List of (response_a, response_b) tuples
            
        Returns:
            Tuple of (preferences, confidence)
        """
        if not self.use_preference_model or self.preference_model is None:
            # Return neutral preferences if no model
            batch_size = len(prompts)
            return (
                torch.ones(batch_size, device=self.device) * 0.5,
                torch.zeros(batch_size, device=self.device)
            )
            
        self.preference_model.eval()
        
        batch_size = len(prompts)
        preferences = torch.zeros(batch_size, device=self.device)
        confidence = torch.zeros(batch_size, device=self.device)
        
        # Prepare inputs
        inputs_a = [p + " " + r[0] for p, r in zip(prompts, response_pairs)]
        inputs_b = [p + " " + r[1] for p, r in zip(prompts, response_pairs)]
        
        # Process in smaller batches to avoid OOM
        max_batch = 8
        for i in range(0, batch_size, max_batch):
            batch_inputs_a = inputs_a[i:i+max_batch]
            batch_inputs_b = inputs_b[i:i+max_batch]
            
            # Tokenize inputs
            encoded_a = self.preference_tokenizer(
                batch_inputs_a,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            encoded_b = self.preference_tokenizer(
                batch_inputs_b,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits_a = self.preference_model(encoded_a.input_ids, encoded_a.attention_mask)
                logits_b = self.preference_model(encoded_b.input_ids, encoded_b.attention_mask)
                
                # Calculate preferences
                logits_diff = logits_a - logits_b
                batch_prefs = torch.sigmoid(logits_diff).squeeze(-1)
                
                # Calculate confidence
                batch_conf = torch.abs(batch_prefs - 0.5) * 2  # Scale to [0, 1]
                
                preferences[i:i+len(batch_prefs)] = batch_prefs
                confidence[i:i+len(batch_conf)] = batch_conf
                
        return preferences, confidence
        
    def compute_kl_divergence(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.FloatTensor,
        prompt_lens: List[int]
    ) -> torch.Tensor:
        """
        Compute KL divergence between the current policy and reference model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            prompt_lens: Lengths of prompts
            
        Returns:
            KL divergence tensor
        """
        if self.ref_model is None:
            # Return zero KL if no reference model
            return torch.zeros(input_ids.size(0), device=self.device)
        
        batch_size = input_ids.size(0)
        kl_divs = torch.zeros(batch_size, device=self.device)
        
        try:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
            # Get logits from current model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Compute KL divergence
            for i in range(batch_size):
                # Only consider response tokens
                start_idx = prompt_lens[i]
                if start_idx >= input_ids.size(1) - 1:
                    continue
                    
                # Get log probs for the next token predictions
                p_log_probs = log_probs[i, start_idx-1:-1]  # Current model
                q_log_probs = ref_log_probs[i, start_idx-1:-1]  # Reference model
                
                # Get token probabilities
                p = torch.exp(p_log_probs)
                
                # Get the specific token logprobs
                next_tokens = input_ids[i, start_idx:]
                
                # Gather log probs for the chosen tokens
                p_token_log_probs = torch.gather(
                    p_log_probs, 1, next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                q_token_log_probs = torch.gather(
                    q_log_probs, 1, next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                # Average KL across sequence
                kl_divs[i] = (p_token_log_probs - q_token_log_probs).mean()
                
        except Exception as e:
            logger.warning(f"Error computing KL divergence: {e}")
            
        return kl_divs
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss for a batch of paired data.
        
        Args:
            batch: Batch of data with pairs of responses
            
        Returns:
            Dictionary containing loss and other metrics
        """
        # Unpack batch
        # For response A
        input_ids_a = batch["input_ids_a"].to(self.device)
        attention_mask_a = batch["attention_mask_a"].to(self.device)
        
        # For response B
        input_ids_b = batch["input_ids_b"].to(self.device)
        attention_mask_b = batch["attention_mask_b"].to(self.device)
        
        # Preferences and prompt lengths
        preferences = batch["preferences"].to(self.device)
        confidence = batch["confidence"].to(self.device)
        prompt_lens = batch["prompt_lens"]
        
        # Forward pass for both responses
        outputs_a = self.model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            return_dict=True
        )
        
        outputs_b = self.model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            return_dict=True
        )
        
        # Get logits
        logits_a = outputs_a.logits
        logits_b = outputs_b.logits
        
        # Compute log probabilities
        log_probs_a = F.log_softmax(logits_a, dim=-1)
        log_probs_b = F.log_softmax(logits_b, dim=-1)
        
        batch_size = input_ids_a.size(0)
        policy_loss = torch.zeros(batch_size, device=self.device)
        kl_div_a = torch.zeros(batch_size, device=self.device)
        kl_div_b = torch.zeros(batch_size, device=self.device)
        
        # For each sample in the batch
        for i in range(batch_size):
            # Only consider tokens after the prompt
            start_idx = prompt_lens[i]
            if start_idx >= input_ids_a.size(1) - 1 or start_idx >= input_ids_b.size(1) - 1:
                continue
                
            # Extract the response token logprobs for both A and B
            # For the chosen tokens, we need to gather the specific logprobs
            token_ids_a = input_ids_a[i, start_idx:]
            token_logprobs_a = torch.gather(
                log_probs_a[i, start_idx-1:-1],
                1, 
                token_ids_a.unsqueeze(-1)
            ).squeeze(-1)
            
            token_ids_b = input_ids_b[i, start_idx:]
            token_logprobs_b = torch.gather(
                log_probs_b[i, start_idx-1:-1],
                1, 
                token_ids_b.unsqueeze(-1)
            ).squeeze(-1)
            
            # Average sequence logprobs (sequence-level log probability)
            seq_logprob_a = token_logprobs_a.mean()
            seq_logprob_b = token_logprobs_b.mean()
            
            # Compute preference-weighted loss
            # If A is preferred (preferences[i] > 0.5), we want to increase P(A) and decrease P(B)
            # If B is preferred (preferences[i] < 0.5), we want to increase P(B) and decrease P(A)
            preference_weight = (preferences[i] - 0.5) * 2  # Scale to [-1, 1]
            
            # Skip if no clear preference
            if torch.abs(preference_weight) < self.preference_threshold:
                continue
                
            # Weight by confidence
            preference_weight = preference_weight * confidence[i]
            
            # Compute the GRPO loss
            # We want to maximize: preference_weight * (log_prob_a - log_prob_b)
            policy_loss[i] = -preference_weight * (seq_logprob_a - seq_logprob_b)
            
        # Compute KL divergence if reference model is available
        if self.ref_model is not None:
            kl_div_a = self.compute_kl_divergence(input_ids_a, attention_mask_a, prompt_lens)
            kl_div_b = self.compute_kl_divergence(input_ids_b, attention_mask_b, prompt_lens)
        
        # Mean across batch
        policy_loss_mean = policy_loss.mean()
        kl_div_a_mean = kl_div_a.mean()
        kl_div_b_mean = kl_div_b.mean()
        kl_div_mean = (kl_div_a_mean + kl_div_b_mean) / 2
        
        # Compute total loss with KL regularization
        kl_coef = self.config.get("kl_penalty_weight", 0.05)
        loss = policy_loss_mean + kl_coef * kl_div_mean
        
        return {
            "loss": loss,
            "policy_loss": policy_loss_mean,
            "kl_div": kl_div_mean,
            "kl_div_a": kl_div_a_mean,
            "kl_div_b": kl_div_b_mean,
            "avg_preference": preferences.mean(),
            "avg_confidence": confidence.mean()
        }
        
    def _prepare_training_data(
        self,
        experiences: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare experiences for GRPO training.
        
        Args:
            experiences: Collected experiences from rollout
            
        Returns:
            Dictionary of tensors ready for GRPO training
        """
        # Extract data
        prompts = experiences["prompts"]
        response_pairs = experiences["response_pairs"]
        preference_scores = experiences.get("preference_scores", None)
        preference_strengths = experiences.get("preference_strengths", None)
        binary_preferences = experiences.get("binary_preferences", None)
        
        # Convert to tensors if not already
        if preference_scores is not None and not isinstance(preference_scores, torch.Tensor):
            preference_scores = torch.tensor(preference_scores, device=self.device)
            
        if preference_strengths is not None and not isinstance(preference_strengths, torch.Tensor):
            preference_strengths = torch.tensor(preference_strengths, device=self.device)
            
        if binary_preferences is not None and not isinstance(binary_preferences, torch.Tensor):
            binary_preferences = torch.tensor(binary_preferences, device=self.device)
            
        # If we have a preference model, predict preferences
        if self.use_preference_model and self.preference_model is not None:
            # Combine reward model preferences with learned preferences
            model_preferences, model_confidence = self._predict_preferences(prompts, response_pairs)
            
            # If we have reward preferences, blend them
            if binary_preferences is not None:
                alpha = self.config.get("preference_model_weight", 0.5)
                # Blend preferences
                preferences = alpha * model_preferences + (1 - alpha) * binary_preferences
                # Take max confidence
                confidence = torch.max(model_confidence, preference_strengths)
            else:
                preferences = model_preferences
                confidence = model_confidence
        else:
            # Use the reward model preferences directly
            preferences = binary_preferences if binary_preferences is not None else preference_scores
            confidence = preference_strengths if preference_strengths is not None else torch.ones_like(preferences)
            
        # Train preference model if enabled
        if self.use_preference_model and self.preference_model is not None:
            # Only train if we have rewards to learn from
            if binary_preferences is not None:
                preference_loss = self._train_preference_model(
                    prompts, 
                    response_pairs, 
                    binary_preferences,
                    preference_strengths,
                    num_steps=self.config.get("preference_model_steps", 10)
                )
                logger.info(f"Preference model training loss: {preference_loss:.4f}")
                
        # Prepare response pairs
        responses_a = [pair[0] for pair in response_pairs]
        responses_b = [pair[1] for pair in response_pairs]
        
        # Tokenize full sequences (prompt + response) for each pair
        prompt_response_pairs_a = [p + " " + r for p, r in zip(prompts, responses_a)]
        prompt_response_pairs_b = [p + " " + r for p, r in zip(prompts, responses_b)]
        
        encoded_a = self.tokenizer(
            prompt_response_pairs_a,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 2048)
        ).to(self.device)
        
        encoded_b = self.tokenizer(
            prompt_response_pairs_b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 2048)
        ).to(self.device)
        
        # Get prompt lengths
        prompt_lens = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in prompts
        ]
        
        return {
            "input_ids_a": encoded_a.input_ids,
            "attention_mask_a": encoded_a.attention_mask,
            "input_ids_b": encoded_b.input_ids,
            "attention_mask_b": encoded_b.attention_mask,
            "preferences": preferences,
            "confidence": confidence,
            "prompt_lens": prompt_lens
        }
        
    def train(
        self, 
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: Optional[int] = None,
        eval_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model using GRPO.
        
        Args:
            dataset: Dataset containing prompts and references
            eval_dataset: Optional dataset for evaluation during training
            num_train_epochs: Number of training epochs
            eval_steps: Steps between evaluations
            
        Returns:
            Dictionary of training statistics
        """
        num_train_epochs = num_train_epochs or self.config.get("num_epochs", 5)
        eval_steps = eval_steps or self.config.get("eval_interval", 500)
        
        # Resume from saved global step
        start_step = self.global_step
        
        # Configure training parameters
        save_interval = self.config.get("save_interval", 1000)
        log_interval = self.config.get("log_interval", 10)
        num_sample_pairs = self.config.get("num_sample_pairs", 64)
        paired_batch_size = self.config.get("paired_batch_size", 16)
        
        # Output directory
        output_dir = self.config.get("output_dir", "checkpoints/grpo_model")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting GRPO training")
        logger.info(f"Number of epochs: {num_train_epochs}")
        logger.info(f"Number of steps: {self.max_steps}")
        logger.info(f"Starting from step: {start_step}")
        
        # Training loop
        self.model.train()
        
        # Keep track of best reward
        best_eval_reward = self.best_reward
        
        # Main training loop
        for epoch in range(num_train_epochs):
            logger.info(f"Epoch {epoch+1}/{num_train_epochs}")
            
            epoch_stats = []
            
            # Training steps within epoch
            step_in_epoch = 0
            while step_in_epoch < self.max_steps // num_train_epochs:
                # Collect paired experiences
                logger.info(f"Collecting paired experiences at step {self.global_step}")
                experiences = self.rollout_collector.collect_grpo_experiences(
                    dataset=dataset,
                    batch_size=paired_batch_size,
                    num_sample_pairs=num_sample_pairs
                )
                
                # Log experience statistics
                if "preference_scores" in experiences:
                    pref_scores = torch.tensor(experiences["preference_scores"], device=self.device)
                    logger.info(f"Average preference score: {pref_scores.mean().item():.4f}")
                    
                # Prepare experiences for training
                train_data = self._prepare_training_data(experiences)
                
                # Multiple epochs of GRPO optimization on the data
                for grpo_epoch in range(self.grpo_epochs):
                    # Create mini-batches
                    batch_size = len(experiences["prompts"])
                    indices = list(range(batch_size))
                    random.shuffle(indices)
                    
                    for start_idx in range(0, batch_size, self.mini_batch_size):
                        # Extract mini-batch
                        end_idx = min(start_idx + self.mini_batch_size, batch_size)
                        mini_batch_indices = indices[start_idx:end_idx]
                        
                        mini_batch = {
                            k: v[mini_batch_indices] if isinstance(v, torch.Tensor) else [v[i] for i in mini_batch_indices]
                            for k, v in train_data.items()
                        }
                        
                        # Forward and backward pass
                        self.optimizer.zero_grad()
                        loss_dict = self.compute_loss(mini_batch)
                        loss = loss_dict["loss"]
                        
                        # Skip if loss is invalid
                        if not torch.isfinite(loss).all():
                            logger.warning(f"Invalid loss: {loss}")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        
                        # Update parameters
                        self.optimizer.step()
                        
                        # Update learning rate
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        
                        # Log stats
                        if self.global_step % log_interval == 0:
                            # Convert tensor values to Python scalars
                            log_stats = {k: v.item() if hasattr(v, "item") else v for k, v in loss_dict.items()}
                            log_stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                            
                            # Log stats
                            self.log_stats(log_stats)
                            epoch_stats.append(log_stats)
                            
                        # Increment global step
                        self.global_step += 1
                        
                        # Evaluation
                        if eval_dataset is not None and self.global_step % eval_steps == 0:
                            logger.info(f"Evaluating at step {self.global_step}")
                            self.model.eval()
                            eval_stats = self.evaluate(eval_dataset)
                            self.model.train()
                            
                            # Log evaluation stats
                            self.log_stats({"eval_" + k: v for k, v in eval_stats.items()})
                            
                            # Save best model
                            if eval_stats["mean_reward"] > best_eval_reward:
                                best_eval_reward = eval_stats["mean_reward"]
                                self.best_reward = best_eval_reward
                                logger.info(f"New best reward: {best_eval_reward:.4f}")
                                self.save_checkpoint(
                                    output_dir=os.path.join(output_dir, "best"),
                                    save_optimizer=False
                                )
                                
                        # Save checkpoint
                        if self.global_step % save_interval == 0:
                            self.save_checkpoint(output_dir=output_dir, step=self.global_step)
                            
                        # Break if max steps reached
                        if self.global_step >= self.max_steps:
                            logger.info(f"Reached maximum steps: {self.global_step}")
                            break
                            
                    # Break if max steps reached
                    if self.global_step >= self.max_steps:
                        break
                        
                # Increment steps in epoch
                step_in_epoch += 1
                
                # Break if max steps reached
                if self.global_step >= self.max_steps:
                    break
                    
            # End of epoch - log epoch stats
            if epoch_stats:
                epoch_mean_stats = {
                    k: np.mean([s[k] for s in epoch_stats if k in s])
                    for k in epoch_stats[0].keys()
                }
                logger.info(f"Epoch {epoch+1} stats: {epoch_mean_stats}")
                
        # Final evaluation
        if eval_dataset is not None:
            logger.info("Final evaluation")
            self.model.eval()
            final_eval_stats = self.evaluate(eval_dataset)
            self.model.train()
            
            # Log final evaluation stats
            self.log_stats({"final_eval_" + k: v for k, v in final_eval_stats.items()})
            
            # Save final model
            self.save_checkpoint(output_dir=output_dir, step=self.global_step)
            
            # Save samples
            self.create_evaluation_samples(
                eval_dataset=eval_dataset,
                output_file=os.path.join(output_dir, "final_samples.json"),
            )
            
        logger.info(f"Training completed. Total steps: {self.global_step}")
        
        return {
            "global_step": self.global_step,
            "best_reward": self.best_reward
        } 