"""
Proximal Policy Optimization (PPO) trainer for language models.
This module implements PPO training for fine-tuning language models.
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
from .rollout import RolloutCollector, PPORolloutCollector
from .base_trainer import BaseRLTrainer

logger = logging.getLogger(__name__)

class PPOTrainer(BaseRLTrainer):
    """
    Proximal Policy Optimization (PPO) trainer for language models.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, FineMathLlamaModel],
        tokenizer: PreTrainedTokenizer,
        reward_model: MathRewardModel,
        config: Dict[str, Any],
        rollout_collector: Optional[PPORolloutCollector] = None,
        device: Optional[str] = None,
        ref_model: Optional[PreTrainedModel] = None
    ):
        """
        Initialize the PPO trainer.
        
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
        
        # PPO-specific parameters
        self.ppo_epochs = self.config.get("ppo_epochs", 4)
        self.clip_range = self.config.get("clip_range", 0.2)
        self.value_clip_range = self.config.get("value_clip_range", 0.2)
        self.target_kl = self.config.get("target_kl", 0.01)
        self.normalize_advantage = self.config.get("normalize_advantage", True)
        self.vf_coef = self.config.get("vf_coef", 0.5)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.gamma = self.config.get("gamma", 0.99)
        self.lam = self.config.get("lam", 0.95)
        
        # Adaptive KL penalty
        self.use_adaptive_kl = self.config.get("use_adaptive_kl", True)
        self.init_kl_coef = self.config.get("init_kl_coef", 0.2)
        self.adap_kl_target = self.config.get("adap_kl_target", 6.0)
        self.kl_coef = self.init_kl_coef
        
        # Initialize reference model for KL penalty
        self.ref_model = ref_model
        if self.ref_model is None and self.config.get("use_reference_kl", True):
            self.ref_model = self._create_reference_model()
        
        # Initialize rollout collector if not provided
        if rollout_collector is None:
            from .rollout import create_rollout_collector
            self.rollout_collector = create_rollout_collector(
                model=self.model,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_model.compute_rewards,
                method="ppo",
                config=self.config,
                device=self.device
            )
        else:
            self.rollout_collector = rollout_collector
            
        # Initialize value head for PPO
        self.value_head = nn.Linear(
            self.model.config.hidden_size, 1, bias=False
        ).to(self.device)
        
        # Add value head parameters to optimizer
        self.optimizer.add_param_group({
            "params": self.value_head.parameters(),
            "weight_decay": 0.0
        })
    
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
            
            # Compute KL divergence: KL(p||q) = sum(p * (log(p) - log(q)))
            for i in range(batch_size):
                # Only consider response tokens
                start_idx = prompt_lens[i]
                if start_idx >= input_ids.size(1) - 1:
                    continue
                    
                # Get probability distributions
                p_log_probs = log_probs[i, start_idx-1:-1]  # Current model
                q_log_probs = ref_log_probs[i, start_idx-1:-1]  # Reference model
                
                # Get token probabilities
                p = torch.exp(p_log_probs)
                
                # Get next token ids to compute KL for
                next_tokens = input_ids[i, start_idx:]
                
                # Gather log probs for the chosen tokens
                p_token_log_probs = torch.gather(
                    p_log_probs, 1, next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                q_token_log_probs = torch.gather(
                    q_log_probs, 1, next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                
                # Compute token-level KL
                token_kl = p * (p_log_probs - q_log_probs)
                
                # Sum across vocabulary and average across sequence
                seq_kl = (p_token_log_probs - q_token_log_probs).mean()
                kl_divs[i] = seq_kl
                
        except Exception as e:
            logger.warning(f"Error computing KL divergence: {e}")
            
        return kl_divs
        
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for a batch of data.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary containing loss and other metrics
        """
        # Unpack batch
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        old_logprobs = batch["old_logprobs"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        prompt_lens = batch["prompt_lens"]
        
        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits and hidden states
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
        
        # If hidden states not available from output, use the model's last layer
        if hidden_states is None:
            # This is a fallback and depends on the model architecture
            if hasattr(self.model, "transformer"):
                hidden_states = self.model.transformer.h[-1].output
            else:
                # For models where we can't easily get hidden states, use a workaround
                logger.warning("Hidden states not available from model output. Using alternative method.")
                hidden_states = torch.zeros(
                    (input_ids.size(0), input_ids.size(1), self.model.config.hidden_size),
                    device=self.device
                )
        
        # Compute logprobs for actions
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute values
        values = self.value_head(hidden_states).squeeze(-1)
        
        batch_size = input_ids.size(0)
        policy_loss = torch.zeros(batch_size, device=self.device)
        value_loss = torch.zeros(batch_size, device=self.device)
        entropy_loss = torch.zeros(batch_size, device=self.device)
        kl_div = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # Only consider tokens after the prompt
            start_idx = prompt_lens[i]
            if start_idx >= input_ids.size(1) - 1:
                continue
                
            # Get token logprobs for the chosen tokens
            token_logprobs = torch.gather(
                log_probs[i, start_idx-1:-1], 
                1, 
                input_ids[i, start_idx:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute ratio of new vs old policy
            ratio = torch.exp(token_logprobs - old_logprobs[i, :token_logprobs.size(0)])
            
            # Compute surrogate objectives
            surr1 = ratio * advantages[i, :token_logprobs.size(0)]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages[i, :token_logprobs.size(0)]
            
            # Policy loss
            policy_loss[i] = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_pred = values[i, start_idx:start_idx+token_logprobs.size(0)]
            old_values_batch = old_values[i, :token_logprobs.size(0)]
            returns_batch = returns[i, :token_logprobs.size(0)]
            
            # Clipped value loss
            values_clipped = old_values_batch + torch.clamp(
                values_pred - old_values_batch, 
                -self.value_clip_range, 
                self.value_clip_range
            )
            v_loss1 = F.mse_loss(values_pred, returns_batch, reduction='none')
            v_loss2 = F.mse_loss(values_clipped, returns_batch, reduction='none')
            value_loss[i] = torch.max(v_loss1, v_loss2).mean()
            
            # Entropy loss
            probs = torch.exp(log_probs[i, start_idx-1:-1])
            entropy = -(probs * log_probs[i, start_idx-1:-1]).sum(dim=-1).mean()
            entropy_loss[i] = -entropy
            
            # Compute approximate KL divergence
            log_ratio = token_logprobs - old_logprobs[i, :token_logprobs.size(0)]
            kl = (torch.exp(log_ratio) - 1) - log_ratio
            kl_div[i] = kl.mean()
        
        # Compute KL penalty with reference model if available
        ref_kl_div = self.compute_kl_divergence(input_ids, attention_mask, prompt_lens)
        
        # Compute total losses
        policy_loss_mean = policy_loss.mean()
        value_loss_mean = value_loss.mean()
        entropy_loss_mean = entropy_loss.mean()
        kl_div_mean = kl_div.mean()
        ref_kl_div_mean = ref_kl_div.mean()
        
        # Adjust KL coefficient if using adaptive KL
        if self.use_adaptive_kl:
            if kl_div_mean > self.adap_kl_target * 1.5:
                self.kl_coef *= 1.5
            elif kl_div_mean < self.adap_kl_target / 1.5:
                self.kl_coef /= 1.5
                
        # Compute total loss
        loss = (
            policy_loss_mean 
            + self.vf_coef * value_loss_mean 
            + self.entropy_coef * entropy_loss_mean
            + self.kl_coef * ref_kl_div_mean
        )
        
        return {
            "loss": loss,
            "policy_loss": policy_loss_mean,
            "value_loss": value_loss_mean,
            "entropy_loss": entropy_loss_mean,
            "kl_div": kl_div_mean,
            "ref_kl_div": ref_kl_div_mean,
            "kl_coef": torch.tensor(self.kl_coef, device=self.device)
        }
        
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of value predictions
            masks: Tensor of masks (1 if token should be considered, 0 otherwise)
            
        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.size()
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE and returns for each sequence in batch
        for i in range(batch_size):
            last_gae_lam = 0
            
            # Process the sequence in reverse order
            for t in reversed(range(seq_len)):
                # Skip if mask is 0
                if masks[i, t] == 0:
                    continue
                    
                # For the last step, or if next step has mask 0
                if t == seq_len - 1 or masks[i, t+1] == 0:
                    next_value = 0
                else:
                    next_value = values[i, t+1]
                    
                # Compute delta (TD error)
                delta = rewards[i, t] + self.gamma * next_value - values[i, t]
                
                # Compute GAE
                last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
                advantages[i, t] = last_gae_lam
                
                # Compute returns
                returns[i, t] = advantages[i, t] + values[i, t]
                
        # Normalize advantages if configured
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        return advantages, returns
        
    def _prepare_training_data(
        self, 
        experiences: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare experiences for PPO training.
        
        Args:
            experiences: Collected experiences from rollout
            
        Returns:
            Dictionary of tensors ready for PPO training
        """
        # Extract data
        prompts = experiences["prompts"]
        responses = experiences["responses"]
        rewards = experiences["rewards"]
        logprobs = experiences["response_logprobs"]
        
        batch_size = len(prompts)
        max_response_len = max(len(lp) for lp in logprobs)
        
        # Initialize tensors
        old_logprobs_tensor = torch.full(
            (batch_size, max_response_len), 
            -100.0,  # Padding value
            dtype=torch.float, 
            device=self.device
        )
        reward_tensor = torch.zeros(
            (batch_size, max_response_len),
            dtype=torch.float,
            device=self.device
        )
        mask_tensor = torch.zeros(
            (batch_size, max_response_len),
            dtype=torch.float,
            device=self.device
        )
        
        # Fill tensors with data
        for i, (lp, r) in enumerate(zip(logprobs, rewards)):
            # Fill old logprobs
            old_logprobs_tensor[i, :len(lp)] = torch.tensor(lp, device=self.device)
            
            # Fill rewards (only at the end)
            reward_tensor[i, -1] = r
            
            # Fill masks
            mask_tensor[i, :len(lp)] = 1.0
            
        # Tokenize full sequences (prompt + response)
        prompt_response_pairs = [p + " " + r for p, r in zip(prompts, responses)]
        encoded = self.tokenizer(
            prompt_response_pairs,
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
        
        # Compute values for each token
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get last hidden states
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback for models without hidden states output
                # This will only work for certain model architectures
                hidden_states = torch.zeros(
                    (batch_size, encoded.input_ids.size(1), self.model.config.hidden_size),
                    device=self.device
                )
                logger.warning("Hidden states not available in model output")
                
            # Compute values
            values = self.value_head(hidden_states).squeeze(-1)
            
        # Extract response values and pad
        value_tensor = torch.zeros(
            (batch_size, max_response_len),
            dtype=torch.float,
            device=self.device
        )
        
        for i, p_len in enumerate(prompt_lens):
            # Extract value predictions for response tokens
            response_values = values[i, p_len:p_len+len(logprobs[i])]
            value_tensor[i, :len(response_values)] = response_values
            
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(reward_tensor, value_tensor, mask_tensor)
        
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "old_logprobs": old_logprobs_tensor,
            "old_values": value_tensor,
            "rewards": reward_tensor,
            "advantages": advantages,
            "returns": returns,
            "masks": mask_tensor,
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
        Train the model using PPO.
        
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
        num_rollouts = self.config.get("num_rollouts", 64)
        rollout_batch_size = self.config.get("rollout_batch_size", 16)
        
        # Output directory
        output_dir = self.config.get("output_dir", "checkpoints/ppo_model")
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Starting PPO training")
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
                # Collect rollouts
                logger.info(f"Collecting rollouts at step {self.global_step}")
                experiences = self.rollout_collector.collect_ppo_experiences(
                    dataset=dataset,
                    batch_size=rollout_batch_size,
                    num_rollouts=num_rollouts
                )
                
                # Log rollout statistics
                reward_stats = self.compute_reward_stats(experiences["rewards"])
                logger.info(f"Rollout stats: {reward_stats}")
                
                # Prepare experiences for training
                train_data = self._prepare_training_data(experiences)
                
                # Multiple epochs of PPO optimization on the rollout data
                for ppo_epoch in range(self.ppo_epochs):
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
                            
                            # Add reward stats
                            log_stats.update(reward_stats)
                            
                            # Log stats
                            self.log_stats(log_stats)
                            epoch_stats.append(log_stats)
                            
                        # Early stopping based on KL divergence
                        if self.target_kl is not None and loss_dict["kl_div"] > self.target_kl * 1.5:
                            logger.info(f"Early stopping PPO epoch {ppo_epoch} due to high KL divergence: {loss_dict['kl_div']:.4f}")
                            break
                            
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