"""
Abstract base class for Reinforcement Learning trainers.
This module defines the common functionality for PPO and GRPO trainers.
"""

import os
import time
import logging
import yaml
import math
import random
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler, PreTrainedTokenizer, PreTrainedModel

from datasets import Dataset
from tqdm.auto import tqdm

from ..models.finemathllama import FineMathLlamaModel
from .reward_model import MathRewardModel
from .rollout import RolloutCollector

logger = logging.getLogger(__name__)

class BaseRLTrainer:
    """
    Abstract base class for RL trainers.
    Provides common utilities for PPO and GRPO trainers.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, FineMathLlamaModel],
        tokenizer: PreTrainedTokenizer,
        reward_model: MathRewardModel,
        config: Dict[str, Any],
        rollout_collector: Optional[RolloutCollector] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the RL trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            reward_model: Reward model for computing rewards
            config: Training configuration
            rollout_collector: Rollout collector for gathering experiences
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up model and tokenizer
        if isinstance(model, FineMathLlamaModel):
            # Extract model from wrapper
            self.model_wrapper = model
            self.model = model.model
        else:
            self.model_wrapper = None
            self.model = model
            
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        
        # Use an existing rollout collector or create a new one
        self.rollout_collector = rollout_collector
        
        # Training parameters
        self.seed = self.config.get("seed", 42)
        self.max_steps = self.config.get("max_steps", 20000)
        self.mini_batch_size = self.config.get("mini_batch_size", 8)
        self.batch_size = self.config.get("batch_size", 128)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 8)
        self.learning_rate = self.config.get("learning_rate", 1.0e-5)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        
        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Training tracking
        self.global_step = 0
        self.best_reward = float('-inf')
        self.train_stats = []
        
        # Set random seeds for reproducibility
        self._set_seed(self.seed)
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create the AdamW optimizer for training."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.999)),
            eps=self.config.get("adam_epsilon", 1e-8)
        )
        
    def _create_lr_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create a learning rate scheduler."""
        warmup_steps = self.config.get("warmup_steps", 0)
        
        if warmup_steps > 0:
            return get_scheduler(
                name=self.config.get("lr_scheduler_type", "linear"),
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.max_steps
            )
        
        return None
    
    def save_checkpoint(
        self,
        output_dir: str,
        step: Optional[int] = None,
        save_optimizer: bool = True
    ):
        """
        Save a checkpoint of the model and training state.
        
        Args:
            output_dir: Directory to save the checkpoint to
            step: Current training step (used in filename)
            save_optimizer: Whether to save optimizer state
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine checkpoint path
        checkpoint_name = f"checkpoint-{step}" if step is not None else "checkpoint-latest"
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        
        # Save model
        if self.model_wrapper:
            self.model_wrapper.save_model(checkpoint_dir)
        else:
            # Save model and tokenizer directly
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
        # Save optimizer and scheduler
        if save_optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            if self.lr_scheduler:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                
        # Save training stats
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            import json
            json.dump({
                "global_step": self.global_step,
                "best_reward": self.best_reward,
                "train_stats": self.train_stats[-100:] if self.train_stats else []
            }, f, indent=2)
            
        # Save configuration
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
            
        logger.info(f"Checkpoint saved successfully")
        
    def load_checkpoint(self, checkpoint_dir: str, load_optimizer: bool = True):
        """
        Load a checkpoint of the model and training state.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
            load_optimizer: Whether to load optimizer state
        """
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load the model
        if self.model_wrapper:
            self.model_wrapper = FineMathLlamaModel.from_pretrained(checkpoint_dir)
            self.model = self.model_wrapper.model
        else:
            # Load model directly
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            self.tokenizer = PreTrainedTokenizer.from_pretrained(checkpoint_dir)
            
        # Move model to device
        self.model.to(self.device)
        
        # Load optimizer and scheduler
        if load_optimizer:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                
            scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
            if os.path.exists(scheduler_path) and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(torch.load(scheduler_path))
                
        # Load training stats
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                import json
                state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.best_reward = state.get("best_reward", float('-inf'))
                self.train_stats = state.get("train_stats", [])
                
        logger.info(f"Checkpoint loaded successfully. Global step: {self.global_step}")
        
    def compute_reward_stats(self, rewards: List[float]) -> Dict[str, float]:
        """
        Compute statistics for rewards.
        
        Args:
            rewards: List of reward values
            
        Returns:
            Dictionary of statistics
        """
        if not rewards:
            return {
                "mean_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "std_reward": 0.0
            }
            
        return {
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "std_reward": float(np.std(rewards))
        }
        
    def log_stats(self, stats: Dict[str, Any], step: int = None):
        """
        Log training statistics.
        
        Args:
            stats: Dictionary of statistics to log
            step: Current training step
        """
        step = step or self.global_step
        
        # Log to console
        log_str = f"Step {step}: "
        log_str += ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in stats.items()])
        logger.info(log_str)
        
        # Store stats
        self.train_stats.append({"step": step, **stats})
        
        # Log to appropriate tracking service if configured
        if self.config.get("wandb_logging", False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(stats, step=step)
            except ImportError:
                logger.warning("wandb not installed, skipping wandb logging")
        
        if self.config.get("tensorboard_dir"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.config.get("tensorboard_dir")
                writer = SummaryWriter(tb_dir)
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(k, v, step)
                writer.close()
            except ImportError:
                logger.warning("tensorboard not installed, skipping tensorboard logging")
                
    def train(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Train the model using reinforcement learning.
        Must be implemented by subclasses.
        
        Args:
            dataset: Dataset containing prompts and references
            
        Returns:
            Dictionary of training statistics
        """
        raise NotImplementedError("Subclasses must implement the train method")
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        num_samples: int = 100,
        prompt_key: str = "prompt",
        reference_key: Optional[str] = "reference"
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            prompt_key: Key for prompts in the dataset
            reference_key: Key for reference answers in the dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Sample from the dataset
        if num_samples < len(eval_dataset):
            eval_samples = eval_dataset.select(
                random.sample(range(len(eval_dataset)), num_samples)
            )
        else:
            eval_samples = eval_dataset
            
        # Extract prompts and references
        prompts = eval_samples[prompt_key]
        references = eval_samples[reference_key] if reference_key and reference_key in eval_samples.features else None
        
        # Generate responses
        responses = []
        for prompt in tqdm(prompts, desc="Generating responses"):
            response = self.rollout_collector.generate_response(
                prompt,
                temperature=0.7,  # Use lower temperature for evaluation
                do_sample=False  # Use greedy decoding for evaluation
            )
            responses.append(response)
            
        # Compute rewards
        rewards, reward_components = self.reward_model.compute_rewards(prompts, responses, references)
        rewards_list = rewards.cpu().numpy().tolist()
        
        # Calculate metrics
        reward_stats = self.compute_reward_stats(rewards_list)
        
        # Calculate component metrics
        component_stats = {}
        for component, values in reward_components.items():
            component_values = values.cpu().numpy().tolist()
            component_stats[f"{component}_mean"] = float(np.mean(component_values))
            component_stats[f"{component}_std"] = float(np.std(component_values))
            
        # Combine all metrics
        metrics = {
            "num_samples": len(prompts),
            **reward_stats,
            **component_stats
        }
        
        return metrics
    
    def create_evaluation_samples(
        self,
        eval_dataset: Dataset,
        output_file: str,
        num_samples: int = 20,
        prompt_key: str = "prompt",
        reference_key: Optional[str] = "reference"
    ):
        """
        Create evaluation samples for qualitative assessment.
        Generates responses using both greedy and sampling approaches.
        
        Args:
            eval_dataset: Dataset to evaluate on
            output_file: File to save the samples to
            num_samples: Number of samples to evaluate
            prompt_key: Key for prompts in the dataset
            reference_key: Key for reference answers in the dataset
        """
        # Sample from the dataset
        if num_samples < len(eval_dataset):
            eval_samples = eval_dataset.select(
                random.sample(range(len(eval_dataset)), num_samples)
            )
        else:
            eval_samples = eval_dataset
            
        # Extract prompts and references
        prompts = eval_samples[prompt_key]
        references = eval_samples[reference_key] if reference_key and reference_key in eval_samples.features else None
        
        samples = []
        for i, prompt in enumerate(tqdm(prompts, desc="Generating evaluation samples")):
            # Generate greedy response
            greedy_response = self.rollout_collector.generate_response(
                prompt,
                temperature=1.0,
                do_sample=False
            )
            
            # Generate sampled response
            sampled_response = self.rollout_collector.generate_response(
                prompt,
                temperature=0.7,
                do_sample=True
            )
            
            sample = {
                "id": i,
                "prompt": prompt,
                "greedy_response": greedy_response,
                "sampled_response": sampled_response
            }
            
            if references:
                sample["reference"] = references[i]
                
            samples.append(sample)
            
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            import json
            json.dump(samples, f, indent=2)
            
        logger.info(f"Saved {len(samples)} evaluation samples to {output_file}")
        
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch of data.
        Must be implemented by subclasses.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary containing loss and other metrics
        """
        raise NotImplementedError("Subclasses must implement the compute_loss method") 