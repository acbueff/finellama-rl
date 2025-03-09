import os
import sys
import logging
import datetime
import yaml
from typing import Dict, Optional, Any, Union

import wandb
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """A class for logging training and evaluation metrics.
    
    This logger supports both console logging, file logging, TensorBoard, and Weights & Biases.
    
    Attributes:
        log_dir (str): Directory where log files will be stored
        logger (logging.Logger): Python logger instance
        tb_writer (SummaryWriter, optional): TensorBoard writer
        use_wandb (bool): Whether to use Weights & Biases for logging
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the logger.
        
        Args:
            log_dir: Directory where log files will be stored
            experiment_name: Name of the experiment for identifying logs
            use_tensorboard: Whether to use TensorBoard for logging
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name (required if use_wandb is True)
            wandb_entity: W&B entity name (required if use_wandb is True)
            config: Configuration dictionary for tracking
            log_level: Python logging level
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        
        # Set up Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Set up TensorBoard if requested
        self.tb_writer = None
        if use_tensorboard:
            tensorboard_dir = os.path.join(log_dir, 'tensorboard', experiment_name)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Set up Weights & Biases if requested
        self.use_wandb = use_wandb
        if use_wandb:
            if wandb_project is None or wandb_entity is None:
                raise ValueError("wandb_project and wandb_entity are required when use_wandb is True")
            
            # Initialize W&B
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=experiment_name,
                config=config,
                dir=os.path.join(log_dir, 'wandb')
            )
            
            # Save config to disk if provided
            if config:
                config_path = os.path.join(log_dir, f"{experiment_name}_config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
    
    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, torch.Tensor]], 
        step: int,
        prefix: str = ""
    ):
        """Log metrics to all configured logging mechanisms.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Current step number
            prefix: Optional prefix for metric names
        """
        # Convert tensor values to Python floats
        metrics_dict = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics_dict[k] = v.item() if v.numel() == 1 else v.detach().cpu().numpy()
            else:
                metrics_dict[k] = v
        
        # Prefix metric names if requested
        if prefix:
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        
        # Log to Python logger
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics_dict.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)
    
    def log_model(self, model_path: str, step: int):
        """Log model checkpoint.
        
        Args:
            model_path: Path to the saved model checkpoint
            step: Current step number
        """
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"model-{step}", 
                type="model",
                description=f"Model checkpoint at step {step}"
            )
            artifact.add_dir(model_path)
            wandb.log_artifact(artifact)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        if self.tb_writer is not None:
            self.tb_writer.add_hparams(params, {})
        
        if self.use_wandb:
            wandb.config.update(params)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text data.
        
        Args:
            name: Name of the text data
            text: The text to log
            step: Current step number
        """
        self.logger.info(f"Text {name} at step {step}:\n{text}")
        
        if self.tb_writer is not None:
            self.tb_writer.add_text(name, text, step)
        
        if self.use_wandb:
            wandb.log({name: wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def log_comparison(self, model_id: str, problem: str, solution: str, step: int):
        """Log model solution to a problem for comparison.
        
        Args:
            model_id: Identifier for the model (e.g., "baseline", "ppo", "grpo")
            problem: The mathematical problem text
            solution: The model's solution to the problem
            step: Current step number
        """
        # Log to file system
        comparisons_dir = os.path.join(self.log_dir, 'comparisons')
        os.makedirs(comparisons_dir, exist_ok=True)
        
        comparison_file = os.path.join(comparisons_dir, f"{model_id}_step{step}.txt")
        with open(comparison_file, 'a') as f:
            f.write(f"Problem:\n{problem}\n\n")
            f.write(f"Solution ({model_id}):\n{solution}\n\n")
            f.write("-" * 80 + "\n\n")
        
        # Log to W&B
        if self.use_wandb:
            table_key = f"comparisons/{model_id}"
            if table_key not in wandb.run.summary:
                table = wandb.Table(columns=["step", "problem", "solution"])
                table.add_data(step, problem, solution)
                wandb.log({table_key: table})
            else:
                table = wandb.run.summary[table_key]
                table.add_data(step, problem, solution)
                wandb.log({table_key: table})
    
    def close(self):
        """Clean up resources used by the logger."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()


def get_logger(
    config: Dict[str, Any],
    experiment_name: str
) -> Logger:
    """Helper function to create a logger from config.
    
    Args:
        config: Configuration dictionary with logging settings
        experiment_name: Name of the experiment
    
    Returns:
        Configured Logger instance
    """
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', 'logs')
    use_tensorboard = log_config.get('use_tensorboard', True)
    use_wandb = log_config.get('use_wandb', False)
    wandb_project = log_config.get('wandb_project')
    wandb_entity = log_config.get('wandb_entity')
    
    return Logger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        config=config
    ) 