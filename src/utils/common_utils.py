import os
import random
import json
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
import shutil
import datetime


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path where to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """Get the device to use for computation.
    
    Returns:
        torch.device: The device to use (CUDA if available, otherwise CPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """Create a directory for an experiment with a timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path where to save the metrics
    """
    # Convert any tensor values to Python types
    converted_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            converted_metrics[k] = v.item() if v.numel() == 1 else v.detach().cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            converted_metrics[k] = v.tolist()
        else:
            converted_metrics[k] = v
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(converted_metrics, f, indent=2)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load evaluation metrics from a JSON file.
    
    Args:
        metrics_path: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    save_dir: str,
    step: int,
    additional_data: Optional[Dict[str, Any]] = None
):
    """Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler
        save_dir: Directory where to save the checkpoint
        step: Current step number
        additional_data: Any additional data to include in the checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
    
    checkpoint = {
        'model': model.state_dict(),
        'step': step
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
        
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
        
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
    shutil.copy(checkpoint_path, latest_path)


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """Load a model checkpoint.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        optimizer: The optimizer to load state into (optional)
        scheduler: The learning rate scheduler to load state into (optional)
        map_location: Device mapping for loading the checkpoint
        
    Returns:
        Dictionary with additional data from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Return any additional data
    additional_data = {k: v for k, v in checkpoint.items() 
                      if k not in ['model', 'optimizer', 'scheduler']}
    return additional_data


def format_time(seconds: float) -> str:
    """Format a time duration in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 2m 34s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def compute_parameter_norm(model: torch.nn.Module) -> float:
    """Compute the norm of the parameters of a model.
    
    Args:
        model: The model
        
    Returns:
        L2 norm of the parameters
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the norm of the gradients of a model.
    
    Args:
        model: The model
        
    Returns:
        L2 norm of the gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def flatten_dict(nested_dict: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten a nested dictionary.
    
    Args:
        nested_dict: Nested dictionary to flatten
        prefix: Prefix for the keys
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    for key, value in nested_dict.items():
        new_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key))
        else:
            flattened[new_key] = value
            
    return flattened


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch of data to the specified device.
    
    Args:
        batch: Dictionary containing tensors or other data
        device: Device to move the data to
        
    Returns:
        Batch with all tensors moved to the device
    """
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, dict):
            result[k] = batch_to_device(v, device)
        elif isinstance(v, list):
            if v and isinstance(v[0], torch.Tensor):
                result[k] = [x.to(device) for x in v]
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def ensure_dir(directory: str):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True) 