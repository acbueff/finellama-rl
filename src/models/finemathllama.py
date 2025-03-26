"""
FineMath model implementation.
"""

import os
import logging
from typing import Dict, Any, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

class FineMathLlamaModel:
    """
    Wrapper for the FineMath model (now uses Qwen2.5 instead of LLaMA).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model wrapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Load the model and tokenizer.
        """
        model_name = self.config.get("model_name_or_path", "SunnyLin/Qwen2.5-7B-DPO-VP")
        use_peft = self.config.get("use_peft", True)
        load_in_4bit = self.config.get("load_in_4bit", True)
        trust_remote_code = self.config.get("trust_remote_code", True)
        
        logger.info(f"Loading model {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=self.config.get("use_fast_tokenizer", True),
            trust_remote_code=trust_remote_code
        )
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {}
        if load_in_4bit:
            # Configure 4-bit quantization
            from transformers import BitsAndBytesConfig
            
            model_kwargs["load_in_4bit"] = True
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"
            
        # Add trust_remote_code for Qwen models
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            
        # Load with appropriate dtype
        model_kwargs["torch_dtype"] = torch.bfloat16 if self.config.get("bf16", True) else torch.float16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Apply PEFT if required
        if use_peft:
            self._apply_peft()
            
        logger.info(f"Model loaded successfully")
        
    def _apply_peft(self):
        """Apply LoRA adapters to the model."""
        if not self.model:
            logger.error("Cannot apply PEFT: Model not loaded")
            return
            
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            logger.info("Applying LoRA PEFT to the model")
            
            # Get PEFT config from main config
            peft_config_dict = self.config.get("peft_config", {})
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=peft_config_dict.get("r", 16),
                lora_alpha=peft_config_dict.get("lora_alpha", 32),
                lora_dropout=peft_config_dict.get("lora_dropout", 0.05),
                task_type="CAUSAL_LM",
                target_modules=peft_config_dict.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
                bias=peft_config_dict.get("bias", "none")
            )
            
            # Prepare model for k-bit training if using quantization
            if self.config.get("load_in_4bit", True) or self.config.get("load_in_8bit", False):
                self.model = prepare_model_for_kbit_training(self.model)
                
            # Apply PEFT
            self.model = get_peft_model(self.model, lora_config)
            logger.info(f"LoRA adapter applied to model")
            
        except ImportError:
            logger.error("PEFT or bitsandbytes not installed. Cannot apply PEFT.")
        except Exception as e:
            logger.error(f"Error applying PEFT: {str(e)}")
        
    @classmethod
    def from_pretrained(cls, model_path: str):
        """
        Load a model from a pretrained checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Initialized model wrapper
        """
        instance = cls()
        
        # Check if path is a Hugging Face model ID or a local path
        is_local_path = os.path.exists(model_path)
        trust_remote_code = True  # Default to True for Qwen models
        
        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Ensure tokenizer has padding token
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        
        # Load model (with 4-bit quantization for efficiency)
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        # Add quantization config if not a local path
        if not is_local_path:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Extract config from loaded files
        config_path = os.path.join(model_path, "config.yaml") if is_local_path else None
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                instance.config = yaml.safe_load(f)
        else:
            instance.config = {
                "model_name_or_path": model_path,
                "trust_remote_code": trust_remote_code
            }
            
        return instance
        
    def save_model(self, output_dir: str):
        """
        Save the model and tokenizer.
        
        Args:
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        import yaml
        with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
            
        logger.info(f"Model saved to {output_dir}") 