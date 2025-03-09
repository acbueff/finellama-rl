"""
FineMathLlama model definition and utilities.
This module handles loading, saving, and configuring the FineMath-Llama-3B model.
"""

import os
import yaml
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union

from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    PeftModel, 
    prepare_model_for_kbit_training
)
from accelerate import Accelerator, dispatch_model, init_empty_weights

logger = logging.getLogger(__name__)

class FineMathLlamaModel:
    """
    A wrapper class for the FineMath-Llama-3B model, handling model loading,
    configuration, and generation utilities.
    """
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """
        Initialize the FineMathLlama model.
        
        Args:
            config_path: Path to the model configuration YAML file.
            config: Dictionary containing model configuration. Only used if config_path is None.
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}
            
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.accelerator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
        )
        
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the FineMath-Llama model and tokenizer based on the configuration.
        
        Returns:
            Tuple containing (model, tokenizer)
        """
        model_name = self.config.get("model_name_or_path", "huggyllama/finemath-llama-3b")
        use_auth_token = self.config.get("use_auth_token", False)
        
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=self.config.get("use_fast_tokenizer", True),
            use_auth_token=use_auth_token
        )
        
        # Ensure the tokenizer has pad_token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load the model with appropriate quantization settings
        logger.info(f"Loading model from {model_name}")
        load_in_8bit = self.config.get("load_in_8bit", False)
        load_in_4bit = self.config.get("load_in_4bit", True)
        
        # Set up quantization configuration
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=load_in_8bit,
            quantization_config=quantization_config if load_in_4bit else None,
            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float16,
            device_map="auto",
            use_auth_token=use_auth_token
        )
        
        # Apply PEFT if configured
        if self.config.get("use_peft", True):
            self._apply_peft()
            
        # Set up generation config
        self.model.generation_config.max_length = self.config.get("max_length", 2048)
        self.model.generation_config.max_new_tokens = self.config.get(
            "generation_max_length", 512
        )
        self.model.generation_config.temperature = self.config.get("temperature", 0.7)
        self.model.generation_config.top_p = self.config.get("top_p", 0.9)
        self.model.generation_config.top_k = self.config.get("top_k", 50)
        self.model.generation_config.num_beams = self.config.get("num_beams", 1)
        self.model.generation_config.do_sample = self.config.get("do_sample", True)
        self.model.generation_config.repetition_penalty = self.config.get("repetition_penalty", 1.1)
        
        return self.model, self.tokenizer
    
    def _apply_peft(self):
        """Apply Parameter-Efficient Fine-Tuning (PEFT) to the model."""
        if self.model is None:
            raise ValueError("Model must be loaded before applying PEFT")
            
        peft_config_dict = self.config.get("peft_config", {})
        peft_type = peft_config_dict.get("peft_type", "lora")
        
        if peft_type == "lora":
            logger.info("Applying LoRA PEFT to the model")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=peft_config_dict.get("r", 16),
                lora_alpha=peft_config_dict.get("lora_alpha", 32),
                lora_dropout=peft_config_dict.get("lora_dropout", 0.05),
                target_modules=peft_config_dict.get(
                    "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
                ),
                bias=peft_config_dict.get("bias", "none")
            )
            
            # Prepare the model for kbit training if using quantization
            if self.config.get("load_in_8bit", False) or self.config.get("load_in_4bit", True):
                self.model = prepare_model_for_kbit_training(self.model)
                
            self.model = get_peft_model(self.model, peft_config)
            self.peft_config = peft_config
            
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: A single prompt or a list of prompts
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation")
            
        # Handle single prompt
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.get("generation_max_length", 512),
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
            "top_p": kwargs.get("top_p", self.config.get("top_p", 0.9)),
            "top_k": kwargs.get("top_k", self.config.get("top_k", 50)),
            "num_beams": kwargs.get("num_beams", self.config.get("num_beams", 1)),
            "do_sample": kwargs.get("do_sample", self.config.get("do_sample", True)),
            "repetition_penalty": kwargs.get(
                "repetition_penalty", 
                self.config.get("repetition_penalty", 1.1)
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config.get("max_length", 2048) - gen_kwargs["max_new_tokens"]
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs
            )
            
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Only take the newly generated tokens
            generated_text = self.tokenizer.decode(
                output[inputs.input_ids[i].shape[0]:],
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
            
        return generated_texts
    
    def save_model(self, output_dir: str, save_full_model: bool = False):
        """
        Save the model, tokenizer, and configuration.
        
        Args:
            output_dir: Directory to save the model to
            save_full_model: Whether to save the full model or just the PEFT adapter
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(output_dir, "model_config.yaml"), 'w') as f:
            yaml.dump(self.config, f)
            
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model
        if self.config.get("use_peft", True) and not save_full_model:
            logger.info(f"Saving PEFT adapter to {output_dir}")
            self.model.save_pretrained(output_dir)
        else:
            logger.info(f"Saving full model to {output_dir}")
            # If using PEFT, merge the adapter weights first
            if self.config.get("use_peft", True):
                logger.info("Merging PEFT adapter weights before saving")
                self.model = self.model.merge_and_unload()
            self.model.save_pretrained(output_dir)
            
    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        if self.model is None:
            raise ValueError("Model must be loaded before counting parameters")
            
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def setup_for_training(self):
        """
        Set up the model for training, including gradient checkpointing and optimizer.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before setting up for training")
            
        # Set up gradient checkpointing if enabled
        if self.config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            
        # Set requires grad
        for param in self.model.parameters():
            if param.ndim == 1:  # Skip biases and layer norms
                param.requires_grad = False
                
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            mixed_precision="bf16" if self.config.get("bf16", True) else "fp16" if self.config.get("fp16", False) else "no"
        )
        
        return self.accelerator
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "FineMathLlamaModel":
        """
        Load a model from a saved checkpoint.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            FineMathLlamaModel instance
        """
        config_path = os.path.join(model_path, "model_config.yaml")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
            
        model = cls(config_path=config_path)
        model.load_model()
        return model 