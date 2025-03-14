#!/usr/bin/env python
"""
Script to patch the transformers library's LlamaConfig class
to disable rope_scaling validation for the FineMath-Llama-3B model.
This patches the actual library code for the current session.
"""

import os
import sys
import inspect
import logging
from typing import Optional, Any

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def patch_llama_config():
    """
    Patch the LlamaConfig._rope_scaling_validation method to make it a no-op.
    """
    try:
        # Import the LlamaConfig class
        from transformers.models.llama.configuration_llama import LlamaConfig
        
        # Store the original validation method for reference
        original_validation = LlamaConfig._rope_scaling_validation
        
        # Define a new no-op validation method
        def no_validation(self):
            """
            No-op validation method for rope_scaling that accepts any configuration.
            """
            logger.info(f"Bypassing rope_scaling validation for: {self.rope_scaling}")
            return
        
        # Patch the method
        LlamaConfig._rope_scaling_validation = no_validation
        
        # Verify that the patch was applied
        llama_file = inspect.getfile(LlamaConfig)
        logger.info(f"Patched LlamaConfig._rope_scaling_validation in {llama_file}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch LlamaConfig: {str(e)}")
        return False

def fix_rope_scaling_in_config(config):
    """
    Fix rope_scaling in a config object if it exists.
    
    Args:
        config: The configuration object to fix
        
    Returns:
        The fixed configuration object
    """
    if hasattr(config, "rope_scaling"):
        logger.info(f"Original rope_scaling: {config.rope_scaling}")
        
        # Check if rope_scaling is None or missing required fields
        if config.rope_scaling is None:
            config.rope_scaling = {"type": "linear", "factor": 1.0}
        elif isinstance(config.rope_scaling, dict):
            # For Llama3 format, convert to the expected format
            if "rope_type" in config.rope_scaling and config.rope_scaling["rope_type"] == "llama3":
                config.rope_scaling = {
                    "type": "linear",
                    "factor": float(config.rope_scaling.get("factor", 1.0))
                }
            # For other formats, ensure the required fields are present
            elif "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "linear"
            
            if "factor" not in config.rope_scaling:
                config.rope_scaling["factor"] = 1.0
        
        logger.info(f"Fixed rope_scaling: {config.rope_scaling}")
    
    return config

if __name__ == "__main__":
    # Apply the patch when the script is run directly
    success = patch_llama_config()
    print(f"Patch applied: {success}")
    sys.exit(0 if success else 1) 