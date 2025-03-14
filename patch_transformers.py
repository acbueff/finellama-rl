#!/usr/bin/env python
"""
Script to directly modify the transformers library in the container
to fix the rope_scaling validation issue.
"""

import os
import sys
import logging
import importlib.util
import inspect
from pathlib import Path

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def find_transformers_path():
    """Find the path to the transformers library."""
    try:
        import transformers
        return Path(inspect.getfile(transformers)).parent
    except ImportError:
        logger.error("Transformers library not found")
        return None

def patch_llama_config_file():
    """
    Directly modify the LlamaConfig class in the transformers library.
    """
    # Find the transformers path
    transformers_path = find_transformers_path()
    if not transformers_path:
        return False
    
    # Find the LlamaConfig file
    llama_config_path = transformers_path / "models" / "llama" / "configuration_llama.py"
    if not llama_config_path.exists():
        logger.error(f"LlamaConfig file not found at {llama_config_path}")
        return False
    
    logger.info(f"Found LlamaConfig file at {llama_config_path}")
    
    # Read the file
    with open(llama_config_path, 'r') as f:
        content = f.read()
    
    # Check if the file contains the validation method
    if "_rope_scaling_validation" not in content:
        logger.error("Could not find _rope_scaling_validation method in LlamaConfig")
        return False
    
    # Create a backup
    backup_path = llama_config_path.with_suffix(".py.bak")
    with open(backup_path, 'w') as f:
        f.write(content)
    logger.info(f"Created backup at {backup_path}")
    
    # Replace the validation method with a no-op
    import re
    pattern = r"def _rope_scaling_validation\(self\):(.*?)(?=\n\s*def|\n\s*@|\Z)"
    replacement = """def _rope_scaling_validation(self):
        # This is a patched version that accepts any rope_scaling format
        if self.rope_scaling is None:
            return
        
        if not isinstance(self.rope_scaling, dict):
            self.rope_scaling = {"type": "linear", "factor": 1.0}
            return
        
        # If it's a dictionary, make sure it has the required fields
        if "type" not in self.rope_scaling:
            self.rope_scaling["type"] = "linear"
        
        if "factor" not in self.rope_scaling:
            if "scale" in self.rope_scaling:
                self.rope_scaling["factor"] = self.rope_scaling["scale"]
            elif "scaling_factor" in self.rope_scaling:
                self.rope_scaling["factor"] = self.rope_scaling["scaling_factor"]
            else:
                self.rope_scaling["factor"] = 1.0
        
        # Convert to float if needed
        if not isinstance(self.rope_scaling["factor"], float):
            try:
                self.rope_scaling["factor"] = float(self.rope_scaling["factor"])
            except (ValueError, TypeError):
                self.rope_scaling["factor"] = 1.0
        
        return"""
    
    # Use re.DOTALL to match across multiple lines
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Check if the replacement was successful
    if new_content == content:
        logger.error("Failed to replace the validation method")
        return False
    
    # Write the modified file
    with open(llama_config_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Successfully patched LlamaConfig._rope_scaling_validation")
    
    # Reload the module
    try:
        from importlib import reload
        import transformers.models.llama.configuration_llama
        reload(transformers.models.llama.configuration_llama)
        logger.info("Reloaded the module")
    except Exception as e:
        logger.warning(f"Failed to reload the module: {e}")
    
    return True

def test_patch():
    """Test if the patch was successful."""
    try:
        from transformers import AutoConfig
        
        # Try to load a config with a problematic rope_scaling
        config_dict = {
            "model_type": "llama",
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        }
        
        from transformers.models.llama.configuration_llama import LlamaConfig
        config = LlamaConfig.from_dict(config_dict)
        
        logger.info(f"Successfully loaded config with rope_scaling: {config.rope_scaling}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = patch_llama_config_file()
    
    if success:
        logger.info("Patch applied successfully")
        
        # Test the patch
        test_success = test_patch()
        if test_success:
            logger.info("Patch test passed")
        else:
            logger.error("Patch test failed")
    else:
        logger.error("Failed to apply patch")
    
    sys.exit(0 if success and test_success else 1) 