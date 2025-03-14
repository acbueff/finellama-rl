#!/usr/bin/env python
"""
Wrapper script to apply the LlamaConfig patch and run the baseline evaluation.
"""

import os
import sys
import torch
import logging
from importlib.util import spec_from_file_location, module_from_spec

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the script.
    """
    # First, apply the patch to the LlamaConfig class
    logger.info("Applying the LlamaConfig patch...")
    
    # Import and run the patch
    patch_path = os.path.join(os.getcwd(), "patch_llama_config.py")
    spec = spec_from_file_location("patch_llama_config", patch_path)
    patch_module = module_from_spec(spec)
    spec.loader.exec_module(patch_module)
    
    # Apply the patch
    success = patch_module.patch_llama_config()
    
    if not success:
        logger.error("Failed to apply the patch. Exiting.")
        sys.exit(1)
    
    logger.info("Patch applied successfully.")
    
    # Import and run the baseline evaluation script with the command-line arguments
    module_name = "scripts.run_baseline_eval"
    
    try:
        logger.info(f"Importing {module_name}...")
        from scripts.run_baseline_eval import main as eval_main
        
        # Extract command-line arguments for the evaluation script
        cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
        
        logger.info(f"Running evaluation with args: {cmd_args}")
        eval_main(cmd_args)
        
    except ImportError as e:
        logger.error(f"Failed to import {module_name}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        sys.exit(1)
        
    logger.info("Evaluation completed successfully.")

if __name__ == "__main__":
    main() 