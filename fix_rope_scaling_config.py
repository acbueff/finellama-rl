#!/usr/bin/env python
"""
Script to directly fix the rope_scaling configuration in the cached config.json file.
This allows debugging the model without having to modify it during loading.
"""

import os
import argparse
import json
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fix rope_scaling in config.json")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="HuggingFaceTB/FineMath-Llama-3B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache",
        help="Directory to use for model cache"
    )
    return parser.parse_args()

def find_config_file(cache_dir, model_id):
    """Find the config.json file in the cache directory."""
    # Convert model_id to the format used in cache directories (replace / with --)
    model_dir_name = model_id.replace("/", "--")
    
    # Base directory for the model
    model_base_dir = os.path.join(cache_dir, f"models--{model_dir_name}")
    
    if not os.path.exists(model_base_dir):
        logger.error(f"Model directory not found: {model_base_dir}")
        return None
    
    # Check snapshots directory
    snapshots_dir = os.path.join(model_base_dir, "snapshots")
    if os.path.exists(snapshots_dir):
        # Get the latest snapshot (likely only one)
        snapshot_dirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
        if snapshot_dirs:
            # Sort by modification time (latest first)
            snapshot_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Check config.json in the latest snapshot
            config_path = os.path.join(snapshot_dirs[0], "config.json")
            if os.path.exists(config_path):
                return config_path
    
    # Try refs directory
    refs_dir = os.path.join(model_base_dir, "refs")
    if os.path.exists(refs_dir):
        for ref_name in os.listdir(refs_dir):
            ref_file = os.path.join(refs_dir, ref_name)
            if os.path.isfile(ref_file):
                with open(ref_file, 'r') as f:
                    ref_hash = f.read().strip()
                
                # Check if config exists in this snapshot
                snapshot_dir = os.path.join(snapshots_dir, ref_hash)
                if os.path.exists(snapshot_dir):
                    config_path = os.path.join(snapshot_dir, "config.json")
                    if os.path.exists(config_path):
                        return config_path
    
    # Last resort - search the model_base_dir
    for root, _, files in os.walk(model_base_dir):
        if "config.json" in files:
            return os.path.join(root, "config.json")
    
    logger.error(f"Could not find config.json for {model_id}")
    return None

def fix_rope_scaling(config_path):
    """Fix the rope_scaling configuration in the config file."""
    logger.info(f"Fixing rope_scaling in config file: {config_path}")
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if rope_scaling exists
    if "rope_scaling" not in config:
        logger.warning("No rope_scaling found in config")
        return False
    
    # Get the original rope_scaling
    original_rope_scaling = config["rope_scaling"]
    logger.info(f"Original rope_scaling: {original_rope_scaling}")
    
    # We know from the error message that the issue is:
    # {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
    # We need to simplify it to {'type': 'linear', 'factor': 32.0}
    
    # Create the fixed rope_scaling
    if isinstance(original_rope_scaling, dict):
        # Extract the factor if it exists, otherwise use a default value
        factor = original_rope_scaling.get("factor", 32.0)
        
        # Create the simplified rope_scaling
        fixed_rope_scaling = {
            "type": "linear",
            "factor": factor
        }
        
        # Update the config
        config["rope_scaling"] = fixed_rope_scaling
        logger.info(f"Fixed rope_scaling: {fixed_rope_scaling}")
        
        # Save the updated config
        # First create a backup
        backup_path = config_path + ".bak"
        logger.info(f"Creating backup of original config: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save the modified config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated config saved to: {config_path}")
        return True
    else:
        logger.warning(f"rope_scaling is not a dictionary: {original_rope_scaling}")
        return False

def main():
    args = parse_args()
    
    # Set environment variables for cache
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir
    
    logger.info(f"Finding config for model: {args.model_id}")
    
    # Find the config file
    config_path = find_config_file(args.cache_dir, args.model_id)
    
    if config_path:
        logger.info(f"Found config file: {config_path}")
        
        # Fix the rope_scaling
        success = fix_rope_scaling(config_path)
        
        if success:
            logger.info("Successfully fixed rope_scaling in config")
            logger.info("You can now run the slurm job to evaluate the model without rope_scaling issues.")
        else:
            logger.error("Failed to fix rope_scaling")
    else:
        logger.error(f"Could not find config file for {args.model_id}")
        logger.info("Make sure the model has been downloaded first using download_model_only.py")

if __name__ == "__main__":
    main() 