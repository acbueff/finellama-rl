# Debugging rope_scaling Issues with FineMath-Llama-3B

This document provides instructions for debugging and resolving rope_scaling issues when working with the HuggingFaceTB/FineMath-Llama-3B model on HPC systems.

## The Problem

When trying to load the FineMath-Llama-3B model, you may encounter errors related to the `rope_scaling` configuration parameter. This occurs because:

1. The model may have a `rope_scaling` attribute that is `None`
2. The model may have a `rope_scaling` dictionary missing the required `"type"` or `"factor"` fields
3. The version of transformers being used may have stricter validation for these fields

## Solution Approach

We've created three scripts to help diagnose and fix this issue:

1. `download_model_only.py` - A lightweight script to download the model files to cache without loading the model
2. `test_model_download.py` - A script to test downloading and loading the model with rope_scaling fixes
3. `slurm_gsm8k_eval_improved.sh` - An improved slurm script that incorporates the rope_scaling fix directly

## Instructions

### 1. Download Model Files to Cache (Lightweight)

This is the recommended first step if you want to debug on the login node without using much memory:

```bash
# Set up environment variables
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache" 
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"

# Run download script (minimal memory usage)
python download_model_only.py
```

This script:
- Downloads model files using `snapshot_download` without loading the model
- Downloads tokenizer and config separately
- Checks and reports rope_scaling configuration issues

### 2. Test Model Loading (More Memory-Intensive)

If you want to test loading the model with the rope_scaling fix:

```bash
# Run with 4-bit quantization to reduce memory usage
python test_model_download.py --test_generation false
```

This script:
- Loads the tokenizer
- Loads the config and fixes rope_scaling issues
- Attempts to load the model with the fixed config
- Optionally tests generation (if `--test_generation true`)

**Note:** This requires more memory and may not be suitable for login nodes with limited resources.

### 3. Run the Improved Slurm Job

Once you've verified the model can be downloaded successfully, you can run the improved slurm script:

```bash
sbatch slurm_gsm8k_eval_improved.sh
```

This script:
- Creates a proper GSM8K evaluation config
- Uses a Python wrapper script to monkey-patch the `load_model` method to fix rope_scaling
- Runs the evaluation with the fixed model loading method

### Detailed Fix for rope_scaling Issues

The fix for rope_scaling issues involves:

1. Loading the model config first
2. Checking for rope_scaling issues:
   ```python
   if hasattr(config, "rope_scaling"):
       # Fix if rope_scaling is None
       if config.rope_scaling is None:
           config.rope_scaling = {"type": "linear", "factor": 1.0}
       # Fix if rope_scaling is missing required fields
       elif isinstance(config.rope_scaling, dict):
           if "type" not in config.rope_scaling:
               config.rope_scaling["type"] = "linear"
           if "factor" not in config.rope_scaling:
               config.rope_scaling["factor"] = 1.0
   ```
3. Loading the model with the fixed config

## Troubleshooting

If you still encounter issues:

1. Check for specific error messages in the output log
2. Ensure your HF_HOME and HF_DATASETS_CACHE environment variables are properly set
3. Verify you have read/write permissions for the cache directories
4. Try updating the transformers library to the latest version

## Alternative Approach

If you continue to have issues, you could try another approach:

1. Download the model locally
2. Manually edit the config.json file to fix the rope_scaling issue
3. Upload the modified model to your own HuggingFace space or use locally 