#!/usr/bin/env python
# Auto-generated wrapper script to patch LlamaConfig before running any code

import sys
import os
import importlib.util

# First, make sure our patch script is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import and apply our patch
import patch_llama_config
patch_llama_config.patch_llama_config()

# Now run the original script
if len(sys.argv) < 2:
    print("Usage: wrapper.py <script_to_run> [args...]")
    sys.exit(1)

target_script = sys.argv[1]
target_args = sys.argv[2:]

# Load the target script as a module
spec = importlib.util.spec_from_file_location("target_module", target_script)
if spec is None:
    print(f"Could not load script: {target_script}")
    sys.exit(1)

module = importlib.util.module_from_spec(spec)
sys.argv = [target_script] + target_args
spec.loader.exec_module(module)
