#!/bin/bash
#SBATCH -A berzelius-2024-329         # Project name and partition
#SBATCH --gpus=1 -C "fat"             # 1 node with GPU
#SBATCH -t 0-04:00:00                 # Maximum runtime of 4 hours
#SBATCH -J download_model             # Job name
#SBATCH -o logs/download_model_%j.out # Standard output log
#SBATCH -e logs/download_model_%j.err # Standard error log

# Source the .env file if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. HF_TOKEN may not be set." 
    echo "This might cause authentication issues with the Hugging Face API."
fi

# Set HuggingFace environment variables to use NOBACKUP storage
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
# Token should be set from .env file now

# Create the models directory if it doesn't exist
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models

# Define the path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Execute the download script within the Apptainer environment
apptainer exec --nv ${APPTAINER_ENV} python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Define the model storage path
model_dir = '/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/finemath-llama-3b'

print('Downloading FineMath-Llama-3B model and tokenizer...')
# Define the model name/path on HuggingFace
model_name = 'EleutherAI/finemath-llama-3b'  # Update this with the correct HF model name

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_dir)

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_dir)

print('Model and tokenizer have been downloaded and saved to:', model_dir)
"

echo "Model download completed." 