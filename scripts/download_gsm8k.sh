#!/bin/bash
# Script to download GSM8K dataset on CPU
# This script doesn't require Slurm and can be run directly using a container

# Ensure script fails on error
set -e

# Default config
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k"
SPLITS="train,test"
CACHE_DIR="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --splits)
      SPLITS="$2"
      shift 2
      ;;
    --cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --container)
      APPTAINER_ENV="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--output_dir DIR] [--splits SPLITS] [--cache_dir DIR] [--container PATH]"
      exit 1
      ;;
  esac
done

# Make sure required directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

echo "Starting GSM8K dataset download..."
echo "Data will be saved to: $OUTPUT_DIR"
echo "Using HuggingFace cache: $CACHE_DIR"
echo "Using container: $APPTAINER_ENV"

# Check if the container exists
if [ ! -f "$APPTAINER_ENV" ]; then
  echo "Error: Container file not found at $APPTAINER_ENV"
  echo "Please provide correct path with --container option"
  exit 1
fi

# Run the Python script inside the container
apptainer exec --nv "$APPTAINER_ENV" python scripts/download_gsm8k.py \
  --output_dir "$OUTPUT_DIR" \
  --splits "$SPLITS" \
  --cache_dir "$CACHE_DIR"

echo "Download complete. Dataset saved to $OUTPUT_DIR" 