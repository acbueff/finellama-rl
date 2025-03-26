#!/bin/bash
# Simplified script to evaluate Qwen2.5 on GSM8K

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Create output directories
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/{baseline,ppo,grpo,comparison}
mkdir -p logs

# Define paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
BASELINE_RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/baseline"

echo "Checking if GSM8K data exists..."
if [ -f "$GSM8K_DATA_PATH" ]; then
    echo "GSM8K data found at $GSM8K_DATA_PATH"
else
    echo "GSM8K data not found at $GSM8K_DATA_PATH. This file is required."
    exit 1
fi

# Evaluate the baseline Qwen model
echo "Evaluating baseline Qwen2.5 model on GSM8K..."
apptainer exec --nv --env HF_TOKEN=$HF_TOKEN ${APPTAINER_ENV} python custom_gsm8k_eval.py \
  --model_id $MODEL_ID \
  --gsm8k_path $GSM8K_DATA_PATH \
  --output_dir $BASELINE_RESULTS_PATH \
  --num_examples 50 \
  --max_tokens 256 \
  --batch_size 8 \
  --use_4bit

echo "Evaluation completed. Results saved to $BASELINE_RESULTS_PATH"
echo "Check the following files for results:"
echo "  - $BASELINE_RESULTS_PATH/gsm8k_metrics.json"
echo "  - $BASELINE_RESULTS_PATH/evaluation_summary.json" 