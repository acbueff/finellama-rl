#!/bin/bash
# Script to prepare data and run both PPO and GRPO training/evaluation

# Create necessary directories
echo "Creating directories..."
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/{results/{ppo,grpo},checkpoints/{ppo,grpo},models/{ppo_finetuned,grpo_finetuned}}
mkdir -p logs

# Set up environment variables
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
# Add current directory to PYTHONPATH to find src module
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Set PYTHONPATH to $PYTHONPATH"
# Load Hugging Face token if available
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")
echo "HF_TOKEN is $(if [ -n "$HF_TOKEN" ]; then echo "set"; else echo "not set"; fi)"

# If we have a .env file, source it
if [ -f ".env" ]; then
    source ".env"
    echo "Loaded environment variables from .env file"
fi

# Define paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_ID="SunnyLin/Qwen2.5-7B-DPO-VP"  # Updated to use Qwen2.5 model
DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/data"
GSM8K_DATA_PATH="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_test.json"
PPO_SAVE_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/ppo_finetuned"
PPO_CHECKPOINT_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/checkpoints/ppo"
PPO_RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/ppo"
GRPO_SAVE_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/models/grpo_finetuned"
GRPO_CHECKPOINT_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/checkpoints/grpo"
GRPO_RESULTS_PATH="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo"

# Make sure config files are available in the expected locations
echo "Copying configuration files to necessary locations..."
mkdir -p data/configs
cp configs/model_config.yaml data/configs/

# Step 1: Prepare GSM8K data for training
echo "Preparing GSM8K dataset for training..."
# Use the existing GSM8K data rather than downloading it again
if [ -f "$GSM8K_DATA_PATH" ]; then
    echo "Using existing GSM8K data from $GSM8K_DATA_PATH"
    mkdir -p ${DATA_PATH}/ppo ${DATA_PATH}/grpo
    
    # Copy and process the GSM8K data for PPO and GRPO
    echo "Processing GSM8K data for PPO and GRPO formats..."
    apptainer exec --nv --env PYTHONPATH=$PYTHONPATH ${APPTAINER_ENV} python data/prepare_data.py \
      --datasets $GSM8K_DATA_PATH \
      --output_dir ${DATA_PATH} \
      --process_for all \
      --max_samples 1000 \
      --config data/configs/model_config.yaml
else
    echo "GSM8K data not found at $GSM8K_DATA_PATH"
    echo "Downloading GSM8K data..."
    apptainer exec --nv --env PYTHONPATH=$PYTHONPATH ${APPTAINER_ENV} python data/prepare_data.py \
      --datasets gsm8k \
      --output_dir ${DATA_PATH} \
      --process_for all \
      --max_samples 1000 \
      --config data/configs/model_config.yaml
fi

# Step 2: Run PPO training with Qwen2.5
echo "Starting PPO training with $MODEL_ID..."
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN ${APPTAINER_ENV} python scripts/run_ppo_training.py \
  --config configs/ppo_config.yaml \
  --model_path $MODEL_ID \
  --train_data_path ${DATA_PATH}/ppo/train.json \
  --eval_data_path ${DATA_PATH}/ppo/validation.json \
  --output_dir ${PPO_SAVE_PATH} \
  --seed 42 > logs/ppo_training.log 2>&1 &

echo "PPO training started in background. Check logs/ppo_training.log for progress."
PPO_PID=$!

# Step 3: Run GRPO training with Qwen2.5
echo "Starting GRPO training with $MODEL_ID..."
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN ${APPTAINER_ENV} python scripts/run_grpo_training.py \
  --config configs/grpo_config.yaml \
  --model_path $MODEL_ID \
  --train_data_path ${DATA_PATH}/grpo/train.json \
  --eval_data_path ${DATA_PATH}/grpo/validation.json \
  --output_dir ${GRPO_SAVE_PATH} \
  --seed 42 > logs/grpo_training.log 2>&1 &

echo "GRPO training started in background. Check logs/grpo_training.log for progress."
GRPO_PID=$!

# Wait for training to complete
echo "Waiting for training processes to complete..."
wait $PPO_PID
echo "PPO training completed."
wait $GRPO_PID
echo "GRPO training completed."

# Step 4: Evaluate PPO model on GSM8K using the custom evaluation
echo "Evaluating PPO model on GSM8K..."
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN ${APPTAINER_ENV} python custom_gsm8k_eval.py \
  --model_id ${PPO_SAVE_PATH} \
  --gsm8k_path $GSM8K_DATA_PATH \
  --output_dir ${PPO_RESULTS_PATH} \
  --num_examples 50 \
  --max_tokens 256 \
  --batch_size 8 \
  --use_4bit > logs/ppo_eval.log 2>&1

# Step 5: Evaluate GRPO model on GSM8K using the custom evaluation
echo "Evaluating GRPO model on GSM8K..."
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH --env HF_TOKEN=$HF_TOKEN ${APPTAINER_ENV} python custom_gsm8k_eval.py \
  --model_id ${GRPO_SAVE_PATH} \
  --gsm8k_path $GSM8K_DATA_PATH \
  --output_dir ${GRPO_RESULTS_PATH} \
  --num_examples 50 \
  --max_tokens 256 \
  --batch_size 8 \
  --use_4bit > logs/grpo_eval.log 2>&1

# Step 6: Compare models
echo "Comparing PPO and GRPO models..."
apptainer exec --nv --env PYTHONPATH=$PYTHONPATH ${APPTAINER_ENV} python scripts/compare_methods.py \
  --ppo_results ${PPO_RESULTS_PATH}/evaluation_summary.json \
  --grpo_results ${GRPO_RESULTS_PATH}/evaluation_summary.json \
  --output_dir /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison \
  --dataset gsm8k > logs/comparison.log 2>&1

echo "All processes completed. Check the results in:"
echo "- PPO: ${PPO_RESULTS_PATH}"
echo "- GRPO: ${GRPO_RESULTS_PATH}"
echo "- Comparison: /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison" 