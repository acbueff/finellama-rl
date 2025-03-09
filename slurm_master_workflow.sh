#!/bin/bash
# Master script to orchestrate the entire workflow with job dependencies

# Source the .env file if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. Make sure to set up your environment variables first."
    echo "Run ./setup_env_file.sh to create the .env file with your Hugging Face token."
    exit 1
fi

# Create logs directory for output files
mkdir -p logs

# Submit the dataset download job
echo "Submitting dataset download job..."
DATASET_JOB=$(sbatch slurm_download_datasets.sh | awk '{print $4}')
echo "Dataset job ID: $DATASET_JOB"

# Submit the model download job
echo "Submitting model download job..."
MODEL_JOB=$(sbatch slurm_download_model.sh | awk '{print $4}')
echo "Model job ID: $MODEL_JOB"

# Wait for both dataset and model download to complete before running baseline evaluation
echo "Submitting baseline evaluation job (will wait for downloads to complete)..."
BASELINE_JOB=$(sbatch --dependency=afterok:$DATASET_JOB:$MODEL_JOB slurm_baseline_eval.sh | awk '{print $4}')
echo "Baseline evaluation job ID: $BASELINE_JOB"

# Submit PPO training job (depends on downloads but not baseline)
echo "Submitting PPO training job (will wait for downloads to complete)..."
PPO_JOB=$(sbatch --dependency=afterok:$DATASET_JOB:$MODEL_JOB slurm_ppo_train.sh | awk '{print $4}')
echo "PPO training job ID: $PPO_JOB"

# Submit GRPO training job (depends on downloads but not baseline)
echo "Submitting GRPO training job (will wait for downloads to complete)..."
GRPO_JOB=$(sbatch --dependency=afterok:$DATASET_JOB:$MODEL_JOB slurm_grpo_train.sh | awk '{print $4}')
echo "GRPO training job ID: $GRPO_JOB"

# Submit comparison job (depends on all other jobs)
echo "Submitting method comparison job (will wait for all other jobs to complete)..."
COMPARE_JOB=$(sbatch --dependency=afterok:$BASELINE_JOB:$PPO_JOB:$GRPO_JOB slurm_compare_methods.sh | awk '{print $4}')
echo "Comparison job ID: $COMPARE_JOB"

echo "All jobs have been submitted. Use 'squeue -u $USER' to check status."
echo "Complete workflow scheduled with the following job IDs:"
echo "- Dataset download: $DATASET_JOB"
echo "- Model download: $MODEL_JOB"
echo "- Baseline evaluation: $BASELINE_JOB"
echo "- PPO training: $PPO_JOB"
echo "- GRPO training: $GRPO_JOB"
echo "- Method comparison: $COMPARE_JOB" 