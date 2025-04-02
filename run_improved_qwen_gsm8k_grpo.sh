#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project/account name
#SBATCH --gpus=1 -C "fat"          # Request 1 GPU on fat partition for 80GB VRAM
#SBATCH --mem=0                     # Request all available memory on the node
#SBATCH -t 0-72:00:00              # Increase maximum runtime to 72 hours
#SBATCH -J improved_grpo         # Job name
#SBATCH -o logs/improved_grpo_%j.out  # Standard output log
#SBATCH -e logs/improved_grpo_%j.err  # Standard error log

# Source environment variables if .env file exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. HF_TOKEN may not be set."
    echo "This might cause authentication issues with the Hugging Face API."
fi

# Set HuggingFace environment variables to use storage appropriately
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Increase the generation timeout to prevent the futures from timing out
export GENERATION_TIMEOUT_SECONDS=180  # Set to exactly 3 minutes (180 seconds) per generation

# Create all output directories with detailed structure
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned_improved/checkpoints
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned_improved/progress
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo_improved/logs
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo_improved/tensorboard
mkdir -p logs

# Create more detailed temporary config file with better optimization settings
TEMP_CONFIG="/tmp/qwen_gsm8k_grpo_config_improved.yaml"
cp configs/qwen_gsm8k_grpo_config.yaml $TEMP_CONFIG

# Set up custom monitor with improved error detection
echo "Setting up enhanced monitoring..."
(
  MONITOR_LOG="logs/improved_grpo_monitoring_$(date +%Y%m%d).log"
  echo "Starting enhanced monitoring at $(date)" > $MONITOR_LOG
  echo "Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME" >> $MONITOR_LOG
  echo "---------------------------------------" >> $MONITOR_LOG
  
  # Function to get process info
  get_process_info() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
      echo "Process $pid stats:" >> $MONITOR_LOG
      ps -o pid,ppid,pcpu,pmem,vsz,rss,time,etime,cmd -p $pid >> $MONITOR_LOG
      
      # Get memory info from /proc
      if [ -d "/proc/$pid" ]; then
        echo "Memory details:" >> $MONITOR_LOG
        grep -E "VmSize|VmRSS|VmSwap" /proc/$pid/status 2>/dev/null >> $MONITOR_LOG
      fi
      
      # Get open files count
      echo "Open files: $(lsof -p $pid 2>/dev/null | wc -l)" >> $MONITOR_LOG
    fi
  }
  
  # Function to check progress metrics
  check_progress_metrics() {
    local progress_file="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned_improved/progress/training_progress.json"
    if [ -f "$progress_file" ]; then
      echo "Progress file contents:" >> $MONITOR_LOG
      cat $progress_file >> $MONITOR_LOG
      
      # Extract step from progress file
      STEP=$(grep -o '"step": [0-9]*' $progress_file | awk '{print $2}')
      if [ -n "$STEP" ] && [ "$STEP" -gt "0" ]; then
        echo "Training has progressed to step $STEP" >> $MONITOR_LOG
      else
        echo "Warning: Training is still at step 0" >> $MONITOR_LOG
      fi
    else
      echo "Progress file not found: $progress_file" >> $MONITOR_LOG
    fi
  }
  
  # Check for checkpoints
  check_checkpoints() {
    local checkpoint_dir="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned_improved/checkpoints"
    echo "Checkpoint directory content:" >> $MONITOR_LOG
    ls -la $checkpoint_dir 2>/dev/null >> $MONITOR_LOG || echo "No checkpoints found" >> $MONITOR_LOG
  }
  
  # Function to check log file
  check_log_progress() {
    local logfile="logs/improved_grpo_${SLURM_JOB_ID}.out"
    if [ -f "$logfile" ]; then
      echo "Recent log entries:" >> $MONITOR_LOG
      tail -n 10 $logfile >> $MONITOR_LOG
      
      # Check for errors and progress
      ERROR_COUNT=$(grep -c "ERROR" $logfile)
      TIMEOUT_COUNT=$(grep -c "timeout" $logfile)
      PROGRESS_COUNT=$(grep -c "global step" $logfile)
      GENERATION_COUNT=$(grep -c "Generated .* pairs" $logfile)
      
      echo "Log stats - Errors: $ERROR_COUNT, Timeouts: $TIMEOUT_COUNT, Progress markers: $PROGRESS_COUNT, Generations: $GENERATION_COUNT" >> $MONITOR_LOG
      
      # Check if we seem to be making progress
      LAST_STEP_LOG=$(grep "global step" $logfile | tail -n 1)
      echo "Latest step info: $LAST_STEP_LOG" >> $MONITOR_LOG
    else
      echo "Log file not found: $logfile" >> $MONITOR_LOG
    fi
  }
  
  # Monitor the job every 3 minutes
  while true; do
    echo "===== $(date) =====" >> $MONITOR_LOG
    
    # GPU info
    nvidia-smi >> $MONITOR_LOG
    
    # Find python process
    PYTHON_PID=$(pgrep -f "python.*qwen_gsm8k_grpo_train.py")
    if [ -n "$PYTHON_PID" ]; then
      get_process_info $PYTHON_PID
    else
      echo "GRPO training process not found!" >> $MONITOR_LOG
    fi
    
    # Check progress metrics
    check_progress_metrics
    
    # Check for checkpoints
    check_checkpoints
    
    # Check log progress
    check_log_progress
    
    echo "---" >> $MONITOR_LOG
    sleep 180  # Monitor every 3 minutes
  done
) &
MONITOR_PID=$!

# Define path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Define paths
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
GRPO_CONFIG=$TEMP_CONFIG  # Use our improved config
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned_improved"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo_improved"

echo "Starting improved GRPO training for Qwen2.5 3B on GSM8K"
echo "Model config: $MODEL_CONFIG"
echo "GRPO config: $GRPO_CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Evaluation data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Using container: $APPTAINER_ENV"

# Ensure the script is executable
chmod +x qwen_gsm8k_grpo_train.py

# Run the training script within the Apptainer container with monitoring
echo "Running training script inside container..."
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  --env PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="${HF_HOME}" \
  --env GENERATION_TIMEOUT_SECONDS="${GENERATION_TIMEOUT_SECONDS}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  ${APPTAINER_ENV} python -u qwen_gsm8k_grpo_train.py \
  --model_config $MODEL_CONFIG \
  --grpo_config $GRPO_CONFIG \
  --train_data $TRAIN_DATA \
  --eval_data $EVAL_DATA \
  --output_dir $OUTPUT_DIR \
  --results_dir $RESULTS_DIR \
  --eval_steps 200 \
  --eval_examples 25 \
  --seed 42

TRAINING_EXIT_CODE=$?

# Stop monitoring script
kill $MONITOR_PID

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "GRPO training and evaluation completed successfully"
else
  echo "GRPO training exited with code $TRAINING_EXIT_CODE"
  
  # Create failure notification
  echo "Creating failure notification"
  mkdir -p "${OUTPUT_DIR}/errors"
  echo "Training failed with exit code ${TRAINING_EXIT_CODE}" > "${OUTPUT_DIR}/errors/failure_$(date +%Y%m%d_%H%M%S).txt"
fi

echo "Model saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR"

# Cleanup temporary config
rm -f $TEMP_CONFIG 