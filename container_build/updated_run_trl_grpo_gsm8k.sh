#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project/account name
#SBATCH --gpus=1 -C "fat"                # Request 1 GPU on fat partition for 80GB VRAM
#SBATCH --mem=0                          # Request all available memory on the node
#SBATCH -t 0-72:00:00                    # Increase maximum runtime to 72 hours
#SBATCH -J trl_grpo_gsm8k                # Job name
#SBATCH -o logs/trl_grpo_gsm8k_%j.out    # Standard output log
#SBATCH -e logs/trl_grpo_gsm8k_%j.err    # Standard error log

# Set HuggingFace environment variables to use storage appropriately
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
# Update PYTHONPATH to include only the current directory
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Aggressive memory optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Create all output directories with detailed structure
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned/checkpoints
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned/progress
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo/logs
mkdir -p /proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo/tensorboard
mkdir -p logs

# Create TRL-optimized config 
TEMP_CONFIG="/tmp/trl_grpo_config_optimized.yaml"
cp configs/qwen_gsm8k_grpo_config.yaml $TEMP_CONFIG

# Add TRL-specific settings to config
cat << EOF >> $TEMP_CONFIG

# TRL GRPO specific settings
loss_type: "dr_grpo"  # Use the recommended Dr. GRPO loss formulation
mask_truncated_completions: true  # Better for stability
disable_reward_scaling: true  # Recommended by Dr. GRPO paper
ds3_gather_for_generation: false  # More memory-efficient, but slower generation
sync_ref_model: true  # Periodic update of reference model 
ref_model_sync_steps: 100  # Sync reference model every 100 steps
ref_model_mixup_alpha: 0.6  # Mix 60% current policy, 40% old reference policy
use_liger_loss: false  # Disable Liger-specific GRPO loss
kl_penalty_weight: 0.05  # Reduced KL penalty weight
epsilon: 0.2  # PPO clipping parameter
epsilon_high: 0.28  # Upper bound for asymmetric clipping (DAPO recommendation)
reward_weights: [1.0, 0.5, 0.2]  # Weights for reward functions [correctness, reasoning, brevity]
EOF

echo "Created TRL-optimized config at $TEMP_CONFIG"

# Define path to the Apptainer (Singularity) environment
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Define paths
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
GRPO_CONFIG=$TEMP_CONFIG  # Use our TRL-optimized config
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo"

echo "Starting TRL GRPO training for Qwen2.5 3B on GSM8K"
echo "Model config: $MODEL_CONFIG"
echo "GRPO config: $GRPO_CONFIG"
echo "Training data: $TRAIN_DATA"
echo "Evaluation data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Using container: $APPTAINER_ENV"

# Make the script executable
chmod +x trl_grpo_gsm8k_train.py

# Create a setup script to install required libraries in the container
cat << EOF > /tmp/setup_trl_libs.sh
#!/bin/bash
# Setup script for installing required TRL libraries inside the container

# Install TRL and its dependencies
pip install --no-cache-dir trl>=0.7.10
pip install --no-cache-dir peft>=0.8.0
pip install --no-cache-dir flash-attn==2.5.5
pip install --no-cache-dir einops
pip install --no-cache-dir triton==2.2.0
pip install --no-cache-dir ninja
pip install --no-cache-dir accelerate>=0.26.0

# Additional libraries for GRPO training
pip install --no-cache-dir rouge-score
pip install --no-cache-dir nltk
pip install --no-cache-dir regex
pip install --no-cache-dir pandas>=2.0.0

echo "All TRL libraries installed successfully"
EOF

chmod +x /tmp/setup_trl_libs.sh

# Set up enhanced monitoring
(
  MONITOR_LOG="logs/trl_grpo_monitoring_$(date +%Y%m%d).log"
  echo "Starting enhanced monitoring at $(date)" > $MONITOR_LOG
  echo "Job ID: $SLURM_JOB_ID, Node: $SLURMD_NODENAME" >> $MONITOR_LOG
  echo "---------------------------------------" >> $MONITOR_LOG
  
  # Monitor the job every 5 minutes
  while true; do
    echo "===== $(date) =====" >> $MONITOR_LOG
    
    # GPU info
    nvidia-smi >> $MONITOR_LOG
    
    # Find python process
    PYTHON_PID=$(pgrep -f "python.*trl_grpo_gsm8k_train.py")
    if [ -n "$PYTHON_PID" ]; then
      echo "Process $PYTHON_PID stats:" >> $MONITOR_LOG
      ps -o pid,ppid,pcpu,pmem,vsz,rss,time,etime,cmd -p $PYTHON_PID >> $MONITOR_LOG
      
      # Memory details
      if [ -d "/proc/$PYTHON_PID" ]; then
        echo "Memory details:" >> $MONITOR_LOG
        grep -E "VmSize|VmRSS|VmSwap" /proc/$PYTHON_PID/status 2>/dev/null >> $MONITOR_LOG
      fi
    else
      echo "TRL GRPO training process not found!" >> $MONITOR_LOG
    fi
    
    # Check log progress
    LOG_FILE="logs/trl_grpo_gsm8k_${SLURM_JOB_ID}.out"
    if [ -f "$LOG_FILE" ]; then
      echo "Recent log entries:" >> $MONITOR_LOG
      tail -n 10 $LOG_FILE >> $MONITOR_LOG
      
      # Check for errors and progress
      ERROR_COUNT=$(grep -c "ERROR" $LOG_FILE)
      TIMEOUT_COUNT=$(grep -c "timeout" $LOG_FILE)
      PROGRESS_COUNT=$(grep -c "step [0-9]" $LOG_FILE)
      
      echo "Log stats - Errors: $ERROR_COUNT, Timeouts: $TIMEOUT_COUNT, Progress markers: $PROGRESS_COUNT" >> $MONITOR_LOG
    fi
    
    # Check for output files
    echo "Output files:" >> $MONITOR_LOG
    ls -la $OUTPUT_DIR/checkpoint-* 2>/dev/null | wc -l >> $MONITOR_LOG
    
    echo "---" >> $MONITOR_LOG
    sleep 300  # Monitor every 5 minutes
  done
) &
MONITOR_PID=$!

# Run the training script using the container with installed libraries
echo "Running training script inside container with installed TRL and flash-attn libraries..."
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  --env PYTORCH_NO_CUDA_MEMORY_CACHING="${PYTORCH_NO_CUDA_MEMORY_CACHING}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="${HF_HOME}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  ${APPTAINER_ENV} bash -c "
    # Run the setup script to install required libraries
    bash /tmp/setup_trl_libs.sh
    
    # Run the training script with all arguments
    python -u trl_grpo_gsm8k_train.py \
      --model_config $MODEL_CONFIG \
      --grpo_config $GRPO_CONFIG \
      --train_data $TRAIN_DATA \
      --eval_data $EVAL_DATA \
      --output_dir $OUTPUT_DIR \
      --results_dir $RESULTS_DIR \
      --seed 42
  "

TRAINING_EXIT_CODE=$?

# Stop monitoring script
kill $MONITOR_PID

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
  echo "TRL GRPO training and evaluation completed successfully"
else
  echo "TRL GRPO training exited with code $TRAINING_EXIT_CODE"
  
  # Create failure notification
  mkdir -p "${OUTPUT_DIR}/errors"
  echo "Training failed with exit code ${TRAINING_EXIT_CODE}" > "${OUTPUT_DIR}/errors/failure_$(date +%Y%m%d_%H%M%S).txt"
fi

echo "Model saved to: $OUTPUT_DIR"
echo "Results saved to: $RESULTS_DIR"

# Cleanup temporary config
rm -f $TEMP_CONFIG 