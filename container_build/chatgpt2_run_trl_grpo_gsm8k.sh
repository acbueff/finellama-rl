#!/bin/bash
#SBATCH --account=berzelius-2025-74         # Project/account
#SBATCH --gpus=1 -C "fat"                   # 1 GPU on fat (80 GB VRAM)
#SBATCH --mem=0                             # All available node memory
#SBATCH -t 0-72:00:00                       # 72 h walltime
#SBATCH -J trl_grpo_gsm8k                   # Job name
#SBATCH -o logs/trl_grpo_gsm8k_%j.out       # STDOUT
#SBATCH -e logs/trl_grpo_gsm8k_%j.err       # STDERR

set -euo pipefail

#### 1) Environment variables ####
# --- Use Project Storage for Caches ---
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="$HF_DATASETS_CACHE"
export TRANSFORMERS_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="$(cat ~/.huggingface_token 2>/dev/null || echo "")"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

#### 2) Paths & configs ####
# --- Use Absolute Paths ---
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
MODEL_CONFIG="configs/qwen2.5_3b_config.yaml"
BASE_GRPO_CONFIG="configs/qwen_gsm8k_grpo_config.yaml"
TEMP_CONFIG="/tmp/trl_grpo_config_optimized_$(date +%s).yaml" # Unique temp file
TRAIN_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_train.json"
EVAL_DATA="/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k/gsm8k_val.json"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/trl_grpo_finetuned"
RESULTS_DIR="/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/trl_grpo"
TRL_SOURCE_PATH="$PWD/grpo-libs/trl-main" # Path to your local TRL source

#### 3) Populate host TRL source into /tmp ####
# Apptainer bind‐mounts host /tmp by default, so copy your local checkout there:
echo "→ Copying TRL source from $TRL_SOURCE_PATH into /tmp/trl-source for container…"
if [ ! -d "$TRL_SOURCE_PATH" ]; then
    echo "ERROR: TRL source directory not found at $TRL_SOURCE_PATH" >&2
    exit 1
fi
rm -rf /tmp/trl-source
cp -r "$TRL_SOURCE_PATH" /tmp/trl-source
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to copy TRL source to /tmp" >&2
    exit 1
fi
echo "→ TRL source copied successfully."

#### 4) Pre‐flight import check ####
echo "→ Performing pre-flight check inside container..."
apptainer exec --nv "$APPTAINER_ENV" python - <<'EOF'
import sys
import os
print(f"Checking existence of /tmp/trl-source: {os.path.exists('/tmp/trl-source')}")
print(f"Content of /tmp/trl-source: {os.listdir('/tmp/trl-source')[:5]}...") # List first 5 items
# verify trl is importable now that /tmp/trl-source is populated
try:
    import trl
    from trl import GRPOConfig
    print(f"✔ TRL import test passed: {trl.__version__} from {trl.__file__}")
except Exception as e:
    print(f"✘ TRL import test failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF
PREFLIGHT_EXIT_CODE=$?
if [ $PREFLIGHT_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Pre-flight check failed. Exiting." >&2
    exit 1
fi
echo "→ Pre-flight check passed."


#### 5) Prepare output dirs ####
mkdir -p logs \
         "$OUTPUT_DIR"/{checkpoints,progress} \
         "$RESULTS_DIR"/{logs,tensorboard}

#### 6) Build optimized GRPO config ####
cp "$BASE_GRPO_CONFIG" "$TEMP_CONFIG"
cat <<EOF >>"$TEMP_CONFIG"

# === appended by Slurm script ===
loss_type: "dr_grpo"
mask_truncated_completions: true
disable_reward_scaling: true
ds3_gather_for_generation: false
sync_ref_model: true
ref_model_sync_steps: 100
ref_model_mixup_alpha: 0.6
use_liger_loss: false
kl_penalty_weight: 0.05
epsilon: 0.2
epsilon_high: 0.28
reward_weights: [1.0, 0.5, 0.2]
EOF
echo "→ Optimized config written to $TEMP_CONFIG"

#### 7) Run training ####
echo "→ Starting training script execution..."
apptainer exec --nv \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_TOKEN="$HF_TOKEN" \
  --env PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  --env OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  --env MKL_NUM_THREADS="$MKL_NUM_THREADS" \
  --env PYTORCH_NO_CUDA_MEMORY_CACHING="$PYTORCH_NO_CUDA_MEMORY_CACHING" \
  --env PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
  "$APPTAINER_ENV" \
  python -u trl_grpo_gsm8k_train.py \
    --model_config "$MODEL_CONFIG" \
    --grpo_config  "$TEMP_CONFIG" \
    --train_data   "$TRAIN_DATA" \
    --eval_data    "$EVAL_DATA" \
    --output_dir   "$OUTPUT_DIR" \
    --results_dir  "$RESULTS_DIR" \
    --seed         42

EXIT_CODE=$?

#### 8) Post‑run ####
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Training completed successfully"
else
  echo "❌ Training failed (exit code=$EXIT_CODE)"
  mkdir -p "$OUTPUT_DIR/errors"
  echo "$EXIT_CODE" >"$OUTPUT_DIR/errors/failure_$(date +%Y%m%d_%H%M%S).txt"
fi

rm -f "$TEMP_CONFIG"
echo "Model dir: $OUTPUT_DIR"
echo "Results dir: $RESULTS_DIR"
