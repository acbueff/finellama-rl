#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --time=0:30:00
#SBATCH --gpus=1 -C "fat"
#SBATCH -J test_container
#SBATCH -o test_container_%j.out
#SBATCH -e test_container_%j.err

# Environment setup
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"

# Container location
CONTAINER_PATH="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Test script location
TEST_SCRIPT="$HOME/finellama-rl/container_build/test_container.py"

echo "Starting container test at $(date)"
echo "Using container: $CONTAINER_PATH"

# Run the test script inside the container
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  --env HF_HOME="${HF_HOME}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  ${CONTAINER_PATH} python ${TEST_SCRIPT}

# Capture test script exit code
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Container tests PASSED!"
    echo "The container is ready for use with TRL and flash-attention."
else
    echo "Container tests FAILED with exit code: $TEST_EXIT_CODE"
    echo "Please check the logs for details on what failed."
fi

echo "Test completed at $(date)"
exit $TEST_EXIT_CODE
