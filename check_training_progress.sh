#!/bin/bash
# Script to check the progress of the PPO and GRPO training

# Check job status
echo "Checking SLURM job status..."
squeue -u $USER

# Check logs
echo -e "\nChecking training logs..."
tail -n 20 logs/rl_train_*.out 2>/dev/null || echo "No output logs found"
grep -i error logs/rl_train_*.err 2>/dev/null || echo "No errors found in logs"

# Check Python module setup
echo -e "\nTesting Python module imports..."
# Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH=$PYTHONPATH"

# Test importing from our src module
echo "Testing import of src module:"
python -c "import sys; print(sys.path); import src; print('src module found!'); import src.models.finemathllama; print('finemathllama module found!')" 2>&1

# Check PPO progress
echo -e "\nChecking PPO training progress..."
if [ -f "logs/ppo_training.log" ]; then
    echo "Last 10 lines of PPO training log:"
    tail -n 10 logs/ppo_training.log
else
    echo "PPO training log not found yet."
fi

# Check GRPO progress
echo -e "\nChecking GRPO training progress..."
if [ -f "logs/grpo_training.log" ]; then
    echo "Last 10 lines of GRPO training log:"
    tail -n 10 logs/grpo_training.log
else
    echo "GRPO training log not found yet."
fi

# Check if results are available
echo -e "\nChecking for available results..."
PPO_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/ppo"
GRPO_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo"
COMPARISON_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison"

if [ -d "$PPO_RESULTS" ]; then
    echo "PPO results directory exists:"
    ls -la $PPO_RESULTS
    if [ -f "$PPO_RESULTS/evaluation_summary.json" ]; then
        echo -e "\nPPO evaluation summary:"
        cat $PPO_RESULTS/evaluation_summary.json | jq '.gsm8k.final_answer_accuracy' || echo "Could not parse JSON"
    fi
else
    echo "PPO results directory not found yet."
fi

if [ -d "$GRPO_RESULTS" ]; then
    echo -e "\nGRPO results directory exists:"
    ls -la $GRPO_RESULTS
    if [ -f "$GRPO_RESULTS/evaluation_summary.json" ]; then
        echo -e "\nGRPO evaluation summary:"
        cat $GRPO_RESULTS/evaluation_summary.json | jq '.gsm8k.final_answer_accuracy' || echo "Could not parse JSON"
    fi
else
    echo "GRPO results directory not found yet."
fi

if [ -d "$COMPARISON_RESULTS" ]; then
    echo -e "\nComparison results available:"
    ls -la $COMPARISON_RESULTS
else
    echo "Comparison results directory not found yet."
fi

echo -e "\nDone checking progress." 