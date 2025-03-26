#!/bin/bash
#SBATCH -A berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0-2:00:00
#SBATCH -J compare_models
#SBATCH -o logs/compare_models_%j.out
#SBATCH -e logs/compare_models_%j.err

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Define paths
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
PPO_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/ppo"
GRPO_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo"
BASELINE_RESULTS="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/baseline"
COMPARISON_DIR="/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison"

mkdir -p ${COMPARISON_DIR}

# Run comparison
echo "Comparing baseline, PPO, and GRPO models"
apptainer exec --nv ${APPTAINER_ENV} python - << 'PYTHONSCRIPT'
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load results
def load_metrics(path):
    try:
        with open(os.path.join(path, "gsm8k_metrics.json"), 'r') as f:
            return json.load(f)
    except:
        return {"accuracy": 0, "correct": 0, "total": 0}

baseline = load_metrics("/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/baseline")
ppo = load_metrics("/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/ppo")
grpo = load_metrics("/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/grpo")

# Print comparison
print("=== Model Comparison on GSM8K ===")
print(f"Baseline: {baseline['accuracy']:.2%} ({baseline['correct']}/{baseline['total']})")
print(f"PPO: {ppo['accuracy']:.2%} ({ppo['correct']}/{ppo['total']})")
print(f"GRPO: {grpo['accuracy']:.2%} ({grpo['correct']}/{grpo['total']})")

# Calculate improvement
ppo_improvement = (ppo['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
grpo_improvement = (grpo['accuracy'] - baseline['accuracy']) / baseline['accuracy'] * 100 if baseline['accuracy'] > 0 else 0
print(f"\nPPO improvement over baseline: {ppo_improvement:.2f}%")
print(f"GRPO improvement over baseline: {grpo_improvement:.2f}%")

# Create comparison plot
models = ['Baseline', 'PPO', 'GRPO']
accuracies = [baseline['accuracy'], ppo['accuracy'], grpo['accuracy']]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['gray', 'blue', 'green'])
plt.ylim(0, max(max(accuracies) * 1.2, 0.1))  # Ensure reasonable y-axis even if all accuracies are 0
plt.title('GSM8K Accuracy Comparison')
plt.ylabel('Accuracy')

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}', ha='center', va='bottom')

plt.savefig("/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison/model_comparison.png")

# Save detailed comparison
comparison = {
    "baseline": baseline,
    "ppo": ppo, 
    "grpo": grpo,
    "improvements": {
        "ppo_vs_baseline": ppo_improvement,
        "grpo_vs_baseline": grpo_improvement,
        "ppo_vs_grpo": (ppo['accuracy'] - grpo['accuracy']) / grpo['accuracy'] * 100 if grpo['accuracy'] > 0 else 0
    }
}

with open("/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison/comparison_results.json", 'w') as f:
    json.dump(comparison, f, indent=2)

print(f"Comparison results saved to /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison/comparison_results.json")
print(f"Comparison plot saved to /proj/berzelius-aiics-real/users/x_anbue/ppo_grpo/results/comparison/model_comparison.png")
PYTHONSCRIPT

echo "Comparison completed" 