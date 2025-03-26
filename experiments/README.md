# FineLlama-RL: Experiment Results

This directory contains documentation and analysis of training experiments conducted in the FineLlama-RL project.

## Experiment Catalog

### Baseline Models

| Model          | Dataset | Job ID    | Accuracy | Evaluation Date    | Details                                          |
|----------------|---------|-----------|----------|--------------------|-------------------------------------------------|
| Qwen2.5-7B-DPO-VP | GSM8K   | 13222486  | 38.0%    | Mar 16, 2025       | [Baseline Results](baseline_evaluation_results.md) |

### PPO Experiments

| Model          | Dataset | Job ID    | Accuracy | Training Date       | Details                                    |
|----------------|---------|-----------|----------|---------------------|-------------------------------------------|
| Qwen2.5-3B     | GSM8K   | 13232553  | 64.0%    | Mar 18-20, 2025     | [PPO Training Results](ppo_training_results.md) |

### GRPO Experiments

| Model          | Dataset | Job ID    | Status   | Training Date       | Details                                    |
|----------------|---------|-----------|----------|---------------------|-------------------------------------------|
| Qwen2.5-3B     | GSM8K   | 13284186  | TIMEOUT  | Mar 27-29, 2025     | [GRPO Training Results](grpo_training_results.md) |

## Comparison Summary

| Model                  | Size | Method      | Accuracy | Improvement |
|------------------------|------|-------------|----------|-------------|
| Qwen2.5-7B-DPO-VP      | 7B   | Baseline    | 38.0%    | -           |
| Qwen2.5-3B (PPO-tuned) | 3B   | PPO         | 64.0%    | +26.0%      |
| Qwen2.5-3B (GRPO)      | 3B   | GRPO        | N/A      | N/A         |

The PPO fine-tuned Qwen2.5-3B model significantly outperforms the larger baseline model despite having fewer parameters. The GRPO implementation encountered stability issues and reached the 48-hour time limit without completing training or saving any checkpoints. Despite our fixes to tensor dimension issues and loss computation, the script appeared to hang or run extremely slowly in the response generation phase. Unlike the successful PPO training, the GRPO job showed no signs of progress beyond model and data loading. The computational demands for generating preference-based paired responses appear to be significantly higher than those of PPO.

## Method Comparisons

See the [PPO vs GRPO Comparison](ppo_vs_grpo_comparison.md) document for a detailed analysis of these two reinforcement learning approaches.

## Scripts

- `compare_methods.py`: Script for comparing different training methods
- `hyperparameter_sweep.py`: Script for hyperparameter optimization

## Running New Experiments

To run a new experiment:

1. Configure the appropriate parameters in `configs/`
2. Use the corresponding SLURM script (`run_qwen_gsm8k_ppo.sh` for PPO or `run_qwen_gsm8k_grpo.sh` for GRPO)
3. Submit the job using `sbatch`
4. Monitor progress with `check_job.sh` and `check_training_progress.sh`
5. Document results in this directory once completed

## Best Practices

- Always use fat nodes (A100-80GB) for training Qwen2.5 3B models
- Apply memory optimizations as documented in `ppo_training_results.md`
- Save checkpoints frequently (every 200 steps recommended)
- Perform evaluation on a subset of examples during training for early feedback
- For baseline evaluations, ensure proper answer extraction methods are used
- For GRPO and preference-based methods, ensure consistent path configuration and proper error handling 