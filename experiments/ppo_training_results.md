# PPO Training Results: Qwen2.5 3B on GSM8K

This document catalogs the performance details and results of training the Qwen2.5 3B model on the GSM8K dataset using Proximal Policy Optimization (PPO).

## Job Information

- **Job ID**: 13232553
- **Training Duration**: 1 day, 12 hours, 55 minutes (2025-03-18 12:42 to 2025-03-20 01:38)
- **Status**: COMPLETED SUCCESSFULLY
- **Node**: node093 (fat node with A100-80GB GPU)

## Training Statistics

- **Training Steps**: 5000 (completed all planned steps)
- **Checkpoints**: Saved at intervals of 200 steps (25 checkpoints total)
- **Initial Memory Issues**: Required multiple attempts with progressively smaller batch sizes before successful run

### Training Metrics at Key Points

- **Initial (Step 0)**:
  - Average reward: 1.0739
  - Loss: 0.9439
  - Policy loss: -0.8574
  - Value loss: 3.6025

- **Middle (Step 2500)**:
  - Metrics show gradual improvement
  - (Specific metrics not extracted)

- **Final (Step 4960)**:
  - Average reward: 0.5630
  - Loss: -0.4986
  - Policy loss: -0.6336
  - Value loss: 0.2699

## Final Evaluation Results

- **Accuracy**: 64.0% (32 correct out of 50 examples)
- **Evaluation Set**: GSM8K validation dataset (subset of 50 examples)
- **Evaluation Method**: Direct answer comparison

## File Locations

### Saved Models and Results

- **Final Model**: `/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/ppo_finetuned/checkpoint-5000`
- **Evaluation Results**: `/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/eval_results.json`
- **Training Statistics**: `/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/training_stats.json`

### Configuration Files

- **PPO Config**: `/home/x_anbue/finellama-rl/configs/qwen_gsm8k_ppo_config.yaml`
- **Model Config**: `/home/x_anbue/finellama-rl/configs/qwen2.5_3b_config.yaml`

### Training Scripts

- **Main Script**: `/home/x_anbue/finellama-rl/qwen_gsm8k_ppo_train.py`
- **SLURM Job Script**: `/home/x_anbue/finellama-rl/run_qwen_gsm8k_ppo.sh`

### Log Files

- **Output Log**: `/home/x_anbue/finellama-rl/logs/qwen_gsm8k_ppo_13232553.out`
- **Error Log**: `/home/x_anbue/finellama-rl/logs/qwen_gsm8k_ppo_13232553.err`

## Configuration Details

### Hardware Configuration

- **GPU**: 1x NVIDIA A100-80GB ("fat" node)
- **Memory**: Full node memory allocation (`--mem=0`)
- **Runtime Allocation**: 48 hours

### Training Parameters

- **Batch Size**: 4 (reduced from 64 in original configuration)
- **Mini-batch Size**: 4
- **Gradient Accumulation Steps**: 16
- **Learning Rate**: 2.0e-5
- **PPO Epochs**: 2 (reduced from 4 to save memory)
- **Maximum Sequence Length**: 384 tokens (reduced from 512)
- **Rollout Batches**: 16 (reduced from 32)

### Model Settings

- **Base Model**: Qwen/Qwen2.5-3B
- **Quantization**: 4-bit
- **PEFT Method**: LoRA with reduced rank (r=8, alpha=16)
- **Memory Optimizations**:
  - FlashAttention-2
  - Disabled KV-cache during training
  - Low CPU memory usage setting

### Optimization Lessons

- A100-40GB GPUs were insufficient for this model even with reduced batch sizes
- A100-80GB GPUs (fat nodes) were required for successful training
- The following memory optimizations were critical:
  - Reducing batch and sequence sizes
  - Using 4-bit quantization
  - Enabling gradient checkpointing
  - Using FlashAttention-2
  - Disabling KV-cache during training

## Next Steps

- **Full Evaluation**: Run a comprehensive evaluation on the complete GSM8K validation set
- **Compare with GRPO**: Train a model using GRPO with similar parameters for comparison
- **Inference Testing**: Test the model with new math problems to assess generalization
- **Ablation Studies**: Experiment with different hyperparameters to improve performance

## References

- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Qwen2.5 Model Series](https://huggingface.co/Qwen)
- [PPO Implementation Details](https://arxiv.org/abs/1707.06347) 