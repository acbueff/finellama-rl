# GRPO Training Results: Qwen2.5 3B on GSM8K

This document catalogs the performance details and results of attempting to train the Qwen2.5 3B model on the GSM8K dataset using Generative Reward Policy Optimization (GRPO).

## Job Information

- **Job ID**: 13284186
- **Training Duration**: 48 hours (time limit reached)
- **Status**: TIMEOUT
- **Node**: node085 (fat node with A100-80GB GPU)

## Training Statistics

- **Training Steps**: 0 (no training progress observed)
- **Checkpoints**: None saved
- **Memory Usage**: High GPU utilization (99%) but no apparent training progress

## Issues Encountered

### Tensor Dimension Issues

- Initial attempts encountered tensor dimension errors during backward pass
- Fixed code to properly handle tensor dimensions in loss computation
- Added proper reduction operations for non-scalar tensors

### Computational Bottleneck

- Response generation for paired comparisons appears to be extremely compute-intensive
- Script successfully loaded models and data but appeared to hang in the generation phase
- Despite 99% GPU utilization and high CPU usage, no output or checkpoints were produced

### Time Limit Constraints

- Job was terminated after reaching the 48-hour time limit
- No apparent progress was made in the training phase
- No model checkpoints or evaluation results were saved

## Comparison with PPO

- PPO training successfully completed within 36 hours
- GRPO requires generating paired responses for each problem, significantly increasing computational requirements
- Response generation phase in GRPO appears to be a major bottleneck not present in PPO

## Recommendations for Future Work

1. **Optimization Strategies**:
   - Reduce the number of generated response pairs
   - Simplify the preference computation mechanism
   - Implement more efficient response generation

2. **Implementation Adjustments**:
   - Add more detailed logging for progress tracking
   - Implement incremental checkpointing to save partial progress
   - Consider a multi-stage approach where responses are pre-generated

3. **Resource Allocation**:
   - Allocate longer runtime for GRPO jobs (72+ hours)
   - Consider using multiple GPUs for parallel response generation
   - Pre-compute and cache responses to reduce online generation time

## Conclusion

The GRPO training attempt was unsuccessful due to computational bottlenecks and time constraints. While the implementation issues related to tensor dimensions were resolved, the fundamental challenge appears to be the computational demands of generating paired responses for preference learning. For future experiments, significant optimization of the GRPO implementation would be necessary, or alternative approaches that reduce the generation overhead should be considered.

## Job Information

- **Job ID**: 13222580 (initial failed attempt)
- **Training Duration**: < 1 hour (did not complete successfully)
- **Status**: FIXED, READY TO RUN
- **Node**: fat node with A100-80GB GPU

## Training Issues and Fixes

The initial GRPO training job encountered several errors that have now been addressed:

1. **Model Loading Error**: The job attempted to load `SunnyLin/Qwen2.5-7B-DPO-VP` instead of the 3B model.
   - **Fix**: Updated configuration to explicitly use `Qwen/Qwen2.5-3B` consistently across all scripts.
   - **Implementation**: Added robust error handling and explicit model path specification.

2. **Directory Structure Mismatch**: There was a path mismatch between configured output paths.
   - **Fix**: Standardized all paths to match the successful PPO training configuration.
   - **Implementation**: Updated all scripts to use consistent path structure.

3. **Script Synchronization**: The training script wasn't properly aligned with the job submission script.
   - **Fix**: Added proper environment variable passing and ensured script parameters match.
   - **Implementation**: Added explicit error handling for model and tokenizer loading.

4. **Memory Efficiency**: Added optimizations to improve memory efficiency.
   - **Fix**: Applied all memory optimizations from PPO training.
   - **Implementation**: Updated accelerator initialization with mixed precision and other optimizations.

## Preparation for Re-run

The GRPO training script has been updated based on the successful PPO training blueprint:

1. **Model Configuration**: Now explicitly set to use Qwen2.5 3B with 4-bit quantization and LoRA.
2. **Error Handling**: Added comprehensive error handling to provide clear diagnostic messages.
3. **Memory Optimization**: Incorporated all memory optimizations from the successful PPO training.
4. **Path Consistency**: Ensured consistent paths throughout all configuration files.

## File Locations

### Training Scripts

- **Main Script**: `/home/x_anbue/finellama-rl/qwen_gsm8k_grpo_train.py`
- **SLURM Job Script**: `/home/x_anbue/finellama-rl/run_qwen_gsm8k_grpo.sh`

### Configuration Files

- **GRPO Config**: `/home/x_anbue/finellama-rl/configs/qwen_gsm8k_grpo_config.yaml`
- **Model Config**: `/home/x_anbue/finellama-rl/configs/qwen2.5_3b_config.yaml`

### Output Locations

- **Checkpoint Directory**: `/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/models/grpo_finetuned`
- **Results Directory**: `/proj/berzelius-aiics-real/users/x_anbue/qwen_gsm8k/results/grpo`

## Next Steps

1. **Submit GRPO Job**: Use the updated scripts to submit the GRPO training job.
   ```bash
   sbatch run_qwen_gsm8k_grpo.sh
   ```

2. **Monitor Progress**: Watch the logs for any issues.
   ```bash
   tail -f logs/qwen_gsm8k_grpo_<JOB_ID>.out
   ```

3. **Evaluate Results**: Once training completes, compare with PPO results.

## References

- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Qwen2.5 Model Series](https://huggingface.co/Qwen)
- [GRPO Implementation Details](https://arxiv.org/abs/2305.18290) 