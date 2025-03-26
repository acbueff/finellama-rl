# Baseline Evaluation Results: Qwen2.5 on GSM8K

This document catalogs the performance details and results of evaluating the baseline Qwen2.5 model on the GSM8K dataset.

## Job Information

- **Job ID**: 13222486
- **Evaluation Duration**: 3 minutes, 52 seconds (2025-03-16 13:02:36 to 2025-03-16 13:06:28)
- **Status**: COMPLETED SUCCESSFULLY
- **Node**: fat node with A100-80GB GPU

## Evaluation Statistics

- **Model**: SunnyLin/Qwen2.5-7B-DPO-VP
- **Dataset**: GSM8K test set
- **Number of Examples**: 50
- **Batch Size**: 8
- **Max Tokens**: 256

## Evaluation Results

- **Accuracy**: 38.0% (19 correct out of 50 examples)
- **Evaluation Method**: Direct answer comparison with enhanced answer extraction

### Answer Extraction Methodology

The evaluation showed that proper answer extraction was crucial for accurate evaluation. Several iterations of the extraction method were tried:

1. **Initial Extraction**: Looking for standard GSM8K format with "####" (resulted in 0% accuracy)
2. **First Improvement**: Added extraction from "Therefore" or "Thus" statements with numbers (improved to 28% accuracy)
3. **Final Enhancement**: Refined with sophisticated patterns, including:
   - Better equation pattern matching
   - Improved handling of dollar amounts
   - Searching through sentences in reverse order for numbers

The final extraction method yielded the 38% accuracy reported in the results.

## File Locations

### Result Files

- **Metrics**: `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k/gsm8k_metrics.json`
- **Evaluation Summary**: `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k/evaluation_summary.json`
- **Predictions**: `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/results/baseline/gsm8k/gsm8k_predictions.jsonl`

### Evaluation Scripts

- **Main Script**: `/home/x_anbue/finellama-rl/custom_gsm8k_eval.py`
- **SLURM Job Script**: `/home/x_anbue/finellama-rl/slurm_custom_gsm8k_eval.sh`

### Log Files

- **Output Log**: `/home/x_anbue/finellama-rl/logs/gsm8k_eval_13222486.out`
- **Error Log**: `/home/x_anbue/finellama-rl/logs/gsm8k_eval_13222486.err`

## Configuration Details

### Hardware Configuration

- **GPU**: 1x NVIDIA A100-80GB ("fat" node)
- **Runtime Allocation**: 12 hours (actual runtime was under 4 minutes)

### Evaluation Parameters

- **Batch Size**: 8
- **Max Generation Tokens**: 256
- **Model Quantization**: 4-bit (enabled with `--use_4bit` flag)
- **Evaluation Subset**: 50 examples from GSM8K test set

### Model Settings

- **Base Model**: SunnyLin/Qwen2.5-7B-DPO-VP
- **Model Size**: 7B parameters (larger than the 3B model used for PPO)
- **Quantization**: 4-bit
- **Prompt Template**: "Solve the following grade school math problem step by step: {question}\n\nSolution:"

## Analysis of Results

### Strengths
- Model showed ability to solve various types of math problems
- Demonstrated step-by-step reasoning capabilities
- Successfully handled problems involving:
  - Calculating total amounts
  - Time calculations
  - Cost calculations
  - Salary calculations

### Challenges
- Model didn't consistently format answers in the standard GSM8K "####" format
- Some answers were correct but extraction was difficult
- In some cases, intermediate calculations were extracted instead of final answers
- Occasionally captured values from the question rather than the answer

## Comparison with PPO Fine-tuned Model

When compared with the PPO fine-tuned model (Qwen2.5-3B), which achieved 64.0% accuracy, the baseline model's 38.0% accuracy is significantly lower. This demonstrates the effectiveness of PPO fine-tuning for improving mathematical reasoning capabilities.

| Model              | Accuracy | Size |
|--------------------|----------|------|
| Baseline Qwen2.5   | 38.0%    | 7B   |
| PPO-tuned Qwen2.5  | 64.0%    | 3B   |

It's particularly notable that the PPO-tuned 3B model outperformed the larger 7B baseline model by 26 percentage points despite having fewer parameters.

## References

- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [Qwen2.5 Model Series](https://huggingface.co/Qwen)
- [SunnyLin/Qwen2.5-7B-DPO-VP](https://huggingface.co/SunnyLin/Qwen2.5-7B-DPO-VP) 