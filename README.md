# FineLlama-RL: Optimizing FineMath-Llama-3B with Reinforcement Learning

This repository contains the implementation for optimizing the FineMath-Llama-3B model using two reinforcement learning techniques:
1. Proximal Policy Optimization (PPO)
2. Gradient-based Reinforcement with Paired Optimization (GRPO)

The project aims to compare these methods for improving mathematical reasoning capabilities in language models.

## About FineMath-Llama-3B

FineMath-Llama-3B is a 3.21B parameter language model specialized on high-quality English mathematical content. This project fine-tunes this model to further enhance its mathematical reasoning capabilities.

## Repository Structure

```
finellama-rl/
├── README.md                  # Project overview, installation, usage instructions
├── requirements.txt           # All dependencies with versions
├── configs/                   # Configuration files
├── data/                      # Data preparation and utilities
├── src/                       # Source code
│   ├── models/                # Model definition and loading utilities
│   ├── rl/                    # RL algorithms implementation
│   ├── eval/                  # Evaluation framework
│   └── utils/                 # Utility functions
├── scripts/                   # Training and evaluation scripts
├── experiments/               # Comparative experiments
└── notebooks/                 # Analysis notebooks
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 16GB memory (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finellama-rl.git
cd finellama-rl
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare the FineMath dataset:

```bash
python data/prepare_data.py --output_dir data/processed --split_ratio 0.9,0.1
```

### Training

#### PPO Training

```bash
python scripts/run_ppo_training.py --config configs/ppo_config.yaml
```

#### GRPO Training

```bash
python scripts/run_grpo_training.py --config configs/grpo_config.yaml
```

### Evaluation

#### Evaluate Baseline Model

```bash
python scripts/run_baseline_eval.py --config configs/eval_config.yaml
```

#### Evaluate PPO-Optimized Model

```bash
python scripts/run_ppo_eval.py --config configs/eval_config.yaml --model_path checkpoints/ppo_model
```

#### Evaluate GRPO-Optimized Model

```bash
python scripts/run_grpo_eval.py --config configs/eval_config.yaml --model_path checkpoints/grpo_model
```

### Comparative Analysis

Compare all methods:

```bash
python experiments/compare_methods.py --baseline_results results/baseline --ppo_results results/ppo --grpo_results results/grpo
```

## Configuration

All hyperparameters can be configured through YAML files in the `configs/` directory:
- `model_config.yaml`: Model parameters
- `ppo_config.yaml`: PPO training hyperparameters
- `grpo_config.yaml`: GRPO training hyperparameters
- `eval_config.yaml`: Evaluation settings

## Evaluation Datasets

The framework evaluates model performance on several mathematical reasoning datasets:
- GSM8K
- MATH dataset
- Custom test sets

## Results and Analysis

Analysis of training runs and model comparisons are available in Jupyter notebooks in the `notebooks/` directory.

## Citation

If you use this code for your research, please cite our paper:

```
@article{author2023comparing,
  title={Comparing PPO and GRPO for Optimizing Mathematical Reasoning in Language Models},
  author={Author, A. and Author, B.},
  journal={Conference/Journal Name},
  year={2023}
}
```

## License

MIT License 