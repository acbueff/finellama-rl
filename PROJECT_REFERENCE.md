# FineLlama-RL: Project Reference Document

## Project Overview

This project aims to compare two reinforcement learning approaches for optimizing language models on mathematical reasoning tasks:
1. Proximal Policy Optimization (PPO)
2. Gradient-based Reinforcement with Paired Optimization (GRPO)

The primary goal is to train the Qwen2.5 3B model on the GSM8K dataset using both PPO and GRPO techniques, then evaluate and compare their performance on the validation set to determine which approach yields better results.

## System Architecture

### Code Location
- **Repository Path**: `/home/x_anbue/finellama-rl`
- This is where all code and scripts are stored on the Berzelius HPC.

### Storage Locations
- **Primary Storage**: `/proj/berzelius-aiics-real/users/x_anbue`
  - **Data Storage**: `/proj/berzelius-aiics-real/users/x_anbue/finellama-data`
  - **GSM8K Dataset**: `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k`
  - **Training Results**: `/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo`
  - **HuggingFace Cache**: `/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache`

### Models
- **Base Model**: Qwen/Qwen2.5-3B-Instruct
- This model serves as the foundation for both PPO and GRPO training.

## SLURM Configuration and Batch Handling

### SLURM Account Information
- **Account Name**: `berzelius-2025-74`
- All job submissions should use this account with the `--account=berzelius-2025-74` parameter

### GPU Resources and Partitions
- **GPU Request**: Use `--gpus=1 -C "fat"` to request 1 GPU on the fat nodes
- **Typical Time Limit**: `0-24:00:00` (24 hours) for training jobs, `0-12:00:00` (12 hours) for evaluation

### Apptainer/Singularity Container
- **Container Path**: `/proj/berzelius-aiics-real/users/x_anbue/env.sif`
- All Python scripts should be executed inside this container using:
  ```bash
  apptainer exec --nv --env PYTHONPATH="${PYTHONPATH}" ${APPTAINER_ENV} python script.py
  ```

### Critical Environment Variables
- **HuggingFace Token**: Store in `~/.huggingface_token` or `.env` file
- **PYTHONPATH**: Always set to include the current directory with `export PYTHONPATH=$PYTHONPATH:$(pwd)`
- **Cache Directories**:
  ```bash
  export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
  export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache" 
  export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
  ```

### SLURM Job Submission Template
```bash
#!/bin/bash
#SBATCH --account=berzelius-2025-74      # Project account
#SBATCH --gpus=1 -C "fat"                # Request 1 GPU on fat partition
#SBATCH -t 0-24:00:00                    # Time limit: 24 hours
#SBATCH -J job_name                      # Job name
#SBATCH -o logs/job_name_%j.out          # Standard output log
#SBATCH -e logs/job_name_%j.err          # Standard error log

# Environment setup
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_DATASETS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_HOME="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export TRANSFORMERS_CACHE="/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache"
export HF_TOKEN=$(cat ~/.huggingface_token 2>/dev/null || echo "")

# Container definition
APPTAINER_ENV="/proj/berzelius-aiics-real/users/x_anbue/env.sif"

# Run inside container
apptainer exec --nv \
  --env PYTHONPATH="${PYTHONPATH}" \
  ${APPTAINER_ENV} python your_script.py
```

### Best Practices for SLURM Jobs
1. **Always verify account name** before submission - using incorrect account will result in submission errors
2. **Create output directories** before job starts to avoid runtime errors
3. **Add descriptive job names** to easily identify jobs in the queue
4. **Use the container** for all Python executions to ensure environment consistency
5. **Monitor jobs** with `squeue -u $USER` and `sacct -j <job_id>`
6. **Check logs** in real-time with `tail -f logs/job_name_<job_id>.out`

## Repository Structure

### Root Directory

```
finellama-rl/
├── README.md                      # Project overview and basic usage instructions
├── TOKEN_SECURITY.md              # Security guidelines for handling tokens
├── ENV_FILE_USAGE.md              # Documentation on environment file usage
├── requirements.txt               # Project dependencies
├── .env                           # Environment variables configuration
├── .gitignore                     # Git ignore specifications
```

### Source Code (`src/`)

The `src` directory contains the core implementation of the project:

```
src/
├── __init__.py                    # Package initialization
├── models/                        # Model implementations
│   ├── __init__.py                # Package initialization
│   ├── finemathllama.py           # FineMath Llama model implementation
│   └── custom_config.py           # Custom model configuration
├── rl/                            # Reinforcement Learning implementations
│   ├── __init__.py                # Package initialization
│   ├── base_trainer.py            # Base class for RL trainers
│   ├── ppo_trainer.py             # PPO algorithm implementation
│   ├── grpo_trainer.py            # GRPO algorithm implementation
│   ├── reward_model.py            # Reward modeling for RL
│   └── rollout.py                 # Environment rollout implementation
├── eval/                          # Evaluation framework
│   ├── __init__.py                # Package initialization
│   ├── evaluator.py               # Model evaluation utilities
│   └── metrics.py                 # Performance metrics calculation
└── utils/                         # Utility functions
    ├── __init__.py                # Package initialization
    ├── common_utils.py            # Common utility functions
    └── logging_utils.py           # Logging utilities
```

### Configurations (`configs/`)

Configuration files containing hyperparameters and settings:

```
configs/
├── model_config.yaml              # Model configuration parameters
├── ppo_config.yaml                # PPO training hyperparameters
├── grpo_config.yaml               # GRPO training hyperparameters
└── eval_config.yaml               # Evaluation settings
```

### Data Processing (`data/`)

Scripts for data preparation and utilities:

```
data/
├── prepare_data.py                # Dataset preparation script
├── data_utils.py                  # Data processing utilities
└── configs/                       # Data configuration files
```

### Scripts (`scripts/`)

Scripts for running various aspects of the project:

```
scripts/
├── run_ppo_training.py            # Run PPO training
├── run_grpo_training.py           # Run GRPO training
├── run_ppo_eval.py                # Evaluate PPO-trained model
├── run_grpo_eval.py               # Evaluate GRPO-trained model
├── run_baseline_eval.py           # Evaluate baseline model
├── download_gsm8k.py              # Python script to download GSM8K dataset
├── download_gsm8k.sh              # Shell script to download GSM8K dataset
└── compare_methods.py             # Compare different training methods
```

### Shell Scripts (Root Directory)

Training and evaluation shell scripts:

```
├── setup_env.sh                   # Environment setup script
├── setup_env_file.sh              # Environment file setup script
├── ppo_finetune.sh                # PPO fine-tuning script
├── grpo_finetune.sh               # GRPO fine-tuning script
├── run_rl_training.sh             # Main RL training script
├── run_qwen_gsm8k.sh              # Script to run Qwen on GSM8K
├── slurm_grpo_train.sh            # SLURM job for GRPO training
├── slurm_download_model.sh        # SLURM job for model downloading
├── check_job.sh                   # Script to check job status
├── check_training_progress.sh     # Monitor training progress
├── run_ppo_gsm8k_eval.sh          # Evaluate PPO on GSM8K
├── slurm_custom_gsm8k_eval.sh     # SLURM job for GSM8K evaluation
├── run_simplified_ppo.sh          # Run simplified PPO training
├── run_direct_ppo_eval.sh         # Direct PPO evaluation
├── run_direct_ppo_eval_apptainer.sh # PPO evaluation with Apptainer
├── run_direct_ppo_eval_fixed.sh   # Fixed version of PPO evaluation
└── slurm_master_workflow.sh       # Master workflow for SLURM jobs
```

### Analysis and Evaluation Scripts

Scripts for analysis and evaluation:

```
├── custom_gsm8k_eval.py           # Custom GSM8K evaluation
├── custom_ppo_eval.py             # Custom PPO evaluation
├── custom_ppo_train.py            # Custom PPO training
├── analyze_predictions.py         # Analyze model predictions
├── analyze_improved_predictions.py # Analyze improved predictions
├── check_ppo_model.py             # Check PPO model performance
├── ppo_model_inference.py         # PPO model inference utilities
└── direct_ppo_eval.py             # Direct PPO evaluation
```

### Support Directories

```
├── logs/                          # Log files from training and evaluation
├── experiments/                   # Experimental configurations and results
├── notebooks/                     # Analysis notebooks
└── __pycache__/                   # Python cache files (should not be committed)
```

## Workflow

### Training Process
1. **Data Preparation**: Use `prepare_gsm8k_data.py` to prepare the GSM8K dataset from `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k`.
2. **PPO Training**: Use `ppo_finetune.sh` to fine-tune Qwen2.5 3B with PPO.
3. **GRPO Training**: Use `grpo_finetune.sh` to fine-tune Qwen2.5 3B with GRPO.
4. **Training is managed** through SLURM scripts (`slurm_grpo_train.sh`) on the Berzelius HPC.

### Evaluation Process
1. **Model Checkpoint Management**: Trained model checkpoints are stored in `/proj/berzelius-aiics-real/users/x_anbue/ppo_grpo`.
2. **PPO Evaluation**: Evaluate PPO-trained model using `run_ppo_gsm8k_eval.sh`.
3. **GRPO Evaluation**: Similar evaluation for GRPO-trained model.
4. **Comparison**: Use `compare_models.sh` to compare the performance of PPO vs GRPO.

### Monitoring
- Use `check_job.sh` to monitor job status.
- Use `check_training_progress.sh` to track training progress.

## Important Guidelines

1. **Always check file locations** before generating new content to ensure proper integration with the project structure.
2. **Storage considerations**: 
   - Code and scripts go to `/home/x_anbue/finellama-rl`
   - Data, model weights, and checkpoints go to `/proj/berzelius-aiics-real/users/x_anbue`
3. **Model caching**: All HuggingFace models are cached at `/proj/berzelius-aiics-real/users/x_anbue/.huggingface_cache`
4. **Dataset location**: GSM8K dataset is stored at `/proj/berzelius-aiics-real/users/x_anbue/finellama-data/gsm8k`

## Project Goals

1. **Primary Objective**: Compare PPO and GRPO for fine-tuning language models on mathematical reasoning tasks.
2. **Approach**:
   - Train Qwen2.5 3B on GSM8K training set using PPO
   - Evaluate on GSM8K validation set
   - Train Qwen2.5 3B on GSM8K training set using GRPO
   - Evaluate on GSM8K validation set
3. **Success Metrics**: Compare validation performance to determine which approach (PPO or GRPO) yields better results.

This document serves as the primary reference for understanding the project structure, purpose, and implementation details. It should be consulted before making any changes to ensure consistency with the project's architecture and goals. 