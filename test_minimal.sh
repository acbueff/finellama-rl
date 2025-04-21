#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH -t 0:05:00
#SBATCH -J test_job
#SBATCH -o test_job_%j.out
#SBATCH -e test_job_%j.err

echo "Hello from SLURM job"
sleep 10
echo "Job completed"
