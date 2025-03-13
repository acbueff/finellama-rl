#!/bin/bash
# Script to check the status of slurm jobs and tail their output logs

# Get job ID from command line or use the latest job
if [ $# -eq 1 ]; then
    JOB_ID=$1
else
    # Get the latest job ID if none provided
    LATEST_JOB=$(squeue -u $USER --sort=-i --noheader | head -1)
    if [ -z "$LATEST_JOB" ]; then
        echo "No running jobs found."
        exit 1
    fi
    JOB_ID=$(echo $LATEST_JOB | awk '{print $1}')
    echo "Using latest job ID: $JOB_ID"
fi

# Check the job status
echo "Checking status of job $JOB_ID"
scontrol show job $JOB_ID

# Check if the job is still running
JOB_STATE=$(scontrol show job $JOB_ID | grep JobState | awk -F= '{print $2}' | awk '{print $1}')
echo "Job state: $JOB_STATE"

# Find the output and error logs
OUTPUT_LOG="/home/x_anbue/finellama-rl/logs/gsm8k_eval_${JOB_ID}.out"
ERROR_LOG="/home/x_anbue/finellama-rl/logs/gsm8k_eval_${JOB_ID}.err"

# Check if the output log exists and tail it
if [ -f "$OUTPUT_LOG" ]; then
    echo -e "\nTailing output log ($OUTPUT_LOG):"
    echo "----------------------------------------"
    tail -n 50 "$OUTPUT_LOG"
else
    echo "Output log not found: $OUTPUT_LOG"
fi

# Check if the error log exists and tail it
if [ -f "$ERROR_LOG" ]; then
    echo -e "\nTailing error log ($ERROR_LOG):"
    echo "----------------------------------------"
    tail -n 50 "$ERROR_LOG"
else
    echo "Error log not found: $ERROR_LOG"
fi

echo -e "\nTo continuously monitor the output, use:"
echo "tail -f $OUTPUT_LOG" 