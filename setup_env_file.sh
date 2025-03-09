#!/bin/bash
# Script to set up the .env file for storing the HF_TOKEN
# This script does not store any tokens itself and is safe to commit

ENV_FILE="$(dirname "$0")/.env"

echo "Setting up Hugging Face API token in .env file"
echo "=============================================="
echo ""
echo "IMPORTANT: This script will create a .env file with your Hugging Face API token."
echo "The .env file is listed in .gitignore and should NEVER be committed to git."
echo ""
echo "Enter your Hugging Face API token: "
read -s token

# Check if token was provided
if [ -z "$token" ]; then
    echo "No token provided. Exiting."
    exit 1
fi

# Create or update .env file
if [ -f "$ENV_FILE" ]; then
    # File exists, update it
    sed -i '/HF_TOKEN=/d' "$ENV_FILE"
    echo "HF_TOKEN=$token" >> "$ENV_FILE"
    echo "Updated existing .env file with your token."
else
    # Create new file
    echo "# Environment variables for the finellama-rl project" > "$ENV_FILE"
    echo "# WARNING: This file contains secrets - NEVER commit it to version control" >> "$ENV_FILE"
    echo "" >> "$ENV_FILE"
    echo "HF_TOKEN=$token" >> "$ENV_FILE"
    echo "Created new .env file with your token."
fi

# Set permissions to be readable only by the user
chmod 600 "$ENV_FILE"
echo "Set permissions on .env file to be readable only by you."

echo ""
echo "Your HF_TOKEN has been saved to .env file."
echo "When running scripts, make sure they source this .env file."
echo ""
echo "To start the entire workflow, run:"
echo "./slurm_master_workflow.sh" 