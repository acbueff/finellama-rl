#!/bin/bash
# Script to set up the HF_TOKEN environment variable securely
# This script should NOT be committed to version control

echo "Setting up Hugging Face API token environment"
echo "=============================================="
echo ""
echo "IMPORTANT: This script will ask for your Hugging Face API token"
echo "to set it as an environment variable. The token will be stored"
echo "in your ~/.bashrc file, but will not be committed to git."
echo ""
echo "Enter your Hugging Face API token: "
read -s token

# Check if token was provided
if [ -z "$token" ]; then
    echo "No token provided. Exiting."
    exit 1
fi

# Add to current session
export HF_TOKEN="$token"

# Check if already in bashrc
if grep -q "export HF_TOKEN=" ~/.bashrc; then
    # Replace existing token
    sed -i "s/export HF_TOKEN=.*/export HF_TOKEN=\"$token\"/" ~/.bashrc
    echo "Updated existing HF_TOKEN in ~/.bashrc"
else
    # Add new token
    echo "" >> ~/.bashrc
    echo "# Hugging Face API token - Added by setup_env.sh" >> ~/.bashrc
    echo "export HF_TOKEN=\"$token\"" >> ~/.bashrc
    echo "Added HF_TOKEN to ~/.bashrc"
fi

echo ""
echo "HF_TOKEN has been set for your current session and added to your ~/.bashrc"
echo "You can now run the workflows with proper authentication."
echo ""
echo "To start the entire workflow, run:"
echo "./slurm_master_workflow.sh" 