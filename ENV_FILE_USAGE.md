# Using Environment Files for API Tokens

## Overview

This project uses a `.env` file to securely store API tokens required for accessing Hugging Face resources. This approach keeps sensitive credentials out of the codebase while making them available to the scripts that need them.

## Setting Up Your .env File

You have two options for setting up your .env file:

### Option 1: Using the Setup Script (Recommended)

1. Run the setup script:
   ```bash
   ./setup_env_file.sh
   ```

2. Enter your Hugging Face API token when prompted.

3. The script will create a `.env` file with your token and set the proper permissions.

### Option 2: Manual Setup

1. Copy the template file:
   ```bash
   cp .env.template .env
   ```

2. Edit the `.env` file and replace `your_token_here` with your actual Hugging Face API token:
   ```
   HF_TOKEN=your_actual_token_here
   ```

3. Set secure permissions:
   ```bash
   chmod 600 .env
   ```

## How It Works

All the SLURM scripts in this project are configured to:

1. Check for the presence of a `.env` file
2. Source this file to load environment variables
3. Use the `HF_TOKEN` variable to authenticate with Hugging Face

Example code from our scripts:

```bash
# Source the .env file if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
    echo "Loaded environment variables from .env file"
else
    echo "Warning: .env file not found. HF_TOKEN may not be set." 
    echo "This might cause authentication issues with the Hugging Face API."
fi
```

## Security Considerations

- The `.env` file is included in `.gitignore` to prevent accidental commits
- The setup script sets file permissions to be readable only by the owner (600)
- If you accidentally commit your token, revoke it immediately and create a new one

## Troubleshooting

If you encounter authentication issues:

1. Verify your `.env` file exists in the project root directory
2. Check that it contains the correct `HF_TOKEN=your_token` line
3. Ensure the token is valid and hasn't been revoked
4. Try running the setup script again with a new token

For more detailed security information, see [TOKEN_SECURITY.md](TOKEN_SECURITY.md). 