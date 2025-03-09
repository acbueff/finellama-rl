# Secure Token Handling

## Why This Is Important

When working with APIs like Hugging Face that require authentication tokens, it's critical to handle these tokens securely. Hardcoding tokens in scripts that get committed to version control (especially public repositories) is a serious security risk:

1. **Exposed Credentials**: Anyone with access to the repository can see and use your token
2. **Token Abuse**: Unauthorized users can use the token to consume your API quota or make unauthorized changes
3. **Account Compromise**: In some cases, tokens with high privilege can lead to account compromise

## How We Handle Tokens in This Project

We use environment variables to manage tokens securely, loaded from a .env file:

1. The `HF_TOKEN` environment variable is used by all scripts that need to authenticate with Hugging Face
2. We never hardcode tokens in scripts that are committed to version control
3. A local .env file contains the actual token, but is excluded from version control via .gitignore
4. Setup scripts are provided to help you create this file securely

## Setting Up Your Token

### Option 1: Using the .env File (Recommended)

This project is configured to use a .env file that's automatically loaded by all scripts:

1. Run the `setup_env_file.sh` script:
   ```bash
   ./setup_env_file.sh
   ```

2. Enter your token when prompted (the input will be hidden)

3. The script will:
   - Create a .env file in the project directory with your token
   - Set appropriate permissions to keep it secure
   - The .env file is automatically included in .gitignore to prevent accidental commits

### Option 2: Setting Environment Variables Manually

If you prefer not to use the .env file approach, you can set the token manually before running any scripts:

```bash
export HF_TOKEN="your_token_here"
```

Then run the workflow script:

```bash
./slurm_master_workflow.sh
```

## Best Practices for Token Security

1. **Use environment variables or .env files** rather than hardcoding tokens
2. **Rotate tokens periodically** for enhanced security
3. **Use the minimum required scope/permissions** for tokens
4. **Never commit tokens** to version control
5. **Add token files to `.gitignore`** to prevent accidental commits
6. **Revoke compromised tokens** immediately if they are accidentally exposed

## What to Do If You Accidentally Commit a Token

If you accidentally push a commit containing a token:

1. **Revoke the token immediately** in your Hugging Face account settings
2. **Create a new token** with the same permissions
3. **Update your .env file** with the new token
4. **Consider using git-filter-repo** to remove the token from git history (although the token should be considered compromised regardless) 