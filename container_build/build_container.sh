#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --gpus=1 -C "fat"
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -J build_container
#SBATCH -o container_build_%j.out
#SBATCH -e container_build_%j.err

# Define important paths
DEFINITION_FILE="$HOME/finellama-rl/container_build/env.def"
TEMP_SIF="/tmp/env_build_$(date +%Y%m%d_%H%M%S).sif"
TEMP_SANDBOX="/tmp/env_sandbox_$(date +%Y%m%d_%H%M%S)"
OUTPUT_SIF="/proj/berzelius-aiics-real/users/x_anbue/env.sif"
BACKUP_SIF="/proj/berzelius-aiics-real/users/x_anbue/env_backup_$(date +%Y%m%d_%H%M%S).sif"

echo "Starting container build at $(date)"
echo "Using definition file: $DEFINITION_FILE"
echo "Building to temporary location: $TEMP_SIF"
echo "Final destination: $OUTPUT_SIF"

# Create backup of current container if it exists
if [ -f "$OUTPUT_SIF" ]; then
    echo "Creating backup of current container..."
    cp "$OUTPUT_SIF" "$BACKUP_SIF"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create backup. Aborting."
        exit 1
    fi
    echo "Backup created at $BACKUP_SIF"
else
    echo "No existing container found at $OUTPUT_SIF"
fi

# Build the new container to a temporary location using --fakeroot as per Berzelius guidelines
echo "Building container using fakeroot... This may take a while."
apptainer build --fakeroot "$TEMP_SIF" "$DEFINITION_FILE"

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Container build failed. Check logs for details."
    exit 1
fi

echo "Container built successfully at $TEMP_SIF"

# Create a sandbox for testing and post-build verification
echo "Creating sandbox for testing..."
apptainer build --fakeroot --sandbox "$TEMP_SANDBOX" "$TEMP_SIF"

# Test within the sandbox
echo "Testing container in sandbox..."
apptainer test "$TEMP_SANDBOX"

if [ $? -ne 0 ]; then
    echo "WARNING: Container test showed issues. Proceeding with additional verification."
    # We continue anyway but will do manual verification
fi

# Perform additional verification of critical components
echo "Performing additional verification of critical components..."
apptainer exec --fakeroot "$TEMP_SANDBOX" python -c "import trl; from trl import GRPOTrainer; print(f'TRL version: {trl.__version__}')"

if [ $? -ne 0 ]; then
    echo "ERROR: Critical component verification failed. Container not deployed."
    echo "Temporary container remains at $TEMP_SIF and sandbox at $TEMP_SANDBOX for inspection."
    exit 1
fi

# Convert sandbox back to final SIF if modifications were made
echo "Converting sandbox back to final SIF..."
apptainer build --fakeroot "$TEMP_SIF" "$TEMP_SANDBOX"

# Move the container to the final location
echo "Moving container to final location..."
mv "$TEMP_SIF" "$OUTPUT_SIF"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to move container to final location."
    echo "Built container remains at $TEMP_SIF"
    exit 1
fi

# Clean up sandbox
echo "Cleaning up sandbox..."
rm -rf "$TEMP_SANDBOX"

echo "Container successfully built and installed at $OUTPUT_SIF"
echo "Build completed at $(date)"
echo "A backup of the previous container is available at $BACKUP_SIF" 