#!/bin/bash
#SBATCH --account=berzelius-2025-74
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=main
#SBATCH --gres=gpu:0
#SBATCH -J build_container
#SBATCH -o build_container_%j.out
#SBATCH -e build_container_%j.err

# Define paths
DEFINITION_FILE="env.def"
OUTPUT_SIF="/proj/berzelius-aiics-real/users/x_anbue/env_flash_attn.sif"

# Build the container
echo "Building container from definition file: $DEFINITION_FILE"
echo "This may take a while..."

# Use sudo if available, otherwise use regular command
if command -v sudo &> /dev/null; then
    sudo apptainer build --force $OUTPUT_SIF $DEFINITION_FILE
else
    apptainer build --force $OUTPUT_SIF $DEFINITION_FILE
fi

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Container built successfully at $OUTPUT_SIF"
    echo "You can now use this container in your scripts by updating the APPTAINER_ENV path."
    echo "To update your script, replace:"
    echo "  APPTAINER_ENV=\"/proj/berzelius-aiics-real/users/x_anbue/env.sif\""
    echo "with:"
    echo "  APPTAINER_ENV=\"$OUTPUT_SIF\""
else
    echo "Container build failed. Check the error log for details."
fi 