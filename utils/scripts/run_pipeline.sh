#!/bin/bash
# Run the complete training pipeline

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of utils)
PROJECT_ROOT="$( dirname "$( dirname "$SCRIPT_DIR" )" )"

# Change to the project root directory
cd "$PROJECT_ROOT" || { echo "Failed to change to project directory"; exit 1; }

echo "Running Sign Language Detector training pipeline..."
python -c "from utils.scripts import run_training_pipeline; run_training_pipeline()"
