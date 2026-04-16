#!/bin/bash
# Convert model reports to JSON format

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of utils)
PROJECT_ROOT="$( dirname "$( dirname "$SCRIPT_DIR" )" )"

# Change to the project root directory
cd "$PROJECT_ROOT" || { echo "Failed to change to project directory"; exit 1; }

echo "Converting model reports using direct model testing..."
python -c "from utils.scripts import batch_convert_all_models; batch_convert_all_models()"
