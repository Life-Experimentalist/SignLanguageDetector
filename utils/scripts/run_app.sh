#!/bin/bash
# Run the Flask application

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the project root directory (parent of utils)
PROJECT_ROOT="$( dirname "$( dirname "$SCRIPT_DIR" )" )"

# Change to the project root directory
cd "$PROJECT_ROOT" || { echo "Failed to change to project directory"; exit 1; }

echo "Starting Sign Language Detector application..."
python -c "from utils.scripts import launch_app; launch_app()"
