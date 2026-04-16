# Useful Scripts and Commands

This document contains useful commands and scripts for working with the Sign Language Detector project.

## Environment Setup (uv)

```powershell
uv venv --python 3.12 .venv
uv sync --python .venv\Scripts\python.exe
```

Run the setup script:

```powershell
.\install_requirements.ps1
```

## Table of Contents

- [Model Management Commands](#model-management-commands)
  - [Generate JSON Reports](#generate-json-reports)
  - [Converting Text Reports to JSON](#converting-text-reports-to-json)

Shell scripts are located in the `utils/scripts` directory:

```powershell
# Convert model reports
./utils/scripts/convert_model_reports.sh

# Run the application
./utils/scripts/run_app.sh

# Run the training pipeline
./utils/scripts/run_pipeline.sh
```

### Using Python Module

You can use the Python utility module directly:

```powershell
# Convert model reports (accurate mode)
uv run --python .venv\Scripts\python.exe python -c "from utils.scripts import batch_convert_all_models; batch_convert_all_models()"

# Convert model reports (faster text-based mode)
uv run --python .venv\Scripts\python.exe python -c "from utils.scripts import convert_model_reports; convert_model_reports(lazy=True)"

# Run the application
uv run --python .venv\Scripts\python.exe python -c "from utils.scripts import launch_app; launch_app()"

# Run the training pipeline
uv run --python .venv\Scripts\python.exe python -c "from utils.scripts import run_training_pipeline; run_training_pipeline()"
```

### Using Command Line Interface

The scripts module can be run as a command-line tool:

```powershell
# Convert model reports
uv run --python .venv\Scripts\python.exe python -m utils.scripts --convert-models

# Convert model reports in lazy mode (text-based)
uv run --python .venv\Scripts\python.exe python -m utils.scripts --convert-models --lazy

# Launch the application
uv run --python .venv\Scripts\python.exe python -m utils.scripts --run-app

# Launch the application with custom port
uv run --python .venv\Scripts\python.exe python -m utils.scripts --run-app --port 8080

# Launch the application in debug mode
uv run --python .venv\Scripts\python.exe python -m utils.scripts --run-app --debug

# Run the training pipeline
uv run --python .venv\Scripts\python.exe python -m utils.scripts --pipeline
```

## uv Command Shortcuts

```powershell
uv run --python .venv\Scripts\python.exe python app.py
uv run --python .venv\Scripts\python.exe python app_multi_client.py
uv run --python .venv\Scripts\python.exe python training_pipeline.py
uv run --python .venv\Scripts\python.exe python training/convert_model_reports.py
```

## Script Descriptions

### Model Report Conversion

Converts model reports from text format to JSON for better visualization in the web interface.

- **Accurate Mode**: Tests models directly against data (default)
- **Lazy Mode**: Uses existing text reports (faster but less comprehensive)

### Flask Application

Launches the Flask web application for sign language detection.

### Training Pipeline

Runs the complete training pipeline, including:

1. Data collection
2. Dataset creation
3. Model training
4. Model analysis
5. Inference testing
