
# Utils Package

This directory contains utility functions and configuration for the Sign Language Detector application.

## Structure

- `utils.py`: Core utility functions for image processing, model handling, and UI
- `config.py`: Configuration settings loaded from environment variables
- `__init__.py`: Package initialization that exports all commonly used functions and variables

## Organization

The package is structured to avoid circular imports:
- `utils.py` contains all the core utility functions 
- `config.py` uses these functions but doesn't import them directly (uses runtime imports)
- `__init__.py` imports from both files and exposes a clean API

## Usage

To use functions and variables from this package:

```python
from utils import print_info, calculate_brightness, mediapipe_hands, MODELS_DIR
```

## Configuration

Configuration is loaded from `.env` in the project root. Important variables:
- `PORT`: The HTTP server port (default: 5000)
- `BRIGHTNESS_THRESHOLD`: Threshold for brightness warnings (default: 85)
- `MODELS_DIR`: Path to model files (default: project_root/models)
