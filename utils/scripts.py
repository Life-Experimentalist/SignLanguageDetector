"""
Utility scripts for common operations.

This module provides executable functions that can be called directly 
or imported and used from other scripts.
"""

import sys
from pathlib import Path

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import print_error, print_info, print_success, print_warning


def convert_model_reports(lazy=False):
    """
    Convert model reports to JSON format.

    Args:
        lazy (bool): If True, use text reports (faster). If False, test models directly (more accurate).
    """
    try:
        print_info("Running model report conversion...")
        from training.convert_model_reports import convert_existing_model_reports

        convert_existing_model_reports(lazy=lazy)
        return 0
    except Exception as e:
        print_error(f"Error running model report conversion: {e}")
        return 1


def batch_convert_all_models():
    """Convert all models to JSON format using the accurate testing method"""
    print_info("Converting all model reports using direct model testing...")
    return convert_model_reports(lazy=False)


def launch_app(debug=False, port=None):
    """
    Launch the Flask application.

    Args:
        debug (bool): Whether to run in debug mode
        port (int): Port number to use (default: use the one from config)
    """
    try:
        from app import app
        from utils import DEBUG_MODE, PORT

        run_port = port if port is not None else PORT
        run_debug = debug if debug is not None else DEBUG_MODE

        print_info(f"Starting application on port {run_port} (debug={run_debug})")
        app.run(host="0.0.0.0", port=run_port, debug=run_debug)
        return 0
    except Exception as e:
        print_error(f"Error launching application: {e}")
        return 1


def run_training_pipeline():
    """Run the complete training pipeline"""
    try:
        from training_pipeline import training_pipeline

        training_pipeline()
        return 0
    except Exception as e:
        print_error(f"Error running training pipeline: {e}")
        return 1


if __name__ == "__main__":
    # This can be called with command-line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Sign Language Detector Utility Scripts"
    )
    parser.add_argument(
        "--convert-models", action="store_true", help="Convert model reports to JSON"
    )
    parser.add_argument(
        "--lazy", action="store_true", help="Use text reports instead of testing models"
    )
    parser.add_argument(
        "--run-app", action="store_true", help="Launch the Flask application"
    )
    parser.add_argument("--port", type=int, help="Port to run the application on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--pipeline", action="store_true", help="Run the training pipeline"
    )

    args = parser.parse_args()

    if args.convert_models:
        sys.exit(convert_model_reports(lazy=args.lazy))
    elif args.run_app:
        sys.exit(launch_app(debug=args.debug, port=args.port))
    elif args.pipeline:
        sys.exit(run_training_pipeline())
    else:
        print_warning("No action specified. Use --help for options.")
