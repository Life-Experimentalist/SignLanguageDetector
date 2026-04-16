"""
Convert existing model text reports to JSON format.

This utility converts existing tabular model reports (.txt files) to the
comprehensive JSON format used by the model info sidebar.
"""

import json
import os
import re
import sys

import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import get_directory_paths
from utils.utils import (
    MODELS_DIR,
    get_labels_dict,
    print_error,
    print_info,
    print_success,
)


def extract_model_metrics(report_text):
    """
    Extract metrics from a classification report text table.

    Args:
        report_text (str): Text content of a model report file

    Returns:
        dict: Dictionary with parsed metrics
    """
    # Parse accuracy
    accuracy_match = re.search(r"Accuracy\s+\|\s+\|\s+\|\s+(\d+\.\d+)", report_text)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None

    if not accuracy:
        print_error("Could not parse accuracy from report")
        return None

    # Regular expression to match class rows in the table
    class_pattern = re.compile(
        r"^\|\s*(\w+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|$",
        re.MULTILINE,
    )

    # Regular expression to match macro/weighted average rows
    avg_pattern = re.compile(
        r"^\|\s*(macro avg|weighted avg)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|$",
        re.MULTILINE,
    )

    # Extract class metrics
    class_matches = class_pattern.finditer(report_text)
    classes = {}

    for match in class_matches:
        class_name, precision, recall, f1_score, support = match.groups()
        # Skip the accuracy row if it was captured
        if class_name == "Accuracy":
            continue

        classes[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1_score),
            "support": float(support),
        }

    averages = {}

    for match in avg_pattern.finditer(report_text):
        avg_type, precision, recall, f1_score, support = match.groups()
        avg_key = avg_type.replace(" ", "_")
        averages[avg_key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1_score),
            "support": float(support),
        }

    # Create the report structure
    report = {
        "model_info": {
            "type": "RandomForestClassifier",  # Default assumption
            "params": {"n_estimators": 100, "random_state": 42},  # Default params
            "training_samples": 0,  # Unknown from text report
            "test_samples": sum([int(classes[c]["support"]) for c in classes]),
        },
        "performance": {"accuracy": accuracy, "classes": classes},
    }

    # Add averages if found
    if "macro_avg" in averages:
        report["performance"]["macro_avg"] = {
            "precision": averages["macro_avg"]["precision"],
            "recall": averages["macro_avg"]["recall"],
            "f1": averages["macro_avg"]["f1-score"],
        }

    if "weighted_avg" in averages:
        report["performance"]["weighted_avg"] = averages["weighted_avg"]

    return report


def convert_txt_to_json(model_name, models_dir=MODELS_DIR, labels_dict=None):
    """
    Convert a model's text report to JSON format.

    Args:
        model_name (str): Name of the model (without extension)
        models_dir (str): Directory where model files are stored
        labels_dict (dict): Optional mapping from numeric labels to text labels

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    txt_path = os.path.join(models_dir, f"{model_name}.txt")
    json_path = os.path.join(models_dir, f"{model_name}.json")

    # Skip if JSON already exists
    if os.path.exists(json_path):
        print_info(f"JSON report for {model_name} already exists, skipping.")
        return False

    # Check if text file exists
    if not os.path.exists(txt_path):
        print_error(f"Text report for {model_name} not found.")
        return False

    try:
        # Read the text file
        with open(txt_path, "r") as f:
            report_text = f.read()

        # Extract metrics
        report = extract_model_metrics(report_text)
        if not report:
            print_error(f"Failed to extract metrics from {txt_path}")
            return False

        # Apply label mapping if provided
        if labels_dict:
            new_classes = {}
            for class_idx, metrics in report["performance"]["classes"].items():
                if class_idx.isdigit() and int(class_idx) in labels_dict:
                    label = labels_dict[int(class_idx)]
                    # Keep both the numeric index and the label
                    new_classes[class_idx] = metrics
                    new_classes[label] = metrics
                else:
                    new_classes[class_idx] = metrics

            report["performance"]["classes"] = new_classes

        # Add dummy confusion matrix (not available from text report)
        classes = [
            c
            for c in report["performance"]["classes"]
            if not (isinstance(c, str) and not c.isdigit())
        ]
        num_classes = len(classes)
        report["confusion_matrix"] = np.zeros((num_classes, num_classes)).tolist()

        # Save the JSON file
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        print_success(f"Created JSON report for {model_name}")
        return True

    except Exception as e:
        print_error(f"Error converting {txt_path} to JSON: {e}")
        return False


def convert_all_models(models_dir=MODELS_DIR):
    """
    Convert all text model reports in the models directory to JSON format.

    Args:
        models_dir (str): Directory where model files are stored

    Returns:
        tuple: (total_models, successful_conversions)
    """
    if not os.path.exists(models_dir):
        print_error(f"Models directory {models_dir} not found.")
        return 0, 0

    # Get all model files (.txt)
    txt_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(models_dir)
        if f.endswith(".txt")
        and os.path.exists(os.path.join(models_dir, f.replace(".txt", ".pkl")))
    ]

    if not txt_files:
        print_info("No model report files found.")
        return 0, 0

    # Get label dictionary for better labels
    labels_dict = get_labels_dict()

    # Convert each file
    success_count = 0
    for model_name in txt_files:
        if convert_txt_to_json(model_name, models_dir, labels_dict):
            success_count += 1

    print_info(
        f"Converted {success_count} of {len(txt_files)} model reports to JSON format."
    )
    return len(txt_files), success_count


# Add a new function to generate JSON directly from model files
def generate_json_from_model(model_path, models_dir=MODELS_DIR, force=False):
    """
    Generate JSON model report directly from a trained model file.WITHOUT using data.pickle.

    This approach doesn't rely on text reports but tests the model directly information
    to produce metrics for the JSON report.

    Args:
        model_path (str): Path to the model .pkl file
        models_dir (str): Directory where models are stored
        force (bool): If True, overwrite existing JSON report

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pickle

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        json_path = os.path.join(models_dir, f"{model_name}.json")

        # Skip if JSON already exists and we're not forcing regeneration
        if os.path.exists(json_path) and not force:
            print_info(f"JSON report for {model_name} already exists, skipping.")
            return False

        # Load model
        print_info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)

        # Extract model if it's in a dictionary structure
        if isinstance(model_dict, dict):
            if "data" in model_dict and "model" in model_dict["data"]:
                model = model_dict["data"]["model"]
                # Check if we have additional metadata in the pickle
                if "metadata" in model_dict["data"]:
                    metadata = model_dict["data"]["metadata"]
                else:
                    metadata = {}
            elif "model" in model_dict:
                model = model_dict["model"]
                metadata = model_dict.get("metadata", {})
            else:
                print_error("Cannot find model in the pickle file structure")
                return False
        else:
            model = model_dict  # Assume it's the model directly
            metadata = {}

        # Generate a basic report with whatever information we can get
        print_info("Generating model info report from model file")

        # Base report structure with model properties
        report = {
            "model_info": {
                "type": model.__class__.__name__,
                "params": model.get_params(),
                "training_samples": metadata.get("training_samples", "Unknown"),
                "test_samples": metadata.get("test_samples", "Unknown"),
                "generated_by": "model_inspection",
                "generation_date": metadata.get("training_date", "Unknown"),
                "note": "This report was generated from the model file without access to training data",
            },
            "performance": {"accuracy": metadata.get("accuracy", None), "classes": {}},
        }

        # Add feature importances if available
        if hasattr(model, "feature_importances_"):
            report["model_info"]["feature_importances"] = (
                model.feature_importances_.tolist()
            )

        # Get class information if available
        if hasattr(model, "classes_"):
            classes = model.classes_

            # Get label mapping if available
            labels_dict = get_labels_dict()

            # If we have performance metrics in metadata, use them
            class_metrics = metadata.get("class_metrics", {})

            for i, class_idx in enumerate(classes):
                class_name = str(class_idx)

                # Check if we have metrics for this class
                if class_name in class_metrics:
                    report["performance"]["classes"][class_name] = class_metrics[
                        class_name
                    ]
                else:
                    # Add placeholder metrics
                    report["performance"]["classes"][class_name] = {
                        "note": "Metrics unavailable (no test data)",
                        "class_index": (
                            int(class_idx)
                            if isinstance(class_idx, (int, np.integer))
                            else class_idx
                        ),
                    }

                # Add label as additional class if available
                if labels_dict and class_idx in labels_dict:
                    label = labels_dict[int(class_idx)]
                    # Copy metrics if we have them
                    if class_name in class_metrics:
                        report["performance"]["classes"][label] = class_metrics[
                            class_name
                        ].copy()
                    else:
                        report["performance"]["classes"][label] = {
                            "note": "Metrics unavailable (no test data)",
                            "class_index": (
                                int(class_idx)
                                if isinstance(class_idx, (int, np.integer))
                                else class_idx
                            ),
                        }

        # If we have macro and weighted averages in metadata, add them
        if "macro_avg" in metadata:
            report["performance"]["macro_avg"] = metadata["macro_avg"]

        if "weighted_avg" in metadata:
            report["performance"]["weighted_avg"] = metadata["weighted_avg"]

        # Add whatever confusion matrix we might have
        if "confusion_matrix" in metadata:
            report["confusion_matrix"] = metadata["confusion_matrix"]
        else:
            # Add dummy placeholder confusion matrix
            num_classes = (
                len(report["performance"]["classes"])
                if len(report["performance"]["classes"]) > 0
                else 2
            )
            report["confusion_matrix"] = np.zeros((num_classes, num_classes)).tolist()

        # Add dummy values for metrics if they're missing, to ensure UI works
        # This helps maintain a consistent structure for the sidebar display
        if not report["performance"]["accuracy"]:
            report["performance"]["accuracy"] = 0.0

        # Check for weighted_avg
        if "weighted_avg" not in report["performance"]:
            report["performance"]["weighted_avg"] = {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 0,
            }

        # Handle class metrics - add dummy metrics for any class that has none
        default_class_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
        }

        for class_name, metrics in report["performance"]["classes"].items():
            if "note" in metrics:
                # This class only has a note - add dummy metrics
                for key, value in default_class_metrics.items():
                    if key not in metrics:
                        metrics[key] = value

        # Save the JSON file
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        print_success(f"Created JSON report for {model_name}")
        return True

    except Exception as e:
        print_error(f"Error generating JSON from model {model_path}: {e}")
        import traceback

        traceback.print_exc()
        return False


def find_models_without_json(models_dir=MODELS_DIR):
    """Find all model files that don't have corresponding JSON reports"""
    if not os.path.exists(models_dir):
        print_error(f"Models directory {models_dir} not found.")
        return []

    # Get all .pkl model files
    model_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pkl")
    ]

    # Filter out those that already have JSON
    models_without_json = [
        model_path
        for model_path in model_files
        if not os.path.exists(model_path.replace(".p", ".json"))
    ]

    return models_without_json


def generate_missing_json_reports(models_dir=MODELS_DIR):
    """Generate JSON reports for all models that don't have them"""
    models_without_json = find_models_without_json(models_dir)

    if not models_without_json:
        print_info("All models already have JSON reports.")
        return 0, 0

    print_info(f"Found {len(models_without_json)} models without JSON reports:")
    for i, model_path in enumerate(models_without_json):
        print_info(f"  {i + 1}. {os.path.basename(model_path)}")

    # Generate reports for each model
    success_count = 0
    for model_path in models_without_json:
        print_info(f"\nGenerating report for {os.path.basename(model_path)}...")
        if generate_json_from_model(model_path, models_dir):
            success_count += 1

    print_info(
        f"Successfully generated {success_count} of {len(models_without_json)} JSON reports."
    )
    return len(models_without_json), success_count


def regenerate_all_model_reports(models_dir=MODELS_DIR):
    """Force regeneration of JSON reports for all model files, regardless of existing reports"""
    # Find all model files (.pkl)
    if not os.path.exists(models_dir):
        print_error(f"Models directory {models_dir} not found.")
        return 0, 0

    model_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pkl")
    ]

    if not model_files:
        print_info("No model files found.")
        return 0, 0

    print_info(f"Found {len(model_files)} model files. Regenerating all reports...")

    success_count = 0
    for model_path in model_files:
        print_info(f"\nRegenerating report for {os.path.basename(model_path)}...")
        if generate_json_from_model(model_path, models_dir, force=True):
            success_count += 1

    print_info(
        f"Successfully regenerated {success_count} of {len(model_files)} JSON reports."
    )
    return len(model_files), success_count


def txt_based_conversion(MODEL_DIR):
    """Convert model reports using text files only"""
    # Get all model files (.txt)
    model_files = [
        os.path.splitext(f)[0]
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".txt")
        and os.path.exists(os.path.join(MODEL_DIR, f.replace(".txt", ".pkl")))
    ]

    if not model_files:
        print_error("No model report files found.")
        return

    print_info(f"Found {len(model_files)} model reports:")
    for i, model in enumerate(model_files):
        print_info(f"  {i + 1}. {model}")

    print_info("\nOptions:")
    print_info("  a - Convert all models")
    print_info("  q - Return to main menu")
    print_info("  Or enter the number of a specific model to convert")

    subchoice = input("\nEnter your choice: ").strip().lower()

    if subchoice == "q":
        return
    elif subchoice == "a":
        total, success = convert_all_models(MODEL_DIR)
        print_info(f"\nConverted {success} of {total} model reports.")
    elif subchoice.isdigit() and 1 <= int(subchoice) <= len(model_files):
        model_name = model_files[int(subchoice) - 1]
        success = convert_txt_to_json(model_name, MODEL_DIR)
        if success:
            print_success(f"Successfully converted {model_name}")
        else:
            print_error(f"Failed to convert {model_name}")
    else:
        print_error("Invalid choice.")


def direct_conversion(MODEL_DIR):
    """Generate reports by directly inspecting models (No data.pickle required)"""
    # Find all model files
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    if not model_files:
        print_error("No model files found.")
        return

    # Group models by whether they need JSON reports
    models_with_json = []
    models_without_json = []

    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(MODEL_DIR, model_file)
        json_path = os.path.join(MODEL_DIR, f"{model_name}.json")

        if os.path.exists(json_path):
            models_with_json.append(model_file)
        else:
            models_without_json.append(model_file)

    print_info(f"Found {len(model_files)} total models.")
    print_info(f"- {len(models_without_json)} models need JSON reports")
    print_info(f"- {len(models_with_json)} models already have JSON reports")

    print_info("\nOptions:")
    print_info("  1 - Generate reports for missing models only")
    print_info("  2 - Regenerate ALL model reports (overwrite existing)")
    print_info("  3 - Select specific models to process")
    print_info("  q - Return to main menu")

    choice = input("\nEnter your choice: ").strip().lower()

    if choice == "q":
        return
    elif choice == "1":
        if not models_without_json:
            print_info("All models already have JSON reports.")
            return

        selected_models = [os.path.join(MODEL_DIR, m) for m in models_without_json]
        process_models(selected_models, MODEL_DIR)
    elif choice == "2":
        # Force regeneration of all models
        confirm = (
            input(
                f"Regenerate reports for ALL {len(model_files)} models? This will overwrite existing reports (y/n): "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            print_info("Operation canceled.")
            return

        regenerate_all_model_reports(MODEL_DIR)
    elif choice == "3":
        # Show list of all models for selection
        print_info("\nAvailable models:")
        for i, model in enumerate(model_files):
            json_exists = (
                "✓"
                if os.path.exists(os.path.join(MODEL_DIR, model.replace(".p", ".json")))
                else " "
            )
            print_info(f"  {i + 1}. [{json_exists}] {model}")

        print_info("\nEnter model numbers separated by commas (e.g., 1,3,5)")
        print_info("  all - Select all models")
        print_info("  q   - Cancel")

        selection = input("\nSelection: ").strip().lower()

        if selection == "q":
            return
        elif selection == "all":
            selected_models = [os.path.join(MODEL_DIR, m) for m in model_files]
        else:
            try:
                # Parse comma-separated indices
                indices = [
                    int(idx.strip())
                    for idx in selection.split(",")
                    if idx.strip().isdigit()
                ]
                valid_indices = [i for i in indices if 1 <= i <= len(model_files)]

                if not valid_indices:
                    print_error("No valid models selected.")
                    return

                selected_models = [
                    os.path.join(MODEL_DIR, model_files[i - 1]) for i in valid_indices
                ]
            except Exception as e:
                print_error(f"Invalid input: {e}")
                return

        # Ask about forcing regeneration
        force = (
            input("\nForce regeneration of reports that already exist? (y/n): ")
            .strip()
            .lower()
            == "y"
        )
        process_models(selected_models, MODEL_DIR, force=force)
    else:
        print_error("Invalid choice.")


def process_models(model_paths, models_dir, force=False):
    """Process a list of model paths to generate JSON reports"""
    print_info(f"\nProcessing {len(model_paths)} models...")
    success_count = 0
    for model_path in model_paths:
        print_info(f"\nGenerating report for {os.path.basename(model_path)}...")
        if generate_json_from_model(model_path, models_dir, force=force):
            success_count += 1

    print_info(
        f"\nSuccessfully processed {success_count} of {len(model_paths)} models."
    )


def convert_existing_model_reports(lazy=False):
    """
    Convert existing models to comprehensive JSON reports

    Args:
        lazy (bool): If True, use text-based conversion instead of model testing (faster but less accurate)
    """
    print_info("\nModel Report Generation")
    print_info("=====================")

    # Get models directory
    directories = get_directory_paths()
    MODEL_DIR = directories["models"]

    if lazy:
        print_info("Using TEXT mode: Converting from text reports only (.txt → .json)")
        txt_based_conversion(MODEL_DIR)
    else:
        print_info(
            "Using MODEL mode: Generating reports by inspecting model files (no data.pickle needed)"
        )
        direct_conversion(MODEL_DIR)


# Update command-line interface to include new option
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model report conversion tools")
    parser.add_argument("--model", "-m", help="Specific model name (without extension)")
    parser.add_argument("--all", "-a", action="store_true", help="Convert all models")
    parser.add_argument("--dir", "-d", help="Models directory path")
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate JSON directly from models instead of text reports",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of JSON reports even if they exist",
    )
    parser.add_argument(
        "--regenerate-all",
        "-r",
        action="store_true",
        help="Regenerate all model reports (overwrite existing)",
    )
    parser.add_argument(
        "--text-only",
        "-t",
        action="store_true",
        help="Use text reports only, don't analyze model files",
    )

    args = parser.parse_args()

    if args.dir:
        models_dir = args.dir
    else:
        models_dir = MODELS_DIR

    if args.regenerate_all:
        print_info("Regenerating ALL model reports (overwriting existing)")
        regenerate_all_model_reports(models_dir)
    elif args.generate:
        if args.model:
            model_path = os.path.join(models_dir, args.model + ".p")
            generate_json_from_model(model_path, models_dir, force=args.force)
        elif args.all:
            if args.force:
                print_info("Regenerating ALL model reports (overwriting existing)")
                regenerate_all_model_reports(models_dir)
            else:
                print_info("Generating reports for models without JSON files")
                generate_missing_json_reports(models_dir)
        else:
            print_info(
                "Please specify a model with --model or use --all to process all models"
            )
    elif args.text_only:
        if args.model:
            print_info(f"Converting text report for {args.model}")
            convert_txt_to_json(args.model, models_dir)
        elif args.all:
            print_info("Converting all text reports")
            convert_all_models(models_dir)
        else:
            print_info(
                "Please specify a model with --model or use --all to process all models"
            )
    else:
        # Interactive mode
        convert_existing_model_reports(lazy=args.text_only)
