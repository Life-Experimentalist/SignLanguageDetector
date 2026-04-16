# Project: Sign Language Detector
# Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
# Owner: VKrishna04
# Organization: Life-Experimentalist
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from utils import print_success

# Add the parent directory (project root) to sys.path so that utils can be found.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from typing import Any, cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tabulate import tabulate  # pip install tabulate if needed

from training.convert_model_reports import convert_all_models, convert_txt_to_json
from utils.utils import (
    N_ESTIMATORS,
    RANDOM_STATE,
    Spinner,
    get_directory_paths,
    load_data,
    print_error,
    print_info,
    save_data,
)


def evaluate_model(model, x_test, y_test):
    """Evaluate the model performance and print the classification report."""
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    report_dict = classification_report(y_test, y_predict, output_dict=True)

    # Ensure report_dict is a dictionary
    if not isinstance(report_dict, dict):
        print_info(
            "Classification report returned as string, converting to dict format..."
        )
        # Print the report as is and create an empty dict to continue
        print_info(report_dict)
        report_dict = {"accuracy": score}

    # Build table for the classification report (skip accuracy key)
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for key, values in report_dict.items():
        if key == "accuracy":
            rows.append(["Accuracy", "", "", f"{values:.2f}", ""])
        elif isinstance(values, dict) and all(
            k in values for k in ["precision", "recall", "f1-score", "support"]
        ):
            rows.append(
                [
                    key,
                    f"{values['precision']:.2f}",
                    f"{values['recall']:.2f}",
                    f"{values['f1-score']:.2f}",
                    f"{values['support']}",
                ]
            )
    table = tabulate(rows, headers=headers, tablefmt="grid")

    print_info(f"\nAccuracy: {score * 100:.2f}%")
    print_info("\nDetailed Classification Report:")
    print_info(table)

    return score, table


def generate_model_report(model, x_test, y_test, y_train, class_labels=None):
    """
    Generate a comprehensive model report in both structured and tabular formats

    Args:
        model: The trained classifier model
        x_test: Test features
        y_test: Test labels
        y_train: Training labels
        class_labels: Optional dictionary mapping class indices to readable labels

    Returns:
        tuple: (report_dict, report_table) containing structured data and tabular text
    """
    # Get predictions
    y_predict = model.predict(x_test)

    # Calculate accuracy
    score = accuracy_score(y_test, y_predict)

    # Get detailed classification report as dict and string
    report_dict = cast(
        dict[str, Any], classification_report(y_test, y_predict, output_dict=True)
    )
    report_table = tabulate(
        [["Class", "Precision", "Recall", "F1-Score", "Support"]]
        + [
            [
                class_labels.get(int(k), k)
                if class_labels and isinstance(k, str) and k.isdigit()
                else k,
                f"{v['precision']:.2f}",
                f"{v['recall']:.2f}",
                f"{v['f1-score']:.2f}",
                f"{v['support']}",
            ]
            for k, v in report_dict.items()
            if isinstance(v, dict)
        ]
        + [["Accuracy", "", "", f"{score:.2f}", ""]],
        headers="firstrow",
        tablefmt="grid",
    )

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_predict)

    # Calculate feature importances if available
    feature_importances = None
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_.tolist()

    # Calculate per-class metrics with support
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_test, y_predict, average=None
    )
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    fscore = np.asarray(fscore)
    support = np.asarray(support)

    # Create a comprehensive report
    comprehensive_report = {
        "model_info": {
            "type": model.__class__.__name__,
            "params": model.get_params(),
            "training_samples": len(y_train),
            "test_samples": len(y_test),
        },
        "performance": {
            "accuracy": float(score),
            "macro_avg": {
                "precision": float(np.mean(precision)),
                "recall": float(np.mean(recall)),
                "f1": float(np.mean(fscore)),
            },
            "weighted_avg": report_dict.get("weighted avg", {}),
            "classes": {},
        },
        "confusion_matrix": cm.tolist(),
    }

    # Add feature importances if available
    if feature_importances:
        comprehensive_report["model_info"]["feature_importances"] = feature_importances

    # Add per-class metrics
    unique_classes = np.unique(y_test)
    for i, class_idx in enumerate(unique_classes):
        class_name = (
            class_labels.get(int(class_idx), str(class_idx))
            if class_labels
            else str(class_idx)
        )
        comprehensive_report["performance"]["classes"][class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(fscore[i]),
            "support": int(support[i]),
        }

    return comprehensive_report, report_table


def train_classifier():
    """Train a RandomForest classifier and save the model with comprehensive reports."""
    # Get the project directory and paths
    directories = get_directory_paths()

    # Ask for a data file to use (with default option)
    data_files = [f for f in os.listdir(directories["data"]) if f.endswith(".pickle")]

    if not data_files:
        print_error("Error: No data pickle files found in the data directory.")
        print_info("Please run create_dataset.py first to generate training data.")
        return

    print_info("\nAvailable data files:")
    for i, data_file in enumerate(data_files):
        print_info(f"  {i + 1}. {data_file}")

    default_data = "data.pickle"
    if default_data in data_files:
        default_index = data_files.index(default_data) + 1
        data_choice = input(f"\nSelect data file to use [{default_index}]: ").strip()

        if not data_choice:
            data_choice = str(default_index)
    else:
        data_choice = input("\nSelect data file number: ").strip()

    if data_choice.isdigit() and 1 <= int(data_choice) <= len(data_files):
        selected_data = data_files[int(data_choice) - 1]
    else:
        print_error("Invalid choice. Using the first data file.")
        selected_data = data_files[0]

    DATA_PATH = os.path.join(directories["data"], selected_data)
    MODEL_DIR = directories["models"]

    # Load and validate data
    print_info(f"Loading data from {selected_data}...")
    data, labels = load_data(DATA_PATH)
    if data is None or labels is None:
        print_error("Error: Could not load data.")
        exit(1)

    # Convert data to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    print_info(f"Dataset shape: {data.shape}")
    print_info(f"Number of classes: {len(np.unique(labels))}")
    print_info(
        f"Samples per class: {[list(labels).count(i) for i in np.unique(labels)]}"
    )

    # Split the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Train the model
    print_info("\nTraining Random Forest Classifier...")
    with Spinner():
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

    # Evaluate the model performance
    print_info("Evaluating model...")
    score, table = evaluate_model(model, x_test, y_test)

    # Generate comprehensive report
    print_info("Generating detailed model report...")
    comprehensive_report, _ = generate_model_report(
        model, x_test, y_test, y_train, class_labels=None
    )

    # Prompt user for a model name to save model and reports
    model_name = input(
        "Enter a name for saving the trained model (without extension): "
    ).strip()

    if not model_name:
        model_name = "model"

    model_filepath = os.path.join(MODEL_DIR, model_name + ".pkl")
    report_filepath = os.path.join(MODEL_DIR, model_name + ".txt")
    json_filepath = os.path.join(MODEL_DIR, model_name + ".json")

    # Save the model and reports
    print_info("\nSaving model and reports...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save model
    save_data({"model": model}, None, model_filepath)

    # Save text report (existing functionality)
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(table)

    # Save JSON report (new functionality)
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(comprehensive_report, f, indent=2)

    print_info("Model saved as " + model_filepath)
    print_info("Text report saved as " + report_filepath)
    print_info("JSON report saved as " + json_filepath)

    # Ask if user wants to analyze model reports now
    print_info("\nModel training complete!")
    analyze = (
        input("Do you want to analyze the model reports now? (y/n): ").strip().lower()
    )

    if analyze == "y":
        from training.convert_model_reports import convert_existing_model_reports

        convert_existing_model_reports(
            lazy=False
        )  # Use the accurate model testing mode


def convert_existing_model_reports():
    """Convert existing model text reports to JSON format for sidebar display"""
    directories = get_directory_paths()
    MODEL_DIR = directories["models"]

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
    print_info("  q - Quit")
    print_info("  Or enter the number of a specific model to convert")

    while True:
        choice = input("\nEnter your choice: ").strip().lower()
        if choice == "q":
            print_info("Exiting...")
            break
        elif choice == "a":
            total, success = convert_all_models(MODEL_DIR)
            print_info(f"\nConverted {success} of {total} model reports.")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(model_files):
            model_name = model_files[int(choice) - 1]
            success = convert_txt_to_json(model_name, MODEL_DIR)
            if success:
                print_success(f"Successfully converted {model_name}")
            else:
                print_error(f"Failed to convert {model_name}")
        else:
            print_error("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    print_info("\nSign Language Detector - Model Training Module")
    print_info("===========================================")
    print_info("1. Train a new model")
    print_info("2. Convert existing model reports to JSON")
    print_info("3. Exit")
    print_info("===========================================")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            train_classifier()
        elif choice == "2":
            convert_existing_model_reports()
        elif choice == "3":
            break
        else:
            print_error("Invalid choice.")
