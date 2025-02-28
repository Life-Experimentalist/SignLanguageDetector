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

# Add the parent directory (project root) to sys.path so that utils can be found.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tabulate import tabulate  # pip install tabulate if needed

from utils import (
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


def train_classifier():
    # Get the project directory and paths
    directories = get_directory_paths()
    DATA_PATH = os.path.join(directories["data"], "data.pickle")
    MODEL_DIR = directories["models"]

    # Load and validate data
    print_info("Loading data...")
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
    with Spinner() as spinner:
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)

    # Evaluate the model performance
    print_info("Evaluating model...")
    score, table = evaluate_model(model, x_test, y_test)

    # Prompt user for a model name to save model and report
    model_name = input(
        "Enter a name for saving the trained model (without extension): "
    ).strip()
    if not model_name:
        model_name = "model"
    model_filepath = os.path.join(MODEL_DIR, model_name + ".p")
    report_filepath = os.path.join(MODEL_DIR, model_name + ".txt")

    # Save the model
    print_info("\nSaving model and report...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    save_data({"model": model}, None, model_filepath)
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write(table)
    print_info("Model saved as " + model_filepath)
    print_info("Report saved as " + report_filepath)


if __name__ == "__main__":
    train_classifier()
