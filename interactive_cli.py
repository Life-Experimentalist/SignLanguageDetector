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

from training.collect_imgs import create_img
from training.convert_model_reports import (
    convert_all_models,
    convert_existing_model_reports,
    convert_txt_to_json,
    generate_missing_json_reports,
)
from training.create_dataset import create_dataset
from training.inference_classifier import run_inference
from training.train_classifier import train_classifier

# Import directly from utils package (no circular dependency)
from utils import print_error, print_info, print_success, print_warning


def model_tools_menu():
    """Menu for model manipulation tools"""
    print_info("\nModel Tools")
    print_info("===========")
    print_info("1. Train a new model")
    print_info("2. Generate model reports")
    print_info("3. Test model with inference")
    print_info("4. Return to main menu")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        train_classifier()
    elif choice == "2":
        print_info("\nModel Report Generation Options:")
        print_info("1. Standard mode - Test models and generate accurate reports")
        print_info("2. Fast mode - Use existing text reports (less comprehensive)")

        report_mode = input("\nSelect mode (1-2): ").strip()

        if report_mode == "1":
            convert_existing_model_reports(lazy=False)
        elif report_mode == "2":
            convert_existing_model_reports(lazy=True)
        else:
            print_error("Invalid choice")
    elif choice == "3":
        run_inference()
    elif choice == "4":
        return
    else:
        print_error("Invalid choice")


# Rename enhanced_training_pipeline to just training_pipeline
def training_pipeline():
    """
    Complete training pipeline that includes model analysis stage.
    """
    print_info("Training Pipeline Started")
    print_warning(
        "This pipeline will guide you through the complete model development process."
    )
    print("Training Pipeline Details:")
    print_info(
        "Stage 1: Data Collection - Captures images from your webcam to build a sign language dataset"
    )
    print_info(
        "Stage 2: Dataset Creation - Organizes and processes the collected images into training data"
    )
    print_info(
        "Stage 3: Classifier Training - Trains a machine learning model on your dataset"
    )
    print_info(
        "Stage 4: Model Analysis - Analyzes model performance and generates reports"
    )
    print_info(
        "Stage 5: Inference - Tests the trained model by making predictions on new input"
    )
    print_info("Let's begin!")
    print_warning("Please ensure that you have a webcam connected to the system.")
    print_info("If you have already collected images, you can skip the first stage.")

    # Stage 1: Data Collection
    cont = (
        input("Proceed with the First Stage for Dataset Collection? (y/n): ")
        .strip()
        .lower()
    )
    if cont == "y":
        print_info("\nStage 1: Data Collection (collect_imgs)")
        create_img()

        cont = (
            input(
                "Data collection complete. Do you want to proceed with the Second Stage for Dataset Creation? (y/n): "
            )
            .strip()
            .lower()
        )
    else:
        cont = (
            input("Proceed with the Second Stage for Dataset Creation? (y/n): ")
            .strip()
            .lower()
        )

    # Stage 2: Dataset Creation
    if cont == "y":
        print_info("\nStage 2: Create Dataset (create_dataset)")
        create_dataset()
    else:
        print_info("Skipping dataset creation. Proceeding to Train Classifier...")

    # Stage 3: Classifier Training
    cont = (
        input("Proceed with the Third Stage to train classifier? (y/n): ")
        .strip()
        .lower()
    )
    if cont != "y":
        print_info("Exiting pipeline.")
        return

    print_info("\nStage 3: Train Classifier (train_classifier)")
    train_classifier()

    # NEW - Stage 4: Model Analysis
    cont = (
        input(
            "Classifier training complete. Do you want to analyze the model performance? (y/n): "
        )
        .strip()
        .lower()
    )
    if cont == "y":
        print_info("\nStage 4: Model Analysis (convert_model_reports)")
        convert_existing_model_reports(
            lazy=False
        )  # Use accurate model testing mode by default

    # Stage 5: Inference
    cont = (
        input("Do you want to run inference with your model? (y/n): ").strip().lower()
    )
    if cont == "y":
        print_info("\nStage 5: Running Inference...")
        run_inference()

    print_info("Pipeline completed successfully!")


# Update main menu with option for enhanced pipeline
def main_menu():
    """Main menu for the interactive CLI"""
    while True:
        print_info("\nSign Language Detector - Interactive CLI")
        print_info("=====================================")
        print_info("1. Training Pipeline")
        print_info("2. Model Tools")
        print_info("3. Exit")

        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            training_pipeline()  # Renamed to simply "training_pipeline"
        elif choice == "2":
            model_tools_menu()
        elif choice == "3":
            print_info("Exiting...")
            break
        else:
            print_error("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()
