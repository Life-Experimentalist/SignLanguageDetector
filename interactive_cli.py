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
from training.create_dataset import create_dataset
from training.train_classifier import train_classifier
from training.inference_classifier import run_inference
from utils import print_info, print_warning


def training_pipeline():
    print_info("Training Pipeline Started")
    print_warning(
        "This pipeline will guide you through the data collection, dataset creation, and classifier training process."
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
        "Stage 4: Inference - Tests the trained model by making predictions on new input"
    )
    print_info("Let's begin!")
    print_warning("Please ensure that you have a webcam connected to the system.")
    print_info("If you have already collected images, you can skip the first stage.")
    cont = (
        input("Proceed with the First Stage for Dataset Collection? (y/n): ")
        .strip()
        .lower()
    )
    if cont == "y":
        # Stage 1: Data Collection
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
    if cont == "y":
        # Stage 2: Create Dataset
        print_info("\nStage 2: Create Dataset (create_dataset)")
        create_dataset()
    else:
        print_info("Skipping dataset creation. Proceeding to Train Classifier...")

    cont = (
        input("Proceed with the Third Stage to train classifier? (y/n): ")
        .strip()
        .lower()
    )
    if cont != "y":
        print_info("Exiting pipeline.")
        return

    # Stage 3: Train Classifier
    print_info("\nStage 3: Train Classifier (train_classifier)")
    train_classifier()

    cont = (
        input("Classifier training complete. Do you want to run inference? (y/n): ")
        .strip()
        .lower()
    )
    if cont == "y":
        print_info("Running Inference...")
        run_inference()
    else:
        print_info("Exiting pipeline.")


if __name__ == "__main__":
    training_pipeline()
