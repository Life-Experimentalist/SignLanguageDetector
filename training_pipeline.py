# Add a new version of the pipeline with the convert_model_reports step
from training.inference_classifier import run_inference
from utils import print_info, print_warning


def training_pipeline():
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

    # ... existing code for stages 1-3 ...

    # Stage 4: Model Analysis (after training)
    cont = (
        input(
            "Classifier training complete. Do you want to analyze the model performance? (y/n): "
        )
        .strip()
        .lower()
    )
    if cont == "y":
        print_info("\nStage 4: Model Analysis (convert_model_reports)")
        from training.convert_model_reports import convert_existing_model_reports

        convert_existing_model_reports(lazy=False)  # Use accurate model testing mode

    # Stage 5: Inference
    cont = (
        input("Do you want to run inference with your model? (y/n): ").strip().lower()
    )
    if cont == "y":
        print_info("\nStage 5: Running Inference...")
        run_inference()

    print_info("Pipeline completed successfully!")
