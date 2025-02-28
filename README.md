# Sign Language Detector

A real-time Flask application for detecting and recognizing Indian Sign Language gestures using machine learning and computer vision.

## Project Overview

This project uses MediaPipe for hand landmark detection and machine learning to recognize Indian Sign Language (ISL) alphabet. It includes:

1. Data Collection (collect_imgs.py)
2. Dataset Creation (create_dataset.py)
3. Model Training (train_classifier.py)
4. Real-time Inference (inference_classifier.py)
5. An Interactive CLI tool for training operations (interactive_cli.py)
6. A Flask web application (app.py)

The training component can be used for any hand gesture recognition, not limited to sign language.

## Directory Structure

```
SignLanguageDetector/
├── app.py                       # Flask web application
├── training/                    # Training scripts
│   ├── collect_imgs.py          # Data collection script
│   ├── create_dataset.py        # Dataset creation script
│   ├── train_classifier.py      # Model training script
│   ├── inference_classifier.py  # Inference script
├── interactive_cli.py           # CLI tool for training pipeline
├── data/                        # Training data directory
├── models/                      # Saved model files directory
├── logs/                        # Application logs directory
├── templates/                   # Web application templates
│   └── index.html               # Main web interface
├── test_cv.py                   # Test OpenCV installation
└── README.md                    # Project overview and usage

```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd SignLanguageDetector
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install numpy opencv-python mediapipe flask scikit-learn colorama paho-mqtt
   ```
4. Install Mosquitto:
   - Follow the instructions on the [Mosquitto website](https://mosquitto.org/download/) to install Mosquitto on your system.

## Usage

### Data Collection

```
python training/collect_imgs.py
```

### Create Dataset

```
python training/create_dataset.py
```

### Train Classifier

```
python training/train_classifier.py
```

### Test Inference

```
python training/inference_classifier.py
```

### Interactive CLI

```
python interactive_cli.py
```

### Run Web Application

```
python app.py
```

Then open your browser at `http://127.0.0.1:5000`.

## Logging

Logs are stored in `logs/`, organized by session timestamps. Files include:

- performance.log (timing data)
- debug.log (debug messages)
- error.log (errors)
- access.log (HTTP access logs)

## Notes

- Ensure proper lighting for improved detection.
- A physical camera is required.
- Some gestures need two hands for accurate recognition.
