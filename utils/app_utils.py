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

"""
Shared utility functions for Flask applications in the Sign Language Detector project.
This module contains common functionality used by both app.py and app_multi_client.py.
"""

import base64
import os
import pickle
import re
import signal
import sys
import time
from functools import lru_cache
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
from flask import request
from tqdm import tqdm

# Import directly from sibling modules to avoid circular imports.
from .config import BRIGHTNESS_THRESHOLD, MODELS_DIR
from .utils import (
    calculate_brightness,
    calculate_contrast,
    convert_numpy_types,
    draw_landmarks,
    get_labels_dict,
    get_landmark_style,
    get_two_hand_classes,
    print_error,
    print_info,
)

# Support both legacy and newer MediaPipe package layouts.
try:
    MP_SOLUTIONS = mp.solutions  # type: ignore[attr-defined]
except AttributeError:
    from mediapipe.python import solutions as MP_SOLUTIONS  # type: ignore

# MediaPipe drawing utilities
mp_drawing = MP_SOLUTIONS.drawing_utils


class AppInitializer:
    """Handles application initialization tasks like loading models and setup."""

    def __init__(self):
        self.initialization_progress = 0
        self.shutdown_flag = False
        self.model = None
        self.model_path = None
        self.hands = None
        self.landmark_style = None
        self.connection_style = None
        self.labels_dict = None
        self.two_hand_classes = None
        self.progress_thread = None

    def update_progress_bar(self):
        """Update progress bar during initialization"""
        with tqdm(total=100, desc="Initializing application", ncols=100) as pbar:
            while self.initialization_progress < 100 and not self.shutdown_flag:
                if pbar.n < self.initialization_progress:
                    pbar.update(self.initialization_progress - pbar.n)
                time.sleep(0.01)
            if pbar.n < 100:
                pbar.update(100 - pbar.n)

    @lru_cache(maxsize=None)
    def load_model(self, model_path):
        """Load the ML model with improved error handling"""
        try:
            print_info(f"Loading model from: {model_path}")
            self.initialization_progress += 15
            with open(model_path, "rb") as f:
                model_dict = pickle.load(f)
            self.initialization_progress += 15
            return model_dict["data"]["model"]
        except FileNotFoundError:
            print_error(f"Model file not found at: {model_path}")
            sys.exit(1)
        except KeyError as e:
            print_error(f"Invalid model structure. Missing key: {e}")
            sys.exit(1)
        except Exception as e:
            print_error(f"Error loading model: {e}")
            sys.exit(1)

    def initialize(self, model_path=None):
        """Initialize all required resources"""
        if model_path is None:
            self.model_path = os.path.join(MODELS_DIR, "model.pkl")
        else:
            self.model_path = model_path

        if not os.path.exists(self.model_path):
            print_error(f"Model file not found: {self.model_path}")
            sys.exit(1)

        # Start progress bar in a separate thread
        self.progress_thread = Thread(target=self.update_progress_bar)
        self.progress_thread.daemon = True
        self.progress_thread.start()

        # Load model and initialize resources
        self.model = self.load_model(self.model_path)
        self.initialization_progress += 10
        self.labels_dict = get_labels_dict()
        self.initialization_progress += 5
        self.two_hand_classes = get_two_hand_classes()
        self.initialization_progress += 5

        # Initialize MediaPipe
        mp_hands = MP_SOLUTIONS.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2
        )
        self.initialization_progress += 10

        # Get landmark drawing styles
        self.landmark_style, self.connection_style = get_landmark_style()
        self.initialization_progress += 20

        self.initialization_progress = 100  # Complete the progress bar
        print_info("Application initialized successfully")

        return self.model, self.hands, self.landmark_style, self.connection_style


def process_frame_data(
    frame_data,
    options,
    model,
    hands,
    labels_dict,
    two_hand_classes,
    landmark_style,
    connection_style,
):
    """
    Process a frame received from the client

    Args:
        frame_data (str): Base64 encoded image data
        options (dict): Options for frame processing including landmark display preferences
        model: The ML model for prediction
        hands: MediaPipe hands object
        labels_dict: Dictionary mapping numeric labels to letters
        two_hand_classes: Set of classes that require two hands
        landmark_style: Drawing style for landmarks
        connection_style: Drawing style for connections between landmarks

    Returns:
        dict: Processed frame data with predictions
    """
    try:
        # Set default options if not provided
        if options is None:
            options = {"showLandmarks": True, "landmarkStyle": "default"}

        show_landmarks = options.get(
            "showLandmarks", options.get("show_landmarks", True)
        )
        landmark_style_opt = options.get(
            "landmarkStyle", options.get("landmark_style", "default")
        )

        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(",")[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Keep a copy of the original frame if needed
        original_frame = frame.copy() if not show_landmarks else None

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(frame_rgb)

        # Calculate metrics
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        low_brightness = brightness < BRIGHTNESS_THRESHOLD

        # Draw landmarks on the frame if hands are detected and landmarks should be shown
        if (
            results.multi_hand_landmarks
            and show_landmarks
            and landmark_style_opt != "none"
        ):
            if len(results.multi_hand_landmarks) > 2:
                results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

            # Get landmark style based on option
            if landmark_style_opt == "custom":
                # Get custom landmark style
                current_landmark_style, current_connection_style = (
                    get_custom_landmark_style()
                )
            else:
                # Use default style
                current_landmark_style, current_connection_style = (
                    landmark_style,
                    connection_style,
                )

            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(
                    frame,
                    hand_landmarks,
                    MP_SOLUTIONS.hands.HAND_CONNECTIONS,
                    current_landmark_style,
                    current_connection_style,
                )

        # Make predictions if hands are detected
        data_aux = []
        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

            # Handle single hand case
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))

            if data_aux:
                data_aux = np.asarray(data_aux)
                prediction = model.predict([data_aux])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                # Check if it's a two-hand class but only one hand is detected
                if (
                    predicted_character in two_hand_classes
                    and len(results.multi_hand_landmarks) < 2
                ):
                    predicted_character = ""

        # Prepare response with both processed and original frames if needed
        response = {
            "prediction": predicted_character,
            "brightness": convert_numpy_types(brightness),
            "contrast": convert_numpy_types(contrast),
            "low_brightness": convert_numpy_types(low_brightness),
        }

        # Encode processed image to base64
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode(".jpg", frame, encode_param)
        processed_frame = base64.b64encode(buffer).decode("utf-8")
        response["processed_frame"] = f"data:image/jpeg;base64,{processed_frame}"

        # Include original frame if landmarks are disabled
        if not show_landmarks or landmark_style_opt == "none":
            if original_frame is None:
                original_frame = frame.copy()
            ret, buffer = cv2.imencode(".jpg", original_frame, encode_param)
            original_base64 = base64.b64encode(buffer).decode("utf-8")
            response["original_frame"] = f"data:image/jpeg;base64,{original_base64}"

        return response

    except Exception as e:
        print_error(f"Error processing frame: {str(e)}")
        return {"error": str(e)}


def get_custom_landmark_style():
    """Get custom landmark drawing style based on environment variables."""
    landmark_color = tuple(map(int, os.getenv("LANDMARK_COLOR", "0,255,0").split(",")))
    connection_color = tuple(
        map(int, os.getenv("CONNECTION_COLOR", "0,0,255").split(","))
    )
    thickness = int(os.getenv("LANDMARK_THICKNESS", "2"))
    circle_radius = int(os.getenv("LANDMARK_CIRCLE_RADIUS", "2"))

    return (
        mp_drawing.DrawingSpec(
            color=landmark_color, thickness=thickness, circle_radius=circle_radius
        ),
        mp_drawing.DrawingSpec(color=connection_color, thickness=thickness),
    )


def parse_classification_report(report_text):
    """
    Parse a classification report text into structured data.

    This function takes a text-based classification report (typically from sklearn's
    classification_report function) and converts it into a structured dictionary
    for easier programmatic access.

    Args:
        report_text (str): The classification report text to parse

    Returns:
        dict: A structured dictionary containing metrics like:
            - class_report: Per-class metrics (precision, recall, f1)
            - metrics: Overall metrics
            - accuracy: Overall accuracy
    """
    result = {"class_report": {}, "metrics": {}}

    # Try to extract accuracy
    accuracy_match = re.search(r"Accuracy\s+(\d+\.\d+)", report_text)
    if accuracy_match:
        result["metrics"]["accuracy"] = float(accuracy_match.group(1))

    # Parse class-specific metrics
    lines = report_text.strip().split("\n")
    header_found = False

    for line in lines:
        line = line.strip()

        # Skip empty lines and separator lines
        if not line or line.startswith("---") or line.startswith("==="):
            continue

        # Find header line
        if not header_found and ("Class" in line or "Precision" in line):
            header_found = True
            continue

        # Parse metrics lines
        parts = re.split(r"\s+", line)
        if len(parts) >= 4:
            try:
                cls = parts[0]

                # Handle metrics
                if cls in ["Accuracy", "macro avg", "weighted avg"]:
                    if cls == "Accuracy" and len(parts) >= 2:
                        result["metrics"]["accuracy"] = float(parts[-1])
                    elif len(parts) >= 4:
                        result["metrics"][cls] = {
                            "precision": float(parts[-3]) if parts[-3] != "-" else None,
                            "recall": float(parts[-2]) if parts[-2] != "-" else None,
                            "f1": float(parts[-1]) if parts[-1] != "-" else None,
                        }
                # Handle class reports
                else:
                    result["class_report"][cls] = {
                        "precision": float(parts[-3]) if parts[-3] != "-" else None,
                        "recall": float(parts[-2]) if parts[-2] != "-" else None,
                        "f1Score": float(parts[-1]) if parts[-1] != "-" else None,
                    }
            except (ValueError, IndexError):
                continue

    # If we didn't successfully parse anything meaningful, return None
    if (
        not result["class_report"]
        and not result["metrics"]
        and "accuracy" not in result
    ):
        return None

    return result


def shutdown_server(shutdown_flag, app=None):
    """Properly shutdown the server and release resources"""
    if not shutdown_flag:
        print_info("Shutting down server...")

        # Shutdown Flask
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            print_error("Not running with the Werkzeug Server")
            return
        func()
        print_info("Server shutdown complete.")
        return True
    return False


def handle_signal(sig, frame, shutdown_flag, app=None):
    """Handle system signals gracefully"""
    signal_name = "UNKNOWN"
    if sig == signal.SIGINT:
        signal_name = "SIGINT"
    elif sig == signal.SIGTERM:
        signal_name = "SIGTERM"

    print_info(f"Received {signal_name} signal. Initiating graceful shutdown...")
    shutdown_server(shutdown_flag, app)
    sys.exit(0)
