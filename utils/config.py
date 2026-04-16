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
Configuration module for the Sign Language Detector application.

This module centralizes configuration parameters loaded from environment variables.
"""

import ast
import logging
import os

from dotenv import load_dotenv

# Load .env file from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, ".env"))

# App settings
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))

# Path configurations
DATA_DIR = os.getenv("DATA_DIR", os.path.join(project_root, "data"))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(project_root, "models"))
LOGS_DIR = os.getenv("LOGS_DIR", os.path.join(project_root, "logs"))
TEMPLATES_DIR = os.getenv("TEMPLATES_DIR", os.path.join(project_root, "templates"))

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, TEMPLATES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Training parameters
IMAGES_PER_CLASS = int(os.getenv("IMAGES_PER_CLASS", "500"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "26"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Camera and image processing settings
BRIGHTNESS_THRESHOLD = float(os.getenv("BRIGHTNESS_THRESHOLD", "85"))

# Quiz settings
QUIZ_DURATION = int(os.getenv("QUIZ_DURATION", "30"))
QUIZ_NUM_GUESSES = int(os.getenv("QUIZ_NUM_GUESSES", "5"))
QUIZ_RELOAD_INTERVAL = int(os.getenv("QUIZ_RELOAD_INTERVAL", "2000"))
DEMO_QUIZ_LETTERS = os.getenv("DEMO_QUIZ_LETTERS", "ABCDE").upper().strip()

# Multi-client settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
ADMIN_KEY = os.getenv("ADMIN_KEY", "admin123")

# Telemetry settings (anonymous counter integration)
_disable_telemetry_value = os.getenv(
    "DISABLE_ANONYMOUS_TELEMETRY",
    os.getenv("DISABLE_ANONYMOUS_TELMETRY", "false"),
)
DISABLE_ANONYMOUS_TELEMETRY = _disable_telemetry_value.lower() == "true"
TELEMETRY_COUNTER_BASE_URL = os.getenv(
    "TELEMETRY_COUNTER_BASE_URL", "https://counter.vkrishna04.me"
).rstrip("/")
TELEMETRY_PROJECT_NAME = os.getenv("TELEMETRY_PROJECT_NAME", "sign-language-detector")


# Import from utils.py without creating circular dependencies
def get_logger(name):
    """Set up logger with proper formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create a file handler
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)

        log_file = os.path.join(LOGS_DIR, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_directory_paths():
    """Return paths for logs, models, data, and templates."""
    directories = {
        "logs": LOGS_DIR,
        "models": MODELS_DIR,
        "data": DATA_DIR,
        "templates": TEMPLATES_DIR,
    }
    return directories


def get_labels_dict():
    """Get dictionary mapping integer indices to letter labels."""
    raw_labels = os.getenv("LABELS_DICT", "{}")
    try:
        str_dict = ast.literal_eval(raw_labels)
        return {int(k): v for k, v in str_dict.items()}
    except (ValueError, SyntaxError) as e:
        get_logger(__name__).error(f"Error parsing LABELS_DICT: {e}")
        # Default to A-Z mapping if there's an issue
        return {i: chr(65 + i) for i in range(26)}


def get_two_hand_classes():
    """Get set of classes that require two hands."""
    raw_classes = os.getenv("TWO_HAND_CLASSES", "[]")
    try:
        return set(ast.literal_eval(raw_classes))
    except (ValueError, SyntaxError) as e:
        get_logger(__name__).error(f"Error parsing TWO_HAND_CLASSES: {e}")
        return set()


def get_landmark_style():
    """Import and use the implementation from utils to avoid duplication."""
    # This is a forward reference that will be resolved at runtime
    # to avoid circular imports
    from .utils import get_landmark_style as utils_get_landmark_style

    return utils_get_landmark_style()
