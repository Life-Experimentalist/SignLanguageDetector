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
Utils package for Sign Language Detector

This package provides utility functions used throughout the application.
"""

# First import from utils to avoid circular imports
# Import from app_utils for common Flask app functionality
from .app_utils import (
    AppInitializer,
    get_custom_landmark_style,
    handle_signal,
    parse_classification_report,
    process_frame_data,
    shutdown_server,
)

# Then import from config
from .config import (
    ADMIN_KEY,
    BRIGHTNESS_THRESHOLD,
    DATA_DIR,
    DEBUG_MODE,
    DEMO_QUIZ_LETTERS,
    DISABLE_ANONYMOUS_TELEMETRY,
    MAX_WORKERS,
    MODELS_DIR,
    PORT,
    QUIZ_DURATION,
    QUIZ_NUM_GUESSES,
    QUIZ_RELOAD_INTERVAL,
    TELEMETRY_COUNTER_BASE_URL,
    TELEMETRY_PROJECT_NAME,
)
from .utils import (
    calculate_brightness,
    calculate_contrast,
    calculate_saturation,
    convert_numpy_types,
    draw_landmarks,
    get_directory_paths,
    get_labels_dict,
    get_landmark_style,
    get_logger,
    get_two_hand_classes,
    load_data,
    load_model,
    mediapipe_hands,
    print_error,
    print_info,
    print_success,
    print_warning,
    save_data,
)

__all__ = [
    # Functions from utils.py
    "print_info",
    "print_error",
    "print_warning",
    "print_success",
    "calculate_brightness",
    "calculate_contrast",
    "calculate_saturation",
    "mediapipe_hands",
    "draw_landmarks",
    "convert_numpy_types",
    "load_model",
    "save_data",
    "load_data",
    "get_labels_dict",
    "get_two_hand_classes",
    "get_landmark_style",
    "get_logger",
    "get_directory_paths",
    # Variables from config.py
    "DEBUG_MODE",
    "PORT",
    "DATA_DIR",
    "MODELS_DIR",
    "BRIGHTNESS_THRESHOLD",
    "QUIZ_DURATION",
    "QUIZ_NUM_GUESSES",
    "QUIZ_RELOAD_INTERVAL",
    "DEMO_QUIZ_LETTERS",
    "MAX_WORKERS",
    "ADMIN_KEY",
    "DISABLE_ANONYMOUS_TELEMETRY",
    "TELEMETRY_COUNTER_BASE_URL",
    "TELEMETRY_PROJECT_NAME",
    # Classes and functions from app_utils.py
    "AppInitializer",
    "process_frame_data",
    "get_custom_landmark_style",
    "parse_classification_report",
    "shutdown_server",
    "handle_signal",
]
