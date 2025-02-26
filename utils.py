import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from colorama import Fore, Style
import re
import itertools
from dotenv import load_dotenv

# Import MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils  # type: ignore

# Load variables from .env in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# Expose environment variables for other modules
IMAGES_PER_CLASS = int(os.getenv("IMAGES_PER_CLASS", "500"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "26"))
DATA_DIR = os.getenv(
    "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
)
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "100"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
PORT = int(os.getenv("PORT", "5000"))
BRIGHTNESS_THRESHOLD = float(os.getenv("BRIGHTNESS_THRESHOLD", "85"))


def get_project_dir():
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.abspath(__file__))


def get_directory_paths():
    """Return paths for logs, models, data, and templates; create them if missing."""
    project_dir = get_project_dir()
    directories = {
        "logs": os.path.join(project_dir, "logs"),
        "models": os.path.join(project_dir, "models"),
        "data": os.path.join(project_dir, "data"),
        "templates": os.path.join(project_dir, "templates"),
    }
    for name, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
    return directories


def calculate_brightness(frame):
    """Calculate the brightness of the frame"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


def calculate_contrast(frame):
    """Calculate the contrast of the frame"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    return l_channel.std()


def load_model(model_path):
    """Load a machine learning model from the specified path."""
    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict["models"]
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def mediapipe_hands():
    """Initialize and return the MediaPipe Hands object."""
    mp_hands = mp.solutions.hands  # type: ignore
    return mp_hands.Hands(
        static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2
    )


def draw_landmarks(
    image, hand_landmarks, connections, landmark_style, connection_style
):
    """Draw landmarks on the given image."""
    mp_drawing.draw_landmarks(
        image, hand_landmarks, connections, landmark_style, connection_style
    )


def create_directories(dirs):
    """Create directories if they don't exist."""
    for name, path in dirs.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")


def save_data(data, labels, filepath):
    """Save data and labels to a pickle file."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")


def load_data(filepath):
    """Load data and labels from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            d = pickle.load(f)
        return d["data"], d["labels"]
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None, None


def convert_numpy_types(item):
    """Recursively convert numpy types to native Python types."""
    if isinstance(item, np.generic):
        return item.item()
    elif isinstance(item, dict):
        return {k: convert_numpy_types(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types(x) for x in item]
    return item


def format_numbers(msg, base_color):
    """
    Wrap numbers in the message with a different color (magenta) and then resume the base color.
    """
    return re.sub(r"(\d+)", lambda m: Fore.MAGENTA + m.group(0) + base_color, msg)


def print_info(msg):
    base = Fore.CYAN
    formatted = format_numbers(msg, base)
    print(base + formatted + Style.RESET_ALL)


def print_warning(msg):
    base = Fore.YELLOW
    formatted = format_numbers(msg, base)
    print(base + formatted + Style.RESET_ALL)


def print_error(msg):
    base = Fore.RED
    formatted = format_numbers(msg, base)
    print(base + formatted + Style.RESET_ALL)


class Spinner:
    """
    Context manager to handle spinner animation.
    When entered, it hides the cursor and provides cyclic spinner symbols.
    On exit, it re-enables the cursor.
    """

    def __init__(self, symbols=None):
        self.symbols = symbols if symbols is not None else ["-", "/", "|", "\\"]
        self._cycle = itertools.cycle(self.symbols)

    def __enter__(self):
        print("\033[?25l", end="")  # Hide cursor
        return self

    def next(self):
        return next(self._cycle)

    def __exit__(self, exc_type, exc_value, traceback):
        print("\033[?25h", end="")  # Show cursor
