import logging
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, request

from utils import (
    get_directory_paths,
    convert_numpy_types,
    DATA_DIR,
    MODELS_DIR,
    PORT,
    BRIGHTNESS_THRESHOLD,
)

# Retrieve required directories
directories = get_directory_paths()

# Setup logging
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
session_dir = os.path.join(logs_dir, current_time)
if not os.path.exists(session_dir):
    os.makedirs(session_dir)

log_file_paths = {
    "performance": os.path.join(session_dir, "performance.log"),
    "debug": os.path.join(session_dir, "debug.log"),
    "error": os.path.join(session_dir, "error.log"),
}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file_paths["performance"]),
        logging.FileHandler(log_file_paths["debug"]),
        logging.FileHandler(log_file_paths["error"]),
    ],
)

perf_logger = logging.getLogger("performance")
debug_logger = logging.getLogger("debug")
error_logger = logging.getLogger("error")

# Remove any default console handlers to keep output clean.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Redirect werkzeug logs to file
werkzeug_logger = logging.getLogger("werkzeug")
for handler in werkzeug_logger.handlers[:]:
    werkzeug_logger.removeHandler(handler)
werkzeug_handler = logging.FileHandler(os.path.join(session_dir, "access.log"))
werkzeug_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
werkzeug_handler.setLevel(logging.INFO)
werkzeug_logger.addHandler(werkzeug_handler)

app = Flask(__name__)

def restart_program():
    """Restart the current program, excluding keyboard interrupt"""
    try:
        subprocess.run([sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("Manual shutdown requested")
        error_logger.error("Manual shutdown requested")
        sys.exit(0)

class CameraManager:
    def __init__(self):
        self.camera = None
        self.last_error_time = 0
        self.error_count = 0

    def get_camera(self):
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Camera not accessible")
            return self.camera
        except Exception as e:
            current_time_ = time.time()
            if current_time_ - self.last_error_time > 60:
                self.error_count = 0
            self.error_count += 1
            self.last_error_time = current_time_
            if self.error_count > 5:
                debug_logger.debug("Too many camera errors, restarting program...")
                restart_program()
            debug_logger.debug(f"Camera error: {e}, attempting to reconnect...")
            self.release()
            time.sleep(2)
            return self.get_camera()

    def release(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def read_frame(self):
        start = time.time()
        cam = self.get_camera()
        success, frame = cam.read()
        if success:
            duration = (time.time() - start) * 1000
            perf_logger.info(f"Camera read time: {duration:.1f}ms")
        return success, frame


camera_manager = CameraManager()

# Load model from models directory
model_files = [f for f in os.listdir(directories["models"]) if f.endswith(".p")]
selected_model = None
model = None
if model_files:
    model_path = os.path.join(directories["models"], model_files[0])
    model_dict = pickle.load(open(model_path, "rb"))
    # Adjust to key name as expected. If key isn't "models", update accordingly.
    model = model_dict.get("models")
    selected_model = model_files[0]

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

hands = mp_hands.Hands(
    static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2
)

labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}
two_hand_classes = {
    "A",
    "B",
    "D",
    "E",
    "F",
    "G",
    "H",
    "K",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "X",
    "Y",
    "Z",
}


def calculate_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


def calculate_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    return l_channel.std()


def process_frame(frame, model):
    try:
        start_total = time.time()
        t0 = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        perf_logger.info(f"RGB conversion: {(time.time()-t0)*1000:.1f}ms")
        brightness = calculate_brightness(frame)
        low_brightness = brightness < BRIGHTNESS_THRESHOLD
        contrast = calculate_contrast(frame)
        t0 = time.time()
        results = hands.process(frame_rgb)
        perf_logger.info(f"Hand detection: {(time.time()-t0)*1000:.1f}ms")
        data_aux = []
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 2:
                results.multi_hand_landmarks = results.multi_hand_landmarks[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(xs), min(ys)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
            t0 = time.time()
            perf_logger.info(f"Landmark processing: {(time.time()-t0)*1000:.1f}ms")
            if data_aux and not low_brightness:
                data_aux = np.asarray(data_aux)
                t0 = time.time()
                prediction = model.predict([data_aux])
                perf_logger.info(f"Model prediction: {(time.time()-t0)*1000:.1f}ms")
                predicted_character = labels_dict[int(prediction[0])]
                if (
                    predicted_character in two_hand_classes
                    and len(results.multi_hand_landmarks) < 2
                ):
                    return (
                        frame,
                        "Error: Two hands required",
                        brightness,
                        low_brightness,
                        contrast,
                    )
                return frame, predicted_character, brightness, low_brightness, contrast
        perf_logger.info(f"Total frame time: {(time.time()-start_total)*1000:.1f}ms")
        perf_logger.info("-" * 50)
        return frame, "", brightness, low_brightness, contrast
    except Exception as e:
        error_logger.error(f"Frame processing error: {e}")
        return frame, "Error: Processing failed", 0, False, 0

def generate_frames(model):
    while True:
        try:
            camera = camera_manager.get_camera()
            while True:
                success, frame = camera.read()
                if not success:
                    raise Exception("Failed to read frame")
                processed_frame, prediction, brightness, low_brightness, contrast = (
                    process_frame(frame, model)
                )
                ret, buffer = cv2.imencode(".jpg", processed_frame)
                if not ret:
                    raise Exception("Failed to encode frame")
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_logger.error(f"Stream error: {e}")
            camera_manager.release()
            time.sleep(1)
            continue

prediction_lock = threading.Lock()
last_prediction = {
    "result": "",
    "timestamp": 0,
    "brightness": 0,
    "low_brightness": False,
}
selected_model_lock = threading.Lock()
no_hand_detected_since = None


@lru_cache(maxsize=1)
def get_cached_prediction(timestamp, model):
    try:
        success, frame = camera_manager.read_frame()
        if success:
            _, prediction, brightness, low_brightness, contrast = process_frame(
                frame, model
            )
            return prediction, brightness, low_brightness, contrast
    except Exception as e:
        error_logger.error(f"Prediction error: {e}")
    return "Error", 0, False, 0


@app.route("/video_feed")
def video_feed():
    global model
    return Response(
        generate_frames(model), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/predictions")
def predictions():
    global model, no_hand_detected_since
    current_time = time.time()
    with prediction_lock:
        if current_time - last_prediction["timestamp"] > 0.1:
            (
                last_prediction["result"],
                last_prediction["brightness"],
                last_prediction["low_brightness"],
                last_prediction["contrast"],
            ) = get_cached_prediction(int(current_time * 10), model)
            last_prediction["brightness"] = float(last_prediction["brightness"])
            last_prediction["low_brightness"] = bool(last_prediction["low_brightness"])
            last_prediction["contrast"] = float(last_prediction["contrast"])
            last_prediction["timestamp"] = current_time
            if last_prediction["result"] == "":
                if no_hand_detected_since is None:
                    no_hand_detected_since = current_time
                elif current_time - no_hand_detected_since > 3:
                    last_prediction["result"] = "Waiting..."
            else:
                no_hand_detected_since = None
        response_data = {
            "prediction": last_prediction["result"],
            "brightness": last_prediction["brightness"],
            "low_brightness": last_prediction["low_brightness"],
            "contrast": last_prediction["contrast"],
        }
        return convert_numpy_types(response_data)

@app.route("/")
def index():
    global model_files, selected_model
    return render_template(
        "index.html", model_files=model_files, selected_model=selected_model
    )

@app.route("/select_model/<model_name>")
def select_model(model_name):
    global model, selected_model
    model_path = os.path.join(directories["models"], model_name)
    try:
        model_dict = pickle.load(open(model_path, "rb"))
        model = model_dict.get("models")
        with selected_model_lock:
            selected_model = model_name
        return {"status": "Model selected successfully"}
    except Exception as e:
        error_logger.error(f"Error loading model: {e}")
        return {"status": "Error loading model"}

@app.route("/shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        error_logger.error("Server shutdown function not available. Forcing shutdown.")
        handle_shutdown(None, None)
        return "Server forced to shutdown."
    error_logger.info("Shutdown endpoint called. Shutting down server.")
    func()
    return "Server shutting down..."

def handle_shutdown(signum, frame):
    error_logger.info("Shutting down...")
    camera_manager.release()
    for log_type, log_path in log_file_paths.items():
        if os.path.exists(log_path) and os.path.getsize(log_path) == 0:
            os.remove(log_path)
            error_logger.info(f"Removed empty log file: {log_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

def main():
    try:
        app.run(debug=False, threaded=True, port=PORT)
    except KeyboardInterrupt:
        error_logger.error("Keyboard interrupt received, stopping server.")
    except Exception as e:
        error_logger.error(f"Server error: {e}")
    finally:
        handle_shutdown(None, None)
        sys.exit(0)

if __name__ == "__main__":
    main()
