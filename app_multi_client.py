# Project: Sign Language Detector - Multi-client Version
# Repository: https://github.com/Life-Experimentalist/SignLanguageDetector
# Owner: VKrishna04
# Organization: Life-Experimentalist
# Licensed under the Apache License, Version 2.0

import base64

# Import for performance tuning
import concurrent.futures
import os
import pickle
import re
import signal
import sys
import time
import uuid
from functools import lru_cache
from threading import Lock, Thread
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from flask import (
    Flask,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)
from flask_session import Session  # Add Flask-Session for better session management
from tqdm import tqdm

try:
    MP_SOLUTIONS = mp.solutions  # type: ignore[attr-defined]
except AttributeError:
    from mediapipe.python import solutions as MP_SOLUTIONS  # type: ignore

# Import directly from utils package
from utils import (
    ADMIN_KEY,
    BRIGHTNESS_THRESHOLD,
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
    calculate_brightness,
    calculate_contrast,
    convert_numpy_types,
    draw_landmarks,
    get_directory_paths,
    get_labels_dict,
    get_landmark_style,
    get_logger,
    get_two_hand_classes,
    mediapipe_hands,
    print_error,
    print_info,
)

# Setup logger
logger = get_logger(__name__)

app = Flask(__name__)


@app.context_processor
def inject_telemetry_config():
    return {
        "disable_anonymous_telemetry": DISABLE_ANONYMOUS_TELEMETRY,
        "telemetry_counter_base_url": TELEMETRY_COUNTER_BASE_URL,
        "telemetry_project_name": TELEMETRY_PROJECT_NAME,
    }


@app.context_processor
def inject_label_config():
    return {"labels_dict": labels_dict}


# Configure server-side sessions
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flask_sessions"
)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SECRET_KEY"] = os.urandom(24)
Session(app)

# Global flags and state
shutdown_flag = False
initialization_progress = 0
model_lock = Lock()  # Lock for model access and modification

# Multi-client process pool for parallel frame processing
process_pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Load model
model_path = os.path.join(MODELS_DIR, "model.pkl")
if not os.path.exists(model_path):
    print_error(f"Model file not found: {model_path}")
    sys.exit(1)

# Create session folder if it doesn't exist
if not os.path.exists(app.config["SESSION_FILE_DIR"]):
    os.makedirs(app.config["SESSION_FILE_DIR"])


def update_progress_bar():
    """Update progress bar during initialization"""
    global initialization_progress
    with tqdm(total=100, desc="Initializing application", ncols=100) as pbar:
        while initialization_progress < 100 and not shutdown_flag:
            if pbar.n < initialization_progress:
                pbar.update(initialization_progress - pbar.n)
            time.sleep(0.01)
        if pbar.n < 100:
            pbar.update(100 - pbar.n)


@lru_cache(maxsize=None)
def load_model(model_path):
    """Load the ML model with improved error handling"""
    global initialization_progress
    try:
        print_info(f"Loading model from: {model_path}")
        initialization_progress += 15
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        initialization_progress += 15
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


# Start progress bar in a separate thread
progress_thread = Thread(target=update_progress_bar)
progress_thread.daemon = True
progress_thread.start()

# Load model and initialize resources
model = load_model(model_path)
initialization_progress += 10
labels_dict = get_labels_dict()
initialization_progress += 5
two_hand_classes = get_two_hand_classes()
initialization_progress += 5
hands = mediapipe_hands()
initialization_progress += 10
landmark_style, connection_style = get_landmark_style()
initialization_progress += 20

initialization_progress = 100  # Complete the progress bar
print_info("Mediapipe initialized successfully")

# Track active users for monitoring
active_users = 0
users_lock = Lock()
client_sessions = {}  # Track individual client sessions


def increment_user_count():
    """Increment active user count and generate unique client ID"""
    global active_users
    with users_lock:
        active_users += 1
        client_id = str(uuid.uuid4())
        client_sessions[client_id] = {"last_active": time.time(), "frames_processed": 0}
        print_info(
            f"New client connected. Client ID: {client_id}. Active users: {active_users}"
        )
        return client_id


def decrement_user_count(client_id):
    """Decrement active user count and remove client session"""
    global active_users
    with users_lock:
        if client_id in client_sessions:
            del client_sessions[client_id]
        active_users = max(0, active_users - 1)
        print_info(
            f"Client disconnected. Client ID: {client_id}. Active users: {active_users}"
        )


def update_client_activity(client_id):
    """Update client's last activity timestamp"""
    with users_lock:
        if client_id in client_sessions:
            client_sessions[client_id]["last_active"] = time.time()
            client_sessions[client_id]["frames_processed"] += 1


def cleanup_inactive_clients():
    """Remove clients who haven't been active in the last 5 minutes"""
    with users_lock:
        now = time.time()
        inactive_threshold = 300  # 5 minutes
        inactive_clients = [
            cid
            for cid, data in client_sessions.items()
            if now - data["last_active"] > inactive_threshold
        ]

        for client_id in inactive_clients:
            print_info(f"Removing inactive client: {client_id}")
            del client_sessions[client_id]
            global active_users
            active_users = max(0, active_users - 1)


# Run session cleanup every few minutes
def cleanup_worker():
    while not shutdown_flag:
        cleanup_inactive_clients()
        time.sleep(60)  # Check every minute


cleanup_thread = Thread(target=cleanup_worker)
cleanup_thread.daemon = True
cleanup_thread.start()


@app.before_request
def track_users():
    """Track user connections"""
    if request.endpoint == "index":
        client_id = request.cookies.get("client_id")
        if not client_id or client_id not in client_sessions:
            client_id = increment_user_count()
            # Client ID will be set in response


def process_frame_data(frame_data, client_id=None):
    """Process a frame received from the client"""
    try:
        # Update client activity if client_id provided
        if client_id and client_id in client_sessions:
            update_client_activity(client_id)

        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(",")[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = hands.process(frame_rgb)

        # Calculate metrics
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        low_brightness = brightness < BRIGHTNESS_THRESHOLD

        # Draw landmarks on the frame if hands are detected
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 2:
                results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks(
                    frame,
                    hand_landmarks,
                    MP_SOLUTIONS.hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style,
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
                with model_lock:  # Use lock when accessing the model
                    prediction = model.predict([data_aux])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                # Check if it's a two-hand class but only one hand is detected
                if (
                    predicted_character in two_hand_classes
                    and len(results.multi_hand_landmarks) < 2
                ):
                    predicted_character = ""

        # Encode processed image to base64
        # Use lower quality for faster transfer
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        ret, buffer = cv2.imencode(".jpg", frame, encode_param)
        processed_frame = base64.b64encode(buffer).decode("utf-8")

        return {
            "processed_frame": f"data:image/jpeg;base64,{processed_frame}",
            "prediction": predicted_character,
            "brightness": convert_numpy_types(brightness),
            "contrast": convert_numpy_types(contrast),
            "low_brightness": convert_numpy_types(low_brightness),
            "client_id": client_id,
        }

    except Exception as e:
        print_error(f"Error processing frame: {str(e)}")
        return {"error": str(e), "client_id": client_id}


@app.route("/")
def index():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".p")]
    selected_model = os.path.basename(model_path)
    client_id = request.cookies.get("client_id")

    if not client_id or client_id not in client_sessions:
        client_id = increment_user_count()

    response = make_response(
        render_template(
            "index.html", model_files=model_files, selected_model=selected_model
        )
    )

    # Set client_id cookie for session tracking
    response.set_cookie("client_id", client_id, max_age=86400)  # 24 hours
    return response


@app.route("/process_client_frame", methods=["POST"])
def process_client_frame():
    """Process a frame sent from the client's browser camera"""
    try:
        data = request.json
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data provided"}), 400

        client_id = request.cookies.get("client_id")

        # Process frame in thread pool for parallel processing
        future = process_pool.submit(process_frame_data, data["frame"], client_id)
        result = future.result(timeout=5.0)  # 5 second timeout

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)

    except concurrent.futures.TimeoutError:
        print_error("Frame processing timed out")
        return jsonify({"error": "Processing timed out. Please try again."}), 503

    except Exception as e:
        print_error(f"Error in process_client_frame: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/select_model/<model_name>", methods=["GET"])
def select_model(model_name):
    global model, model_path
    try:
        new_model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(new_model_path):
            return jsonify({"error": f"Model file not found: {model_name}"}), 404

        with model_lock:  # Use lock when modifying model
            model_path = new_model_path
            # Clear cache to ensure model is reloaded
            load_model.cache_clear()
            model = load_model(model_path)
        return jsonify({"status": "Model selected", "model_name": model_name})
    except Exception as e:
        print_error(f"Error selecting model: {e}")
        return jsonify({"error": f"Error selecting model: {e}"}), 500


@app.route("/reload_model", methods=["POST"])
def reload_model():
    global model
    try:
        with model_lock:  # Use lock when modifying model
            load_model.cache_clear()
            model = load_model(model_path)
        return jsonify({"status": "Model reloaded successfully"})
    except Exception as e:
        print_error(f"Error reloading model: {e}")
        return jsonify({"error": f"Error reloading model: {e}"}), 500


@app.route("/quiz")
def quiz():
    client_id = request.cookies.get("client_id")

    if not client_id or client_id not in client_sessions:
        client_id = increment_user_count()

    response = make_response(
        render_template(
            "quiz.html",
            labels_dict=get_labels_dict(),
            quiz_duration=QUIZ_DURATION,
            quiz_num_guesses=QUIZ_NUM_GUESSES,
            quiz_reload_interval=QUIZ_RELOAD_INTERVAL,
            debug_mode=DEBUG_MODE,
        )
    )

    # Set client_id cookie
    response.set_cookie("client_id", client_id, max_age=86400)
    return response


@app.route("/quiz-demo")
def quiz_demo():
    """Demo quiz with limited letter selection from environment variable"""
    client_id = request.cookies.get("client_id")

    if not client_id or client_id not in client_sessions:
        client_id = increment_user_count()

    response = make_response(
        render_template(
            "quiz_demo.html",
            allowed_letters=DEMO_QUIZ_LETTERS,
            quiz_duration=QUIZ_DURATION,
            debug_mode=DEBUG_MODE,
        )
    )

    # Set client_id cookie
    response.set_cookie("client_id", client_id, max_age=86400)
    return response


@app.route("/get_answer_image/<letter>", methods=["GET"])
def get_answer_image(letter):
    try:
        data_dir = get_directory_paths()["data"]
        class_dir = os.path.join(data_dir, letter)
        if not os.path.exists(class_dir):
            return (
                jsonify({"error": f"Class directory not found for letter: {letter}"}),
                404,
            )

        images = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
        if not images:
            return jsonify({"error": f"No images found for class: {letter}"}), 404

        image_url = f"/data/{letter}/{images[0]}"
        return jsonify({"image_url": image_url})
    except Exception as e:
        print_error(f"Error fetching answer image: {e}")
        return jsonify({"error": f"Error fetching answer image: {e}"}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(get_directory_paths()["data"], filename)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Legacy endpoint for compatibility"""
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return (
                jsonify(
                    {"error": "Invalid request data. 'frame' parameter is required."}
                ),
                400,
            )

        client_id = request.cookies.get("client_id")
        result = process_frame_data(data["frame"], client_id)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify({"prediction": result["prediction"]})

    except Exception as e:
        print_error(f"Error processing frame: {e}")
        return jsonify({"error": f"Error processing frame: {e}"}), 500


@app.route("/status", methods=["GET"])
def status():
    """Endpoint to check server status and active users"""
    if request.args.get("admin") == ADMIN_KEY:
        # Detailed status for admin
        status_data = {
            "status": "running",
            "active_users": active_users,
            "model": os.path.basename(model_path),
            "client_sessions": {},  # Will be populated below
            "worker_threads": MAX_WORKERS,
            "uptime_seconds": time.time() - startup_time,
        }

        # Include client data for admin
        for client_id, data in client_sessions.items():
            status_data["client_sessions"][client_id] = {
                "last_active_seconds_ago": time.time() - data["last_active"],
                "frames_processed": data["frames_processed"],
            }

        return jsonify(status_data)

    # Basic status for non-admin
    return jsonify({"status": "running"})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    if request.args.get("admin") == ADMIN_KEY:
        handle_signal(signal.SIGINT, None)
        return "Server shutting down..."
    return jsonify({"error": "Unauthorized"}), 403


@app.route("/model_info/<model_name>")
def model_info(model_name):
    """Get information about a specific model from its associated text file"""
    try:
        # Get model name without extension
        model_basename = os.path.splitext(model_name)[0]
        model_info_path = os.path.join(MODELS_DIR, f"{model_basename}.txt")

        # Check if info file exists
        if not os.path.exists(model_info_path):
            return jsonify({"error": "Model info not found"}), 404

        with open(model_info_path, "r") as f:
            info_content = f.read()

        # Try to parse the content as a classification report table
        result = parse_classification_report(info_content)

        # If parsing failed, return the raw content
        if not result:
            result = {"raw_info": info_content}

        return jsonify(result)

    except Exception as e:
        print_error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500


def parse_classification_report(report_text):
    """Parse a classification report text into structured data"""
    result: dict[str, Any] = {"class_report": {}, "metrics": {}}

    # Try to extract accuracy
    accuracy_match = re.search(r"Accuracy\s+(\d+\.\d+)", report_text)
    if accuracy_match:
        result["accuracy"] = float(accuracy_match.group(1))

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


def shutdown_server():
    """Properly shutdown the server and release resources"""
    global shutdown_flag
    if not shutdown_flag:
        shutdown_flag = True
        print_info("Shutting down server...")

        # Clean up thread pool
        print_info("Shutting down process pool...")
        process_pool.shutdown(wait=False)

        # Shutdown Flask
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            print_error("Not running with the Werkzeug Server")
            return
        func()
        print_info("Server shutdown complete.")


def handle_signal(sig, frame):
    """Handle system signals gracefully"""
    signal_name = "UNKNOWN"
    if sig == signal.SIGINT:
        signal_name = "SIGINT"
    elif sig == signal.SIGTERM:
        signal_name = "SIGTERM"

    print_info(f"Received {signal_name} signal. Initiating graceful shutdown...")
    shutdown_server()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Track startup time
startup_time = time.time()

if __name__ == "__main__":
    try:
        print_info("All components initialized. Starting Flask server...")
        app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE, threaded=True)
    except Exception as e:
        print_error(f"Error running the app: {e}")
        shutdown_server()
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("Keyboard interrupt received. Shutting down...")
        shutdown_server()
        sys.exit(0)
