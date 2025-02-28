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

import base64
import os
import pickle
import signal
import sys
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from utils import (
    MODELS_DIR,
    NUM_CLASSES,
    calculate_brightness,
    calculate_contrast,
    convert_numpy_types,
    draw_landmarks,
    get_directory_paths,
    get_labels_dict,
    get_landmark_style,
    get_two_hand_classes,
    mediapipe_hands,
    print_error,
    print_info,
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load model
model_path = os.path.join(MODELS_DIR, "model.p")
if not os.path.exists(model_path):
    print_error(f"Model file not found: {model_path}")
    sys.exit(1)


@lru_cache(maxsize=None)
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict["data"]["model"]
    except Exception as e:
        print_error(f"Error loading model: {e}")
        sys.exit(1)


model = load_model(model_path)
labels_dict = get_labels_dict()
two_hand_classes = get_two_hand_classes()
hands = mediapipe_hands()
landmark_style, connection_style = get_landmark_style()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print_error("Error: Could not open video capture device.")
    sys.exit(1)


def generate_frames():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print_error("Error: Could not read frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) > 2:
                    results.multi_hand_landmarks = results.multi_hand_landmarks[:2]
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,  # type: ignore
                        landmark_style,
                        connection_style,
                    )
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    except Exception as e:
        print_error(f"Error during frame generation: {e}")


@app.route("/")
def index():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".p")]
    selected_model = os.path.basename(model_path)
    return render_template(
        "index.html", model_files=model_files, selected_model=selected_model
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/predictions", methods=["GET"])
def predictions():
    try:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Could not read frame."}), 500

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        low_brightness = brightness < float(os.getenv("BRIGHTNESS_THRESHOLD", "85"))

        data_aux = []
        predicted_character = ""
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 2:
                results.multi_hand_landmarks = results.multi_hand_landmarks[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
            if data_aux:
                data_aux = np.asarray(data_aux)
                try:
                    prediction = model.predict([data_aux])
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    if (
                        predicted_character in two_hand_classes
                        and len(results.multi_hand_landmarks) < 2
                    ):
                        predicted_character = ""
                except Exception as e:
                    print_error(f"Error during prediction: {e}")
                    return jsonify({"error": f"Error during prediction: {e}"}), 500
        return jsonify(
            {
                "prediction": predicted_character,
                "brightness": convert_numpy_types(brightness),
                "contrast": convert_numpy_types(contrast),
                "low_brightness": convert_numpy_types(low_brightness),
            }
        )
    except Exception as e:
        print_error(f"Error in /predictions endpoint: {e}")
        return jsonify({"error": f"Error in /predictions endpoint: {e}"}), 500


@app.route("/select_model/<model_name>", methods=["GET"])
def select_model(model_name):
    global model, model_path
    model_path = os.path.join(MODELS_DIR, model_name)
    model = load_model(model_path)
    return jsonify({"status": "Model selected", "model_name": model_name})


@app.route("/reload_model", methods=["POST"])
def reload_model():
    global model
    load_model.cache_clear()
    model = load_model(model_path)
    return jsonify({"status": "Model reloaded"})


@app.route("/quiz")
def quiz():
    quiz_duration = int(os.getenv("QUIZ_DURATION", "2"))
    quiz_num_guesses = int(os.getenv("QUIZ_NUM_GUESSES", "5"))
    quiz_reload_interval = int(os.getenv("QUIZ_RELOAD_INTERVAL", "0"))
    return render_template(
        "quiz.html",
        labels_dict=labels_dict,
        quiz_duration=quiz_duration,
        quiz_num_guesses=quiz_num_guesses,
        quiz_reload_interval=quiz_reload_interval,
    )


@app.route("/quiz_video_feed")
def quiz_video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/get_answer_image/<letter>", methods=["GET"])
def get_answer_image(letter):
    try:
        data_dir = get_directory_paths()["data"]
        class_dir = os.path.join(data_dir, letter)
        if not os.path.exists(class_dir):
            return jsonify({"error": "Class directory not found"}), 404

        images = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
        if not images:
            return jsonify({"error": "No images found for this class"}), 404

        image_path = os.path.join(class_dir, images[0])
        image_url = f"/data/{letter}/{images[0]}"
        return jsonify({"image_url": image_url})
    except Exception as e:
        print_error(f"Error fetching answer image: {e}")
        return jsonify({"error": f"Error fetching answer image: {e}"}), 500


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(get_directory_paths()["data"], filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(get_directory_paths()["data"], filename)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.get_json()
        frame_data = data["frame"].split(",")[1]
        frame = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        predicted_character = ""
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 2:
                results.multi_hand_landmarks = results.multi_hand_landmarks[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
            if data_aux:
                data_aux = np.asarray(data_aux)
                try:
                    prediction = model.predict([data_aux])
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    if (
                        predicted_character in two_hand_classes
                        and len(results.multi_hand_landmarks) < 2
                    ):
                        predicted_character = ""
                except Exception as e:
                    print_error(f"Error during prediction: {e}")
                    predicted_character = "Error"
        return jsonify({"prediction": predicted_character})
    except Exception as e:
        print_error(f"Error processing frame: {e}")
        return jsonify({"error": f"Error processing frame: {e}"}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    shutdown_server()
    return "Server shutting down..."


def shutdown_server():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


def handle_signal(signal, frame):
    print_info("Received signal to terminate. Shutting down...")
    shutdown_server()
    cap.release()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
    except Exception as e:
        print_error(f"Error running the app: {e}")
    except KeyboardInterrupt:
        print_info("Keyboard interrupt received. Shutting down...")
        shutdown_server()
        cap.release()
        sys.exit(0)
