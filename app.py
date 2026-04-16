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
import signal
import sys

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# Import directly from utils package - all the needed functions and variables
from utils import (
    DEBUG_MODE,
    DEMO_QUIZ_LETTERS,
    DISABLE_ANONYMOUS_TELEMETRY,
    MODELS_DIR,
    PORT,
    QUIZ_DURATION,
    QUIZ_NUM_GUESSES,
    QUIZ_RELOAD_INTERVAL,
    TELEMETRY_COUNTER_BASE_URL,
    TELEMETRY_PROJECT_NAME,
    # Import shared app utilities
    AppInitializer,
    get_directory_paths,
    get_labels_dict,
    get_logger,
    handle_signal,
    parse_classification_report,
    print_error,
    print_info,
    print_warning,
    process_frame_data,
    shutdown_server,
)

# Setup logger
logger = get_logger(__name__)

app = Flask(__name__)

# Global flag to track shutdown status
shutdown_flag = False

# Initialize application components
app_initializer = AppInitializer()
model, hands, landmark_style, connection_style = app_initializer.initialize()
labels_dict = get_labels_dict()
two_hand_classes = app_initializer.two_hand_classes
model_path = app_initializer.model_path

print_info("Mediapipe initialized successfully")


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


@app.route("/")
def index():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    selected_model = os.path.basename(str(model_path))
    return render_template(
        "index.html", model_files=model_files, selected_model=selected_model
    )


@app.route("/process_client_frame", methods=["POST"])
def process_client_frame():
    """Process a frame sent from the client's browser camera"""
    try:
        data = request.json
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data provided"}), 400

        # Get landmark options if provided
        options = data.get(
            "options", {"showLandmarks": True, "landmarkStyle": "default"}
        )

        # Process the frame with options using shared function
        result = process_frame_data(
            data["frame"],
            options,
            model,
            hands,
            labels_dict,
            two_hand_classes,
            landmark_style,
            connection_style,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)

    except Exception as e:
        print_error(f"Error in process_client_frame: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict sign output from a single image without using the webpage."""
    try:
        options = {"showLandmarks": False, "landmarkStyle": "none"}
        include_visuals = False

        if request.files and "image" in request.files:
            image_file = request.files.get("image")
            if image_file is None or image_file.filename == "":
                return jsonify(
                    {"error": "Image file is required in field 'image'."}
                ), 400

            image_bytes = image_file.read()
            if not image_bytes:
                return jsonify({"error": "Uploaded image is empty."}), 400

            show_landmarks = (
                request.form.get("show_landmarks", "false").strip().lower() == "true"
            )
            include_visuals = (
                request.form.get("include_visuals", "false").strip().lower() == "true"
            )
        else:
            payload = request.get_json(silent=True) or {}
            image_base64 = payload.get("image_base64") or payload.get("frame")
            if not image_base64:
                return (
                    jsonify(
                        {
                            "error": "Provide an image via multipart field 'image' or JSON key 'image_base64'."
                        }
                    ),
                    400,
                )

            show_landmarks = bool(payload.get("show_landmarks", False))
            include_visuals = bool(payload.get("include_visuals", False))

            if "," in image_base64:
                data_uri = image_base64
            else:
                data_uri = f"data:image/jpeg;base64,{image_base64}"

            image_bytes = data_uri.split(",", 1)[1].encode("utf-8")
            image_bytes = base64.b64decode(image_bytes)

        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        frame_data = f"data:image/jpeg;base64,{encoded_image}"

        options["showLandmarks"] = show_landmarks
        options["landmarkStyle"] = "default" if show_landmarks else "none"

        result = process_frame_data(
            frame_data,
            options,
            model,
            hands,
            labels_dict,
            two_hand_classes,
            landmark_style,
            connection_style,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        response = {
            "prediction": result.get("prediction", ""),
            "brightness": result.get("brightness"),
            "contrast": result.get("contrast"),
            "low_brightness": result.get("low_brightness"),
            "model": os.path.basename(str(model_path)),
        }

        if include_visuals:
            response["processed_frame"] = result.get("processed_frame")
            if "original_frame" in result:
                response["original_frame"] = result["original_frame"]

        return jsonify(response)
    except Exception as e:
        print_error(f"Error in api_predict: {e}")
        return jsonify({"error": str(e)}), 500


# The select_model route is important and should be kept
# It allows changing models without restarting the server
@app.route("/select_model/<model_name>", methods=["GET"])
def select_model(model_name):
    """
    Change the active model used for predictions.

    This endpoint allows users to switch between different trained models
    without needing to restart the application. It's used by the model dropdown
    in the UI.
    """
    global model, model_path
    try:
        new_model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(new_model_path):
            return jsonify({"error": f"Model file not found: {model_name}"}), 404

        model_path = new_model_path
        # Clear cache to ensure model is reloaded
        app_initializer.load_model.cache_clear()
        model = app_initializer.load_model(model_path)
        return jsonify({"status": "Model selected", "model_name": model_name})
    except Exception as e:
        print_error(f"Error selecting model: {e}")
        return jsonify({"error": f"Error selecting model: {e}"}), 500


@app.route("/reload_model", methods=["POST"])
def reload_model():
    global model
    try:
        app_initializer.load_model.cache_clear()
        model = app_initializer.load_model(model_path)
        return jsonify({"status": "Model reloaded successfully"})
    except Exception as e:
        print_error(f"Error reloading model: {e}")
        return jsonify({"error": f"Error reloading model: {e}"}), 500


@app.route("/quiz")
def quiz():
    return render_template(
        "quiz.html",
        labels_dict=get_labels_dict(),
        quiz_duration=QUIZ_DURATION,
        quiz_num_guesses=QUIZ_NUM_GUESSES,
        quiz_reload_interval=QUIZ_RELOAD_INTERVAL,
        debug_mode=DEBUG_MODE,
    )


# Add the new quiz-demo route
@app.route("/quiz-demo")
def quiz_demo():
    """Demo quiz with limited letter selection from environment variable"""
    return render_template(
        "quiz_demo.html",  # Use the quiz_demo template
        allowed_letters=DEMO_QUIZ_LETTERS,
        quiz_duration=QUIZ_DURATION,
        debug_mode=DEBUG_MODE,
    )


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
    return send_from_directory(get_directory_paths()["data"], filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(get_directory_paths()["data"], filename)


# The process_frame route is deprecated and can be removed
# It's only kept for backward compatibility with older frontends
# Consider adding a deprecation warning or removing it in the future
@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    DEPRECATED: Legacy endpoint for compatibility with older frontends.

    New implementations should use /process_client_frame instead, which provides
    more features and better error handling.
    """
    # (Keep the implementation but add deprecation warning)
    print_warning(
        "Deprecated /process_frame endpoint used. Consider migrating to /process_client_frame."
    )
    try:
        data = request.get_json()
        if not data or "frame" not in data:
            return (
                jsonify(
                    {"error": "Invalid request data. 'frame' parameter is required."}
                ),
                400,
            )

        result = process_frame_data(
            data["frame"],
            None,
            model,
            hands,
            labels_dict,
            two_hand_classes,
            landmark_style,
            connection_style,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify({"prediction": result["prediction"]})

    except Exception as e:
        print_error(f"Error processing frame: {e}")
        return jsonify({"error": f"Error processing frame: {e}"}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    global shutdown_flag
    shutdown_flag = True
    handle_signal(signal.SIGINT, None, shutdown_flag, app)
    return "Server shutting down..."


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

        # Try to parse the content as a classification report table using shared function
        result = parse_classification_report(info_content)

        # If parsing failed, return the raw content
        if not result:
            result = {"raw_info": info_content}

        return jsonify(result)

    except Exception as e:
        print_error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/simplified_model_info/<model_name>")
def simplified_model_info(model_name):
    """
    Get simplified information about a model for the sidebar display.

    Returns key model performance metrics in a structured format:
    - Overall accuracy
    - Top 3 highest performing classes (by F1 score)
    - Worst performing class (by F1 score)
    - Macro and weighted averages (if available)
    """
    try:
        # Get model name without extension
        model_basename = os.path.splitext(model_name)[0]
        model_info_path = os.path.join(MODELS_DIR, f"{model_basename}.txt")

        # Check if info file exists
        if not os.path.exists(model_info_path):
            return jsonify({"error": "Model info not found"}), 404

        with open(model_info_path, "r") as f:
            info_content = f.read()

        # Parse the content as a classification report table
        full_result = parse_classification_report(info_content)

        # Create a simplified version with just the highlights
        result = {
            "model_name": model_name,
            "highlights": {},
        }

        # Add accuracy if available
        if full_result:
            if "accuracy" in full_result:
                accuracy = full_result["accuracy"]
                if isinstance(accuracy, dict):
                    accuracy_value = accuracy.get("score", 0)
                else:
                    accuracy_value = accuracy
                result["highlights"]["accuracy"] = {
                    "value": accuracy_value,
                    "formatted": f"{accuracy_value * 100:.1f}%",
                }
            elif "metrics" in full_result and "accuracy" in full_result["metrics"]:
                accuracy = full_result["metrics"]["accuracy"]
                if isinstance(accuracy, dict):
                    accuracy_value = accuracy.get("score", 0)
                else:
                    accuracy_value = accuracy
                result["highlights"]["accuracy"] = {
                    "value": accuracy_value,
                    "formatted": f"{accuracy_value * 100:.1f}%",
                }

        # Add top 3 classes by F1 score
        if (
            full_result
            and "class_report" in full_result
            and full_result["class_report"]
        ):
            # Sort classes by F1 score
            classes = []
            for cls, metrics in full_result["class_report"].items():
                if cls not in ["macro avg", "weighted avg"] and "f1Score" in metrics:
                    classes.append((cls, metrics["f1Score"]))

            # Sort by F1 score in descending order
            classes.sort(key=lambda x: x[1], reverse=True)

            # Get top 3 and worst class
            result["highlights"]["top_classes"] = []
            for i, (cls, f1_score) in enumerate(classes[:3]):
                label = labels_dict.get(int(cls), cls) if cls.isdigit() else cls
                result["highlights"]["top_classes"].append(
                    {
                        "class": cls,
                        "label": label,
                        "f1_score": f1_score,
                        "formatted": f"{f1_score * 100:.1f}%",
                    }
                )

            # Add worst class if there are more than 3 classes
            if len(classes) > 3:
                worst_cls, worst_f1 = classes[-1]
                label = (
                    labels_dict.get(int(worst_cls), worst_cls)
                    if worst_cls.isdigit()
                    else worst_cls
                )
                result["highlights"]["worst_class"] = {
                    "class": worst_cls,
                    "label": label,
                    "f1_score": worst_f1,
                    "formatted": f"{worst_f1 * 100:.1f}%",
                }

        # Add macro and weighted averages
        if full_result and "metrics" in full_result:
            result["highlights"]["averages"] = {}
            for avg_type in ["macro avg", "weighted avg"]:
                if (
                    avg_type in full_result["metrics"]
                    and "f1" in full_result["metrics"][avg_type]
                ):
                    f1 = full_result["metrics"][avg_type]["f1"]
                    result["highlights"]["averages"][avg_type.replace(" ", "_")] = {
                        "f1_score": f1,
                        "formatted": f"{f1 * 100:.1f}%",
                    }

        return jsonify(result)

    except Exception as e:
        print_error(f"Error getting simplified model info: {e}")
        return jsonify({"error": str(e)}), 500


def get_file_creation_date(file_path):
    """Get the file creation date formatted nicely"""
    try:
        import datetime

        timestamp = os.path.getctime(file_path)
        date = datetime.datetime.fromtimestamp(timestamp)
        return date.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"


# Add route to serve the JSON model report files
@app.route("/models/<path:filename>")
def serve_models(filename):
    """Serve model report files (JSON, txt) from the models directory"""
    return send_from_directory(MODELS_DIR, filename)


# Register signal handlers with the custom handler that includes shutdown_flag
signal.signal(
    signal.SIGINT, lambda sig, frame: handle_signal(sig, frame, shutdown_flag, app)
)
signal.signal(
    signal.SIGTERM, lambda sig, frame: handle_signal(sig, frame, shutdown_flag, app)
)


if __name__ == "__main__":
    # Reloader is disabled below, so avoid an expensive filesystem walk on startup.
    extra_files = []

    import signal

    # Register signal handlers
    signal.signal(
        signal.SIGINT,
        lambda sig, frame: (
            print_info("Ctrl+C pressed, shutting down..."),
            shutdown_server(shutdown_flag, app),
            sys.exit(0),
        ),
    )
    signal.signal(
        signal.SIGTERM,
        lambda sig, frame: (
            print_info("SIGTERM received, shutting down..."),
            shutdown_server(shutdown_flag, app),
            sys.exit(0),
        ),
    )

    try:
        print_info("All components initialized. Starting Flask server...")
        # Disable the reloader so the main process can capture Ctrl+C
        app.run(
            host="0.0.0.0",
            port=PORT,
            debug=DEBUG_MODE,
            use_reloader=False,  # Disable reloader for proper signal handling
        )
    except Exception as e:
        print_error(f"Error running the app: {e}")
        shutdown_server(shutdown_flag, app)
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("Keyboard interrupt received. Shutting down...")
        shutdown_server(shutdown_flag, app)
        sys.exit(0)
