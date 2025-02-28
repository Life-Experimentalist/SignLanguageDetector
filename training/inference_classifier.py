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
import os
import sys

# Add the parent directory (project root) to sys.path so that utils can be found.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle

import cv2
import mediapipe as mp
import numpy as np

from utils import MODELS_DIR, NUM_CLASSES, get_directory_paths, print_error, print_info


def initialize_hands():
    """Initialize and return the MediaPipe Hands object."""
    mp_hands = mp.solutions.hands  # type: ignore
    return mp_hands.Hands(
        static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2
    )


def run_inference():
    directories = get_directory_paths()
    model_path = os.path.join(directories["models"], "model.p")
    model_dict = pickle.load(open(model_path, "rb"))
    model = model_dict["model"]

    print_info(f"Using a model trained on {NUM_CLASSES} classes")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_error("Error: Could not open video capture device.")
        exit()

    mp_hands = mp.solutions.hands  # type: ignore
    mp_drawing = mp.solutions.drawing_utils  # type: ignore
    mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

    hands = initialize_hands()

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

    while True:
        data_aux = []
        ret, frame = cap.read()
        if not ret:
            print_error("Error: Could not read frame.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
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
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
            if data_aux:
                data_aux = np.asarray(data_aux)
                prediction = model.predict([data_aux])
                predicted_character = labels_dict[int(prediction[0])]
                if (
                    predicted_character in two_hand_classes
                    and len(results.multi_hand_landmarks) < 2
                ):
                    print_error("Error")
                else:
                    print_info(predicted_character)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
