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

import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)  # Try different camera index if 2 doesn't work

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

hands = mp_hands.Hands(
    static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2
)  # Changed max_num_hands to 2

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
}  # This needs to match the labels in your training data

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
}  # Add classes that require two hands


def draw_prediction(frame, predicted_character, x_, y_, W, H):
    """Draw the prediction results on the frame."""
    x1 = int(min(x_) * W) - 10
    y1 = int(min(y_) * H) - 10
    x2 = int(max(x_) * W) - 10
    y2 = int(max(y_) * H) - 10

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(
        frame,
        predicted_character,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )


while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Limit to at most 2 hands
        if len(results.multi_hand_landmarks) > 2:
            results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            min_x, min_y = min(x_), min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # If only one hand is detected, pad the data_aux to ensure consistent feature length
        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))

        # Make prediction
        if data_aux:  # Check if landmarks were detected
            data_aux = np.asarray(data_aux)
            prediction = model.predict([data_aux])

            predicted_character = labels_dict[int(prediction[0])]

            if (
                predicted_character in two_hand_classes
                and len(results.multi_hand_landmarks) < 2
            ):
                print("Error")
            else:
                print(predicted_character)
                draw_prediction(frame, predicted_character, x_, y_, W, H)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
