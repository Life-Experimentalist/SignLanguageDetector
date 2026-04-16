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

# Importing Libraries
import sys

import cv2 as cv
import mediapipe as mp

# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict

from utils.utils import print_info

# List of blocked packages
BLOCKED_PACKAGES = ["opencv-python-headless"]

# Check if any blocked package is installed
for package in BLOCKED_PACKAGES:
    try:
        __import__(package)
        print(f"ERROR: {package} is installed. Please uninstall it before proceeding.")
        sys.exit(1)
    except ImportError:
        pass

print("All checks passed.")


# Function to find the first available camera index
def get_available_camera_index():
    index = 0
    while True:
        cap = cv.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return index - 1 if index > 0 else None
        cap.release()
        index += 1


# Initializing the Model
mpHands = mp.solutions.hands  # type: ignore
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,  # Lowered confidence threshold
    min_tracking_confidence=0.5,  # Lowered tracking threshold
    max_num_hands=2,
)

# Find the first available camera
camera_index = get_available_camera_index()

if camera_index is None:
    print("Error: No camera found.")
    sys.exit(1)

print_info(f"Using camera at index {camera_index}.")

# Start capturing video from the detected camera
cap = cv.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Could not open video capture device at index {camera_index}.")
    sys.exit(1)

while True:
    # Read video frame by frame
    success, img = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Flip the image(frame)
    img = cv.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:
        handedness_labels = []
        for i in results.multi_handedness:
            # Convert protobuf message to dictionary
            label_dict = MessageToDict(i)
            # Return whether it is Right or Left Hand
            label = label_dict["classification"][0]["label"]
            handedness_labels.append(label)

        if (
            len(handedness_labels) == 2
            and "Left" in handedness_labels
            and "Right" in handedness_labels
        ):
            # Display 'Both Hands' on the image
            cv.putText(
                img,
                "Both Hands",
                (250, 50),
                cv.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        elif len(handedness_labels) == 1:
            label = handedness_labels[0]
            if label == "Left":
                # Display 'Left Hand' on left side of window
                cv.putText(
                    img,
                    label + " Hand",
                    (20, 50),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            if label == "Right":
                # Display 'Right Hand' on right side of window
                cv.putText(
                    img,
                    label + " Hand",
                    (460, 50),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    # Display Video and when 'q'
    # is entered, destroy the window
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and destroy all active windows
cap.release()
cv.destroyAllWindows()
