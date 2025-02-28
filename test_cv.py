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
import cv2 as cv
import mediapipe as mp

# Used to convert protobuf message to a dictionary.
from google.protobuf.json_format import MessageToDict

# Initializing the Model
mpHands = mp.solutions.hands  # type: ignore
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2,
)

# Start capturing video from webcam
cap = cv.VideoCapture(0)

while True:
    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        # Two Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
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

        # If any hand present
        else:
            for i in results.multi_handedness:
                # Convert protobuf message to dictionary
                label_dict = MessageToDict(i)

                # Return whether it is Right or Left Hand
                label = label_dict["classification"][0]["label"]

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
