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

import itertools  # Import the itertools module for the spinner
import os
import pickle
import time  # Import the time module

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"

data = []
labels = []

print(f"Starting data processing from directory: {DATA_DIR}")  # Log start

start_time = time.time()  # Start timer

spinner = itertools.cycle(["-", "/", "|", "\\"])  # Define the spinner

for dir_ in os.listdir(DATA_DIR):
    dir_start_time = time.time()  # Start timer for directory
    print(f"Processing directory: {dir_}")
    dir_path = os.path.join(DATA_DIR, dir_)

    if os.path.isdir(dir_path):  # Check if it's a directory
        for img_path in os.listdir(dir_path):
            img_start_time = time.time()  # Start timer for image
            data_aux = []

            x_ = []
            y_ = []

            img_path_full = os.path.join(dir_path, img_path)
            img = cv2.imread(img_path_full)
            if img is None:
                print(f"  Error: Could not read image {img_path_full}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            print(
                f"  Processed image {img_path} in {time.time() - img_start_time:.4f} seconds",
                end="\r",
            )  # Log image processing time with spinner
            print(next(spinner), end="\r")  # Print the spinner
        else:
            print(f"  Warning: No hand landmarks found in {img_path}")
    else:
        print(f"Skipping non-directory: {dir_path}")  # Skip non-directories

    print(
        f"Finished processing directory {dir_} in {time.time() - dir_start_time:.4f} seconds"
    )  # Log directory processing time

end_time = time.time()  # End timer
print(
    f"Finished processing all data in {end_time - start_time:.4f} seconds"
)  # Log total processing time

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()

print("Data saved to data.pickle")  # Log data saving
