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

import itertools
import os
import pickle
import time

import cv2
import mediapipe as mp

# Make sure the log file is in the same directory as this script
SCRIPT_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(SCRIPT_DIR, "missing_landmarks.log")

mp_hands = mp.solutions.hands  # type: ignore
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3
)

DATA_DIR = "./data"
data = []
labels = []

print(f"Starting data processing from directory: {DATA_DIR}")

start_time = time.time()

spinner = itertools.cycle(["-", "/", "|", "\\"])


def process_image(img_full_path, dir_name):
    img = cv2.imread(img_full_path)
    if img is None:
        print(f"  Error: Could not read image {img_full_path}")
        return

    # Resize image to lower resolution (speeds up processing)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        print(
            f"  Warning: No hand landmarks found in {os.path.basename(img_full_path)}"
        )
        with open(LOG_FILE, "a", encoding="utf-8") as logf:
            logf.write(f"No hand landmarks found: {dir_name} {img_full_path}\n")
        return

    # Limit to at most 2 hands
    if len(results.multi_hand_landmarks) > 2:
        results.multi_hand_landmarks = results.multi_hand_landmarks[:2]

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_ = []
        y_ = []
        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        min_x, min_y = min(x_), min(y_)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

    # If only one hand is detected, pad the data_aux to ensure consistent feature length
    if len(results.multi_hand_landmarks) == 1:
        data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))

    data.append(data_aux)
    labels.append(dir_name)


try:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        dir_start_time = time.time()
        print(f"Processing directory: {dir_}")

        if os.path.isdir(dir_path):
            images = [
                f
                for f in os.listdir(dir_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            for img_file in images:
                img_path_full = os.path.join(dir_path, img_file)
                process_image(img_path_full, dir_)
                print(next(spinner), end="\r")
        else:
            print(f"Skipping non-directory: {dir_path}")

        print(
            f"Finished processing directory {dir_} in {time.time() - dir_start_time:.4f} seconds"
        )

except KeyboardInterrupt:
    print("\nScript interrupted. Saving partial data...")

end_time = time.time()
print(f"Finished processing all data in {end_time - start_time:.4f} seconds")

with open(os.path.join(SCRIPT_DIR, "data.pickle"), "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Data saved to data.pickle")
