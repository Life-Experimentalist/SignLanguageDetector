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

import time
import cv2
import mediapipe as mp
from tqdm import tqdm  # for progress bar
from utils import (
    save_data,
    mediapipe_hands,
    get_directory_paths,
    print_info,
    print_warning,
    print_error,
    IMAGES_PER_CLASS,  # configurable target
)

# Each class is expected to have 500 valid images.
DATASET_TARGET = IMAGES_PER_CLASS


def create_dataset():
    directories = get_directory_paths()
    LOG_FILE = os.path.join(directories["data"], "missing_landmarks.log")
    DATA_DIR = directories["data"]

    mp_hands = mp.solutions.hands  # type: ignore
    hands = mediapipe_hands()
    data = []
    labels = []

    print_info(f"Starting data processing from directory: {DATA_DIR}")
    start_time = time.time()

    def process_image(img_full_path, dir_name):
        img = cv2.imread(img_full_path)
        if img is None:
            print_error(f"Error: Could not read image {img_full_path}")
            return False
        # Resize image to lower resolution
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            # Log the warning silently
            with open(LOG_FILE, "a", encoding="utf-8") as logf:
                logf.write(f"No hand landmarks found: {dir_name} {img_full_path}\n")
            return False
        # Limit to at most 2 hands
        if len(results.multi_hand_landmarks) > 2:
            results.multi_hand_landmarks = results.multi_hand_landmarks[:2]
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)
        # If only one hand is detected, pad the values for consistent feature length
        if len(results.multi_hand_landmarks) == 1:
            data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
        data.append(data_aux)
        labels.append(dir_name)
        return True

    try:
        for dir_ in os.listdir(DATA_DIR):
            dir_path = os.path.join(DATA_DIR, dir_)
            dir_start_time = time.time()
            print_info(f"Processing directory: {dir_}")
            if os.path.isdir(dir_path):
                images = [
                    f
                    for f in os.listdir(dir_path)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
                # Set target to the lesser of available images and our expected dataset size
                target = min(DATASET_TARGET, len(images))
                valid_count = 0
                # Configure tqdm progress bar with target total = 500 (or available images)
                with tqdm(total=target, desc=f"Processing {dir_}", leave=False) as pbar:
                    for img_file in images:
                        # Stop if we've reached our valid target
                        if pbar.n >= pbar.total:
                            break
                        img_path_full = os.path.join(dir_path, img_file)
                        if process_image(img_path_full, dir_):
                            valid_count += 1
                            pbar.update(1)
                        else:
                            # Remove a failed image from the target
                            pbar.total -= 1
                            pbar.refresh()
                wasted = target - valid_count
                print_info(
                    f"For class {dir_}: {wasted} images wasted out of target {target}."
                )
            else:
                print_warning(f"Skipping non-directory: {dir_path}")
            elapsed = time.time() - dir_start_time
            print_info(f"Finished processing directory {dir_} in {elapsed:.4f} seconds")
    except KeyboardInterrupt:
        print_warning("\nScript interrupted. Saving partial data...")

    total_time = time.time() - start_time
    print_info(f"Finished processing all data in {total_time:.4f} seconds")
    save_data(data, labels, os.path.join(directories["data"], "data.pickle"))


if __name__ == "__main__":
    create_dataset()
