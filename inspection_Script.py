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

model_path = "V:/Code/ProjectCode/SignLanguageDetector/models/model.p"
with open(model_path, "rb") as f:
    model_data = pickle.load(f)
    print("Type of model_data:", type(model_data))
    print("Contents:", model_data if isinstance(model_data, dict) else "Not a dict")
if isinstance(model_data, dict):
    print("Keys:", model_data.keys())

# import argparse
# import ast
# import os
# import pickle
# import sys

# import cv2
# import mediapipe as mp
# import numpy as np
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv(".env")

# # Constants from .env
# BRIGHTNESS_THRESHOLD = float(os.getenv("BRIGHTNESS_THRESHOLD", "85"))
# MODELS_DIR = os.getenv("MODELS_DIR", "./models")

# # Get labels dictionary and two-hand classes from .env
# raw_labels = os.getenv("LABELS_DICT", "{}")
# labels_dict = {int(k): v for k, v in ast.literal_eval(raw_labels).items()}
# raw_two_hand = os.getenv("TWO_HAND_CLASSES", "[]")
# two_hand_classes = set(ast.literal_eval(raw_two_hand))


# def calculate_brightness(frame):
#     """Calculate average brightness using the V channel of HSV."""
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     return np.mean(hsv[:, :, 2])


# def calculate_contrast(frame):
#     """Calculate contrast using the standard deviation of the L channel in LAB."""
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#     l_channel, _, _ = cv2.split(lab)
#     return l_channel.std()


# def draw_prediction(frame, predicted_character, x_list, y_list):
#     """Draw a rectangle around the detected hand(s) and annotate the prediction."""
#     H, W = frame.shape[:2]
#     if not x_list or not y_list:
#         return frame
#     x1 = int(min(x_list) * W) - 10
#     y1 = int(min(y_list) * H) - 10
#     x2 = int(max(x_list) * W) - 10
#     y2 = int(max(y_list) * H) - 10
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#     cv2.putText(
#         frame,
#         predicted_character if predicted_character else "No Prediction",
#         (x1, y1 - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.3,
#         (0, 0, 0),
#         3,
#         cv2.LINE_AA,
#     )
#     return frame


# def load_model(model_path):
#     """Load the model dictionary from a pickle file and return the model."""
#     with open(model_path, "rb") as f:
#         model_dict = pickle.load(f)
#     return model_dict["data"]["model"]


# def main(args):
#     # Load the image
#     image = cv2.imread(args.image)
#     if image is None:
#         print("Error: Could not load image at path:", args.image)
#         sys.exit(1)

#     # Load model
#     if args.model:
#         model_file = args.model
#     else:
#         # Use the first .p file found in the models directory
#         model_dir = (
#             MODELS_DIR
#             if os.path.isabs(MODELS_DIR)
#             else os.path.join(os.getcwd(), MODELS_DIR)
#         )
#         model_files = [f for f in os.listdir(model_dir) if f.endswith(".p")]
#         if not model_files:
#             print("Error: No model file found in directory:", model_dir)
#             sys.exit(1)
#         model_file = os.path.join(model_dir, model_files[0])

#     print("Loading model from:", model_file)
#     model = load_model(model_file)
#     if not hasattr(model, "predict"):
#         print("Error: Loaded model does not have a predict method.")
#         sys.exit(1)

#     # Initialize MediaPipe Hands (using static image mode)
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     hands = mp_hands.Hands(
#         static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2
#     )

#     # Process the image: convert to RGB and detect hand landmarks
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)

#     brightness = calculate_brightness(image)
#     contrast = calculate_contrast(image)
#     low_brightness = brightness < BRIGHTNESS_THRESHOLD

#     data_aux = []
#     x_list = []
#     y_list = []

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw landmarks for visualization
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             x_local = [lm.x for lm in hand_landmarks.landmark]
#             y_local = [lm.y for lm in hand_landmarks.landmark]
#             x_list.extend(x_local)
#             y_list.extend(y_local)
#             min_x, min_y = min(x_local), min(y_local)
#             for lm in hand_landmarks.landmark:
#                 data_aux.append(lm.x - min_x)
#                 data_aux.append(lm.y - min_y)
#         # If only one hand is detected, add zeros for the missing hand
#         if len(results.multi_hand_landmarks) == 1:
#             hand_landmarks = results.multi_hand_landmarks[0]
#             data_aux.extend([0] * (len(hand_landmarks.landmark) * 2))
#     else:
#         print("No hand landmarks detected in the image.")

#     predicted_character = ""
#     if data_aux:
#         data_aux = np.asarray(data_aux)
#         try:
#             # Predict using the model (note: ensure the input shape matches what the model expects)
#             prediction = model.predict([data_aux])[0]
#             pred_int = int(prediction)
#             if pred_int in labels_dict:
#                 predicted_character = labels_dict[pred_int]
#                 # Check if the predicted character requires two hands
#                 if predicted_character in two_hand_classes and (
#                     not results.multi_hand_landmarks
#                     or len(results.multi_hand_landmarks) < 2
#                 ):
#                     print("Two hands required but not detected.")
#                     predicted_character = ""
#             else:
#                 print("Prediction value", pred_int, "not found in labels dictionary.")
#         except Exception as e:
#             print("Error during prediction:", e)
#     else:
#         print("No landmark data extracted for prediction.")

#     print("Brightness:", brightness)
#     print("Contrast:", contrast)
#     print("Low brightness:", low_brightness)
#     print("Predicted character:", predicted_character)

#     # Draw the prediction result on the image
#     image_out = draw_prediction(image.copy(), predicted_character, x_list, y_list)

#     # Display the resulting image
#     cv2.imshow("Prediction", image_out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Predict sign language class from an image."
#     )
#     parser.add_argument(
#         "image", help="Path to the input image file.", default=".data/0/1.png"
#     )
#     parser.add_argument(
#         "--model",
#         help="Path to the model file (.p) if not using the default.",
#         default=".models/model.p",
#     )
#     args = parser.parse_args()
#     main(args)
