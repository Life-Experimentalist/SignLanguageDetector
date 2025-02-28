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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load and validate data
print("Loading data...")
try:
    with open("./data.pickle", "rb") as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle not found. Run create_dataset.py first.")
    exit(1)

# Convert and validate data shape
raw_data = data_dict["data"]
raw_labels = data_dict["labels"]

# Check if we have any data
if not raw_data:
    print("Error: No data found in data.pickle")
    exit(1)

# Get the expected feature length from the first sample
expected_length = len(raw_data[0])
print(f"Expected features per sample: {expected_length}")

# Filter out samples with incorrect feature count
valid_data = []
valid_labels = []
for sample, label in zip(raw_data, raw_labels):
    if len(sample) == expected_length:
        valid_data.append(sample)
        valid_labels.append(label)

# Convert to numpy arrays
data = np.array(valid_data)
labels = np.array(valid_labels)

print(f"Dataset shape: {data.shape}")
print(f"Number of classes: {len(np.unique(labels))}")
print(f"Samples per class: {[list(labels).count(i) for i in np.unique(labels)]}")

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
print("Evaluating model...")
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"\nAccuracy: {score * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_predict))

# Save the model
print("\nSaving model...")
with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved as model.p")
