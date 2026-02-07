"""
Evaluate model2result.keras on the test set
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Configuration
MODEL_PATH = 'model2result.keras'
TEST_DIR = './chest_xray/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\n" + "="*60)
print("MODEL EVALUATION ON TEST SET")
print("="*60)

# Load model
print(f"\n[1] Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print(f"    ✓ Model loaded successfully")

# Load test data
print(f"\n[2] Loading test data from: {TEST_DIR}")
test_ds = keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='binary'
)

# Get class names
class_names = test_ds.class_names
print(f"    Classes: {class_names}")

# Count images
total_images = sum([len(files) for r, d, files in os.walk(TEST_DIR) if files])
print(f"    Total test images: {total_images}")

# Preprocess (same as training)
from tensorflow.keras.applications.densenet import preprocess_input
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

# Evaluate
print(f"\n[3] Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print("="*60)

# Get predictions for detailed analysis
print(f"\n[4] Analyzing predictions...")
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate metrics per class
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"              NORMAL  PNEUMONIA")
print(f"Actual NORMAL    {cm[0][0]:4d}      {cm[0][1]:4d}")
print(f"     PNEUMONIA   {cm[1][0]:4d}      {cm[1][1]:4d}")

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) * 100  # True Positive Rate (Recall for PNEUMONIA)
specificity = tn / (tn + fp) * 100  # True Negative Rate

print(f"\nClinical Metrics:")
print(f"  Sensitivity (Pneumonia Detection): {sensitivity:.2f}%")
print(f"  Specificity (Normal Detection): {specificity:.2f}%")

print("\n" + "="*60)
print(f"✓ OVERALL TEST ACCURACY: {test_accuracy*100:.2f}%")
print("="*60 + "\n")
