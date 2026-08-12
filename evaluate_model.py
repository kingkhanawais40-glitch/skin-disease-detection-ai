import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"
MODEL_PATH = "skin_disease_efficientnetb0_v2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# LOAD BASELINE MODEL
# ============================================================

print("\nLoading baseline model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print("\nRows = Actual")
print("Columns = Predicted\n")

print("       ", end="")

for name in class_names:
    print(f"{name:>7}", end="")

print()

for i, row in enumerate(cm):

    print(f"{class_names[i]:>7}", end="")

    for value in row:
        print(f"{value:>7}", end="")

    print()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print("========================================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("========================================")