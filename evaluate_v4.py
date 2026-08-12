import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset/test"

MODEL_PATH = "skin_disease_efficientnetb0_v4.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n========================================")
print("V4 MODEL EVALUATION")
print("========================================")

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

print("\nNumber of classes:", len(class_names))

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V4 model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("V4 model loaded successfully.")

# ============================================================
# MODEL TEST
# ============================================================

print("\n========================================")
print("MODEL TEST")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# GENERATE PREDICTIONS
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

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n========================================")
print("ACCURACY")
print("========================================")

print(
    f"Calculated Accuracy: {accuracy * 100:.2f}%"
)

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

print("Rows = Actual")
print("Columns = Predicted\n")

print(
    "       " +
    " ".join(
        f"{name:>7}"
        for name in class_names
    )
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:>7} " +
        " ".join(
            f"{value:7d}"
            for value in row
        )
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# ============================================================
# PER CLASS ACCURACY
# ============================================================

print("========================================")
print("PER-CLASS RECALL")
print("========================================")

for i, class_name in enumerate(class_names):

    total = np.sum(
        y_true == i
    )

    correct = cm[i, i]

    recall = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name}: "
        f"{correct}/{total} "
        f"({recall * 100:.2f}%)"
    )

# ============================================================
# WRONG PREDICTIONS
# ============================================================

wrong_indices = np.where(
    y_true != y_pred
)[0]

print("\n========================================")
print("WRONG PREDICTIONS")
print("========================================")

print(
    "Total wrong predictions:",
    len(wrong_indices)
)

# ============================================================
# SAVE REPORT
# ============================================================

with open(
    "v4_evaluation_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SKIN DISEASE DETECTION - V4 EVALUATION\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "CONFUSION MATRIX\n"
    )

    file.write(
        "Rows = Actual\n"
    )

    file.write(
        "Columns = Predicted\n\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nCLASSIFICATION REPORT\n\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nPER-CLASS RECALL\n"
    )

    for i, class_name in enumerate(class_names):

        total = np.sum(
            y_true == i
        )

        correct = cm[i, i]

        recall = (
            correct / total
            if total > 0
            else 0
        )

        file.write(
            f"{class_name}: "
            f"{correct}/{total} "
            f"({recall * 100:.2f}%)\n"
        )

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "\nReport saved to:"
)

print(
    "v4_evaluation_report.txt"
)