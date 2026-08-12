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
# LOAD TEST DATASET
# ============================================================

print("\n========================================")
print("LOADING TEST DATASET")
print("========================================")

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
# LOAD V2 MODEL
# ============================================================

print("\n========================================")
print("LOADING V2 MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("V2 model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n========================================")
print("GENERATING PREDICTIONS")
print("========================================")

y_true = []
y_pred = []
y_confidence = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    confidence = np.max(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )

    y_confidence.extend(
        confidence
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_confidence = np.array(y_confidence)


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

print(
    "       " +
    "".join(
        f"{name:>8}"
        for name in class_names
    )
)

for i, row in enumerate(cm):

    print(
        f"{class_names[i]:>7}" +
        "".join(
            f"{value:>8}"
            for value in row
        )
    )


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

overall_accuracy = np.mean(
    y_true == y_pred
)

print("========================================")
print(
    f"Overall Test Accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)
print("========================================")


# ============================================================
# CLASS-BY-CLASS ACCURACY
# ============================================================

print("\n========================================")
print("CLASS-BY-CLASS ACCURACY")
print("========================================")

for i, class_name in enumerate(class_names):

    total = np.sum(
        y_true == i
    )

    correct = np.sum(
        (y_true == i) &
        (y_pred == i)
    )

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:>7}: "
        f"{correct}/{total} "
        f"({accuracy * 100:.2f}%)"
    )


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n========================================")
print("WRONG PREDICTIONS")
print("========================================")

wrong_indices = np.where(
    y_true != y_pred
)[0]

print(
    f"Total wrong predictions: "
    f"{len(wrong_indices)}"
)


# ============================================================
# WRONG PREDICTION DETAILS
# ============================================================

file_paths = []

for class_name in class_names:

    class_folder = os.path.join(
        TEST_DIR,
        class_name
    )

    for filename in sorted(
        os.listdir(class_folder)
    ):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            file_paths.append(
                os.path.join(
                    class_folder,
                    filename
                )
            )


print("\nWrong prediction details:\n")

for index in wrong_indices:

    actual = class_names[
        y_true[index]
    ]

    predicted = class_names[
        y_pred[index]
    ]

    confidence = (
        y_confidence[index] * 100
    )

    filename = file_paths[index]

    print(
        f"Actual: {actual:>6} | "
        f"Predicted: {predicted:>6} | "
        f"Confidence: {confidence:>6.2f}% | "
        f"File: {filename}"
    )


# ============================================================
# SAVE WRONG PREDICTIONS
# ============================================================

output_file = "wrong_predictions.txt"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "V2 WRONG PREDICTIONS\n"
    )

    f.write(
        "====================\n\n"
    )

    for index in wrong_indices:

        actual = class_names[
            y_true[index]
        ]

        predicted = class_names[
            y_pred[index]
        ]

        confidence = (
            y_confidence[index] * 100
        )

        filename = file_paths[index]

        f.write(
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Confidence: {confidence:.2f}% | "
            f"File: {filename}\n"
        )


print(
    f"\nWrong predictions saved to: "
    f"{output_file}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("DETAILED EVALUATION COMPLETE")
print("========================================")