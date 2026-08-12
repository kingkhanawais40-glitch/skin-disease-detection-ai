import os
import sys
import numpy as np
import tensorflow as tf

# ==============================
# CONFIG
# ==============================

MODEL_PATH = "skin_disease_efficientnetb0_v2.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ==============================
# CHECK MODEL
# ==============================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found.")
    print(MODEL_PATH)
    sys.exit(1)


# ==============================
# CHECK IMAGE
# ==============================

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python predict.py \"path_to_image.jpg\"")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("\nERROR: Image not found:")
    print(image_path)
    sys.exit(1)


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==============================
# LOAD IMAGE
# ==============================

print("\nLoading image...")

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ==============================
# PREDICTION
# ==============================

print("Generating prediction...")

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

predicted_class = CLASS_NAMES[
    predicted_index
]

confidence = (
    predictions[0][predicted_index] * 100
)


# ==============================
# ALL CLASS PROBABILITIES
# ==============================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print(
    f"\nPredicted Class: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("\nClass probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    probability = (
        predictions[0][i] * 100
    )

    print(
        f"{class_name:>6}: "
        f"{probability:.2f}%"
    )

print("\n========================================")