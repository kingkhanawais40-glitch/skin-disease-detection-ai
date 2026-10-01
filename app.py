from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "FINAL_skin_disease_efficientnetb0_stage2_68pct.keras"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

IMG_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# Create upload folder if it doesn't exist
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING SKIN DISEASE MODEL")
print("========================================")

print(f"Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

model = tf.keras.models.load_model(MODEL_PATH)

print("MODEL FILE:", MODEL_PATH)
print("MODEL NAME:", model.name)
print("TOTAL PARAMETERS:", model.count_params())
print("INPUT SHAPE:", model.input_shape)
print("OUTPUT SHAPE:", model.output_shape)

print("Model loaded successfully.")
print(f"Input size: {IMG_SIZE}")
print(f"Number of classes: {len(CLASS_NAMES)}")
print(f"Classes: {CLASS_NAMES}")

print("========================================\n")


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # --------------------------------------------------------
    # Convert image to array
    # --------------------------------------------------------

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    # Shape becomes:
    # (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    probabilities_array = predictions[0]

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities_array)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        probabilities_array[predicted_index] * 100
    )

    # --------------------------------------------------------
    # All class probabilities
    # --------------------------------------------------------

    probabilities = {}

    for i, class_name in enumerate(CLASS_NAMES):

        probabilities[class_name] = round(
            float(probabilities_array[i] * 100),
            2
        )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities
    }


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check image field
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400

    # --------------------------------------------------------
    # Check file extension
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "error": "Only JPG, JPEG and PNG images are allowed."
        }), 400

    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    file.save(image_path)

    try:

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        result = predict_image(
            image_path
        )

        # ----------------------------------------------------
        # API response
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            "result": result,
            "disclaimer": (
                "This system is for educational and research "
                "purposes only and is not a medical diagnostic tool."
            )
        })

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return jsonify({
            "success": False,
            "error": "Prediction failed. Please try another image."
        }), 500

    finally:

        # ----------------------------------------------------
        # Delete uploaded image
        # ----------------------------------------------------

        if os.path.exists(image_path):

            try:
                os.remove(image_path)

            except Exception as cleanup_error:

                print(
                    f"Cleanup error: {cleanup_error}"
                )


# ============================================================
# FILE TOO LARGE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Image file is too large. Maximum size is 10 MB."
    }), 413


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("SKIN DISEASE DETECTION API")
    print("========================================")

    print("Model: EfficientNetB0 Stage 2")
    print("Input: 224x224")
    print("Classes: 7")
    print("Server: http://127.0.0.1:5000")
    print("Prediction: POST /predict")

    print("========================================\n")

    app.run(
    host="127.0.0.1",
    port=5000,
    debug=False,
    use_reloader=False
)