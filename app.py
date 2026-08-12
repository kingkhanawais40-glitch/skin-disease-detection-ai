from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "skin_disease_efficientnetb0_v2.keras"
UPLOAD_FOLDER = "uploads"

IMG_SIZE = (224, 224)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

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


# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING SKIN DISEASE MODEL")
print("========================================")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


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

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions[0])
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    probabilities = {}

    for i, class_name in enumerate(CLASS_NAMES):

        probabilities[class_name] = round(
            float(predictions[0][i] * 100),
            2
        )

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities
    }


# ============================================================
# HOME / TEST
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ============================================================
# PREDICT API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "error": "Only JPG, JPEG and PNG images are allowed."
        }), 400

    filename = secure_filename(
        file.filename
    )

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    try:

        result = predict_image(
            image_path
        )

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        # Delete uploaded image after prediction
        if os.path.exists(image_path):

            os.remove(image_path)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("SKIN DISEASE DETECTION API")
    print("========================================")

    print("Server: http://127.0.0.1:5000")
    print("Prediction: POST /predict")

    print("========================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )