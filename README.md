# 🩺 Skin Disease Detection AI

An AI-powered skin image classification web application built with **Python, TensorFlow, Keras, EfficientNetB0, and Flask**.

The application allows users to upload a skin image through a web interface and receive an AI-generated prediction across **7 supported skin disease classes**, along with prediction confidence and class probabilities.

> ⚠️ **Medical Disclaimer:** This project is an academic/portfolio AI prototype. It is not a medical diagnostic tool and should not be used as a replacement for a qualified healthcare professional.

---

## ✨ Features

- 🧠 EfficientNetB0-based image classification
- 📷 JPG, JPEG, and PNG image upload
- 🤖 Prediction across 7 skin disease classes
- 📊 AI confidence score
- 📈 Class probability distribution
- 🌐 Flask REST API
- 💻 Browser-based interface
- 🔒 Secure filename handling
- 🧹 Automatic deletion of uploaded images after prediction
- ❌ Invalid file validation
- ❌ Empty upload validation
- ⚡ Fast local prediction
- 📱 Responsive user interface

---

# 🧠 AI Model

The application uses a fine-tuned **EfficientNetB0** deep learning model for skin disease image classification.

## Final Model

| Property | Value |
|---|---|
| Model | EfficientNetB0 V2 |
| Framework | TensorFlow / Keras |
| Input Size | 224 × 224 |
| Number of Classes | 7 |
| Test Images | 105 |
| Test Accuracy | 52.38% |
| Model File | `skin_disease_efficientnetb0_v2.keras` |

---

# 🏷️ Supported Classes

The model supports the following seven classes:

| Class | Description |
|---|---|
| AKIEC | Actinic Keratoses / Intraepithelial Carcinoma |
| BCC | Basal Cell Carcinoma |
| BKL | Benign Keratosis-like Lesions |
| DF | Dermatofibroma |
| MEL | Melanoma |
| NV | Melanocytic Nevi |
| VASC | Vascular Lesions |

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │        USER          │
                    └──────────┬───────────┘
                               │
                               │ Upload Image
                               ▼
                    ┌──────────────────────┐
                    │    WEB INTERFACE     │
                    │    HTML/CSS/JS       │
                    └──────────┬───────────┘
                               │
                               │ HTTP POST
                               ▼
                    ┌──────────────────────┐
                    │      FLASK API       │
                    │      /predict        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ IMAGE PREPROCESSING  │
                    │     224 × 224        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  EfficientNetB0 V2   │
                    │     AI MODEL         │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │          PREDICTION             │
              │                                 │
              │  • Predicted Class              │
              │  • Confidence                   │
              │  • Class Probabilities           │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    WEB INTERFACE     │
                    │    RESULT DISPLAY    │
                    └──────────────────────┘



                    Technologies Used
Artificial Intelligence / Machine Learning
Python
TensorFlow
Keras
EfficientNetB0
NumPy
Matplotlib
Backend
Flask
Flask-CORS
Werkzeug
Frontend
HTML5
CSS3
JavaScript
Development Tools
Visual Studio Code
Python Virtual Environment
Git / GitHub


                    Project Structure
Skin-Disease-Detection/
│
├── dataset/
│   ├── train/
│   │   ├── akiec/
│   │   ├── bcc/
│   │   ├── bkl/
│   │   ├── df/
│   │   ├── mel/
│   │   ├── nv/
│   │   └── vasc/
│   │
│   ├── validation/
│   │   ├── akiec/
│   │   ├── bcc/
│   │   ├── bkl/
│   │   ├── df/
│   │   ├── mel/
│   │   ├── nv/
│   │   └── vasc/
│   │
│   └── test/
│       ├── akiec/
│       ├── bcc/
│       ├── bkl/
│       ├── df/
│       ├── mel/
│       ├── nv/
│       └── vasc/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── uploads/
│
├── skin_disease_efficientnetb0_v2.keras
│
├── app.py
├── predict.py
├── evaluate_model.py
├── requirements.txt
├── .gitignore
└── README.md


                    Model Evaluation

The final V2 model was evaluated using a separate test dataset.

Test Images:       105
Number of Classes: 7
Test Accuracy:     52.38%
Class-wise Performance
Class	Recall
AKIEC	40.00%
BCC	46.67%
BKL	20.00%
DF	93.33%
MEL	40.00%
NV	60.00%
VASC	66.67%


                    Predicted Class: AKIEC
Confidence: 30.59%

Class probabilities:

AKIEC    30.59%
BCC      26.15%
DF       18.18%
NV       11.00%
BKL       6.60%
MEL       5.68%
VASC      1.79%




                    Project Status: Completed

AI Model:          EfficientNetB0 V2
Backend:           Flask
Frontend:          HTML / CSS / JavaScript
REST API:          Working
Image Upload:      Working
Prediction:        Working
Error Handling:    Working
Browser Testing:   Passed


Future Improvements

Potential improvements include:

Larger and more diverse training datasets
Better class balancing
Advanced image preprocessing
Improved data augmentation
Hyperparameter optimization
Additional transfer-learning experiments
Explainable AI using Grad-CAM
Model confidence calibration
Cloud deployment
Prediction history
User authentication
Mobile optimization
Doctor/healthcare professional review workflow

Testing

The application has been tested for:

✅ Image upload
✅ AI prediction
✅ Confidence calculation
✅ Class probability display
✅ JPG image
✅ JPEG image
✅ PNG image
✅ Invalid file upload
✅ Empty upload
✅ Flask API response
✅ Model loading
✅ Browser-to-API communication


Privacy

This project is designed as a local portfolio application.

Uploaded images are processed temporarily and removed after prediction by the Flask API.

Do not upload sensitive or personally identifiable medical images to publicly hosted versions of this application unless appropriate privacy and security controls have been implemented.


Project Highlights
✔ EfficientNetB0 Deep Learning Model
✔ 7-Class Skin Image Classification
✔ Flask REST API
✔ Browser-Based Image Analysis
✔ Confidence & Probability Visualization
✔ Input Validation
✔ Temporary Image Processing
✔ Clean Portfolio-Ready Interface
✔ Complete End-to-End AI Application


Disclaimer

⚠️ IMPORTANT MEDICAL DISCLAIMER

This application is developed strictly for educational, research, and portfolio purposes.

It is an experimental AI prototype and has not been clinically validated.

The predictions generated by this system should not be considered medical advice, diagnosis, or treatment recommendations.

Always consult a qualified healthcare professional for medical evaluation and diagnosis.



Author
Muhammad Awais

BS Software Engineering

Interested in:

Artificial Intelligence
Machine Learning
Software Engineering
Web Development
Computer Vision