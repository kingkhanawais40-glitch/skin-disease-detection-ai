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