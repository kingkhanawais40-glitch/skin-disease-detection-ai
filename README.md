# Skin Disease Detection AI

An AI-powered skin disease classification web application built with **Python, TensorFlow, EfficientNetB0, and Flask**.

The application allows users to upload a skin lesion image and receive a predicted disease class along with the model's confidence score and class probability distribution.

> **Important:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## Overview

This project uses a deep learning image classification model based on **EfficientNetB0** to classify skin lesion images into seven categories.

The trained model was integrated into a Flask web application with a professional responsive interface for image upload, prediction, confidence visualization, and probability analysis.

---

## Model

**Architecture:** EfficientNetB0  
**Pretrained weights:** ImageNet  
**Input size:** 224 × 224  
**Number of classes:** 7  
**Framework:** TensorFlow / Keras

### Final Model

The selected model is the **Stage 2 EfficientNetB0 model**.

- Test Accuracy: **68.04%**
- Macro F1 Score: **54.78%**
- Weighted F1 Score: **67.07%**
- Total Parameters: **4,058,538**
- Trainable Parameters: **8,967**

### Model File

```text
FINAL_skin_disease_efficientnetb0_stage2_68pct.keras

Disease Classes

The model predicts seven skin disease categories:

| Class | Description                                   |
| ----- | --------------------------------------------- |
| AKIEC | Actinic Keratoses / Intraepithelial Carcinoma |
| BCC   | Basal Cell Carcinoma                          |
| BKL   | Benign Keratosis-like Lesions                 |
| DF    | Dermatofibroma                                |
| MEL   | Melanoma                                      |
| NV    | Melanocytic Nevi                              |
| VASC  | Vascular Lesions                              |


Dataset

The project uses a dataset derived from the ISIC 2018 Task 3 skin lesion classification dataset.

The final project dataset contains:

2,808 unique images

Class Distribution
Class	Images
AKIEC	150
BCC	192
BKL	321
DF	115
MEL	324
NV	1,564
VASC	142
Total	2,808
Dataset Split
Split	Images
Training	2,176
Validation	316
Testing	316
Total	2,808

A leakage check was performed to ensure that image IDs did not overlap between the training, validation, and test sets.

Training

The final model uses:

EfficientNetB0 pretrained on ImageNet
Frozen base model
Data augmentation
Random horizontal flipping
Random rotation
Random zoom
Random translation
Adam optimizer
Sparse categorical crossentropy
Mild class weighting

Input images are resized to:

224 × 224 × 3
Performance

Final test-set performance:

Metric	Score
Test Accuracy	68.04%
Macro F1	54.78%
Weighted F1	67.07%

The model performs differently across individual classes, with stronger performance on some classes than others.

Because the dataset is imbalanced, accuracy alone should not be considered sufficient for evaluating model performance.

Tech Stack
Machine Learning
Python
TensorFlow
Keras
EfficientNetB0
NumPy
Pillow
Backend
Flask
Flask-CORS
Frontend
HTML
CSS
JavaScript
Development
Google Colab
VS Code
Git
GitHub
Project Structure
skin-disease-detection-ai/
│
├── app.py
├── FINAL_skin_disease_efficientnetb0_stage2_68pct.keras
├── requirements.txt
├── .gitignore
├── README.md
│
├── dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── upload/

The dataset and local virtual environment are excluded from Git using .gitignore.

Installation
1. Clone the repository
git clone https://github.com/kingkhanawais40-glitch/skin-disease-detection-ai
2. Open the project
cd skin-disease-detection-ai
3. Create a virtual environment

Windows:

python -m venv .venv
4. Activate the environment
.venv\Scripts\Activate.ps1
5. Install dependencies
pip install -r requirements.txt
Run the Application

Start the Flask application:

python app.py

The application will run locally at:

http://127.0.0.1:5000

Open the address in your browser.

How It Works

The prediction workflow is:

User uploads image
        ↓
Image validation
        ↓
Image preprocessing
        ↓
Resize to 224 × 224
        ↓
EfficientNetB0
        ↓
7-class prediction
        ↓
Predicted class
        ↓
Confidence score
        ↓
Class probability distribution
Application Features
Skin lesion image upload
Drag-and-drop support
Image preview
JPG / JPEG / PNG validation
10 MB upload limit
EfficientNetB0 prediction
Confidence score
Class probability visualization
Responsive design
Mobile-friendly interface
Loading state
Clear/reset functionality
Medical/research disclaimer
Security

The Flask backend handles image uploads and prediction processing.

Uploaded images are temporarily processed for prediction and are not intended to be permanently stored by the application.

The application also limits uploaded file size to help prevent unnecessarily large uploads.

Limitations

This project has several limitations:

The dataset is relatively small compared with large-scale medical imaging datasets.
The dataset is class-imbalanced.
Model performance varies between classes.
The model has not been clinically validated.
The predictions should not be interpreted as medical diagnoses.
Performance on real-world images may differ from the test dataset.
Future Improvements

Potential future improvements include:

Larger and more diverse datasets
Better class balancing
Transfer-learning experiments
Fine-tuning EfficientNetB0
Hyperparameter optimization
Model explainability using Grad-CAM
Additional evaluation metrics
Improved validation strategies
Cloud deployment
API-based inference
Model versioning
More extensive real-world testing
Disclaimer

This project is developed for educational and research purposes.

The predictions generated by this application are not medical diagnoses.

Users should consult a qualified healthcare professional for medical evaluation, diagnosis, and treatment.

Author

Muhammad Awais

BS Software Engineering
Sarhad University of Science & IT, Peshawar

Interests:

Machine Learning
Computer Vision
Artificial Intelligence
Software Engineering
License

This project is intended for educational and research purposes.