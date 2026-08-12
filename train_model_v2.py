import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

MODEL_PATH = "skin_disease_efficientnetb0_v2.keras"

# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking dataset...")

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

print("Dataset folders found successfully.")

# ============================================================
# LOAD DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

# Your classes are balanced, so this will produce weights
# close to 1.0. We calculate them automatically instead
# of manually assigning arbitrary weights.

train_labels = []

for _, labels in train_ds:
    train_labels.extend(labels.numpy())

train_labels = np.array(train_labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for i, name in enumerate(class_names):
    print(f"{name}: {class_weights[i]:.4f}")

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================================
# EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(0.45)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING V2 TRAINING")
print("========================================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# TEST
# ============================================================

print("\n========================================")
print("FINAL TEST")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ============================================================
# SAVE
# ============================================================

model.save(MODEL_PATH)

print("\nBest model saved:")
print(MODEL_PATH)

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "V2 Training vs Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "v2_accuracy_graph.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "V2 Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "v2_loss_graph.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\n========================================")
print("V2 TRAINING COMPLETE")
print("========================================")