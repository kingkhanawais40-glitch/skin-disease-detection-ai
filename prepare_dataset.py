import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================
# CONFIGURATION
# ==============================

SOURCE_IMAGE_DIR = "ham10000_images"
METADATA_FILE = "HAM10000_metadata.csv"

OUTPUT_DIR = "dataset"

CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

RANDOM_STATE = 42

# ==============================
# CHECK INPUTS
# ==============================

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

if not os.path.exists(SOURCE_IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found: {SOURCE_IMAGE_DIR}"
    )

# ==============================
# LOAD METADATA
# ==============================

print("Loading metadata...")

df = pd.read_csv(METADATA_FILE)

print(f"Total records: {len(df)}")

# ==============================
# KEEP REQUIRED CLASSES
# ==============================

df = df[df["dx"].isin(CLASSES)].copy()

print(f"Records after class filtering: {len(df)}")

print("\nClass distribution:")
print(df["dx"].value_counts())

# ==============================
# TRAIN / TEMP SPLIT
# ==============================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["dx"],
    random_state=RANDOM_STATE
)

# ==============================
# VALIDATION / TEST SPLIT
# ==============================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["dx"],
    random_state=RANDOM_STATE
)

print("\nDataset split:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

# ==============================
# CREATE DIRECTORIES
# ==============================

for split in ["train", "validation", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )

# ==============================
# COPY IMAGES
# ==============================

def copy_images(dataframe, split_name):

    print(f"\nPreparing {split_name} dataset...")

    copied = 0
    missing = 0

    for _, row in dataframe.iterrows():

        image_id = row["image_id"]
        class_name = row["dx"]

        source = os.path.join(
            SOURCE_IMAGE_DIR,
            image_id + ".jpg"
        )

        destination = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name,
            image_id + ".jpg"
        )

        if os.path.exists(source):

            shutil.copy2(source, destination)
            copied += 1

        else:

            missing += 1

    print(f"Copied: {copied}")
    print(f"Missing: {missing}")


# ==============================
# RUN PREPARATION
# ==============================

copy_images(train_df, "train")
copy_images(validation_df, "validation")
copy_images(test_df, "test")

print("\n================================")
print("DATASET PREPARATION COMPLETED")
print("================================")