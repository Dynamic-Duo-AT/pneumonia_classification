import os
from glob import glob
from pathlib import Path

import anyio
import numpy as np
import pandas as pd
from evidently.metrics import DataDriftTable
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from PIL import Image


def extract_features(images: list) -> np.ndarray:
    """
    Extract basic image features from a set of images.

    Args:
        images (list): List of PIL Image objects.

    Returns:
        np.ndarray: Array of extracted features.
    """
    features = []
    for img in images:
        img = np.array(img)
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)


BUCKET_NAME = "inference_data_pneumonia"


def lifespan(app: FastAPI):
    """
    Load the data and class names before the application starts.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    global training_data
    print("Loading training data...")
    # load raw data
    train_data_dir = Path("data/raw/train/")
    # read all images using glob
    train_files = glob(os.path.join(train_data_dir, "**", "*.jpeg"), recursive=True)
    print(f"Found {len(train_files)} training files.")
    # load training data
    training_data_list = []
    for file in train_files:
        img = Image.open(file).convert("L")
        # rezize to 384x384
        img = img.resize((384, 384))
        training_data_list.append(img)

    print(f"Loaded {len(training_data_list)} training images.")
    training_data = pd.DataFrame(
        {
            "image": training_data_list,
        }
    )

    yield

    del training_data


app = FastAPI(lifespan=lifespan)


def load_files(directory: Path) -> pd.DataFrame:
    """
    Load prediction files from the directory.

    Args:
        directory (Path): Directory where the files are stored.

    Returns:
        pd.DataFrame: DataFrame containing the images and their labels.
    """
    # Download the latest prediction files from the GCP bucket
    download_files()

    # Get all jpeg files in the directory
    files = glob(os.path.join(directory, "*.jpeg"))

    # Sort files based on when they where created
    files = sorted(files, key=os.path.getmtime)

    # Load or process the files as needed
    images = []
    for file in files:
        img = Image.open(file).convert("L")
        img = Image.open(file).convert("L")
        # rezize to 384x384
        img = img.resize((384, 384))
        images.append(img)

    data_new = pd.DataFrame(
        {
            "image": images,
        }
    )

    return data_new


def download_files() -> None:
    """
    Download prediction files from the GCP bucket.
    """
    print(f"Downloading files from GCP bucket '{BUCKET_NAME}'...")
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="images/"))
    for blob in blobs:
        # Download only .jpeg files
        if blob.name.endswith(".jpeg"):
            filename = Path(blob.name).name
            local_path = Path("db/cloud") / filename
            blob.download_to_filename(str(local_path))


@app.get("/report", response_class=HTMLResponse)
async def get_report():
    """
    Generate and return the report.
    """
    prediction_data = load_files(Path("db/cloud"))

    # Extract features
    print("Extracting image features.")
    train_features = extract_features(training_data["image"])
    db_features = extract_features(prediction_data["image"])

    feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

    # Formatting data for Evidently
    train_df = np.column_stack((train_features, ["Train"] * train_features.shape[0]))
    db_df = np.column_stack((db_features, ["Database"] * db_features.shape[0]))
    combined_features = np.vstack((train_df, db_df))
    feature_df = pd.DataFrame(combined_features, columns=feature_columns + ["Dataset"])
    feature_df[feature_columns] = feature_df[feature_columns].astype(float)

    # Generate report
    reference_data = feature_df[feature_df["Dataset"] == "Train"].drop(columns=["Dataset"])
    current_data = feature_df[feature_df["Dataset"] == "Database"].drop(columns=["Dataset"])
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("data_drift.html")

    async with await anyio.open_file("data_drift.html", mode="r", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
