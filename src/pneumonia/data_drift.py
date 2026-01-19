import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import torch
from evidently.metrics import DataDriftTable
from evidently.report import Report
from torchvision import datasets, transforms

# Load environment variables
load_dotenv()

path_db = "db/prediction_database.csv"
path_training_data = "data/raw/train/"

# Load csv data
db = pd.read_csv(path_db)

# Load training data
train_data = datasets.ImageFolder(
    root=path_training_data,
    transform=transforms.Compose([
        transforms.Resize((int(os.getenv("SIZE")), int(os.getenv("SIZE")))),
        transforms.ToTensor()
    ])
)

# Load database files
db_images = []
for filename in db['Filename']:
    image_path = os.path.join("db/images/", filename)
    image = datasets.folder.default_loader(image_path)
    image = transforms.Compose([
        transforms.Resize((int(os.getenv("SIZE")), int(os.getenv("SIZE")))),
        transforms.ToTensor()
    ])(image)
    db_images.append(image)

db_images_tensor = torch.stack(db_images)


def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        img = img[0].numpy().squeeze()  # assuming grayscale images
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)

train_features = extract_features(train_data)
db_features = extract_features(db_images_tensor)

feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

train_df = np.column_stack((train_features, ["Train"] * train_features.shape[0]))
db_df = np.column_stack((db_features, ["Database"] * db_features.shape[0]))

combined_features = np.vstack((train_df, db_df))

feature_df = pd.DataFrame(combined_features, columns=feature_columns + ["Dataset"])
feature_df[feature_columns] = feature_df[feature_columns].astype(float)

reference_data = feature_df[feature_df["Dataset"] == "Train"].drop(columns=["Dataset"])
current_data = feature_df[feature_df["Dataset"] == "Database"].drop(columns=["Dataset"])

report = Report(metrics=[DataDriftTable()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("data_drift.html")
