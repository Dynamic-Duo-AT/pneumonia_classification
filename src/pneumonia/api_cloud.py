import datetime
import json
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image

from pneumonia.model import Model


# Define model and device configuration
BUCKET_NAME_MODEL = "models_pneumonia"
MODEL_FILE_NAME = "model.pth"
BUCKET_NAME_DATA = "inference_data_pneumonia"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, mean, std, size
    load_dotenv()
    print("Loading model")
    # Download the model from GCP
    download_model_from_gcp()
    model = Model()
    # load model weights
    local_model = Path("model.pth")
    model.load_state_dict(torch.load(local_model, map_location=torch.device("cpu")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load preprocessing params
    mean = os.getenv("MEAN_PIXEL")
    std = os.getenv("STD_PIXEL")
    size = os.getenv("SIZE")

    yield

    print("Cleaning up")
    del model, device, mean, std, size


app = FastAPI(lifespan=lifespan)


def download_model_from_gcp():
    """Download the model from GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME_MODEL)
    blob = bucket.blob(MODEL_FILE_NAME)
    blob.download_to_filename(MODEL_FILE_NAME)
    print(f"Model {MODEL_FILE_NAME} downloaded from GCP bucket {BUCKET_NAME_MODEL}.")


def save_prediction_to_gcp(file_name: str, pred: str, image: Image.Image):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME_DATA)
    time = datetime.datetime.now(tz=datetime.UTC)

    # Prepare prediction data
    data = {
        "filename": file_name,
        "prediction": pred,
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }
    blob = bucket.blob(f"json/prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")

    # Save the image
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    image_blob = bucket.blob(f"images/{file_name}")
    image_blob.content_type = "image/png"
    image_blob.upload_from_file(buf, rewind=True)
    print("Image saved to GCP bucket.")


@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"Welcome": "To the Pneumonia Detection API"}


@app.post("/pred/")
async def pred(data: UploadFile, background_tasks: BackgroundTasks):
    """Generate a pred for an image."""
    contents = await data.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    # preprocess
    i_image = Image.open(data.file)
    if i_image.mode != "L":
        i_image = i_image.convert(mode="L")
    i_image = i_image.resize((int(size), int(size)))

    # check if image is stored as 8-bit or float
    _, hi = i_image.getextrema()
    if hi > 1:
        i_image = torch.tensor(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(i_image.tobytes()))).float().view(int(size), int(size))
            / 255.0
        )  # [size,size]
    else:
        i_image = torch.tensor(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(i_image.tobytes()))).float().view(int(size), int(size))
        )  # [size,size]

    # normalize
    i_image_processed = (i_image - float(mean)) / float(std)  # [size,size]
    i_image_processed = i_image_processed.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,size,size]

    # pred
    model.eval()
    with torch.no_grad():
        output = model(i_image_processed)
        sigmoid_output = torch.sigmoid(output)
        preds_binary = (sigmoid_output > 0.5).float()
        preds = ["Pneumonia" if pred.item() == 1.0 else "Normal" for pred in preds_binary]

    # convert image back to PIL for saving
    i_image = Image.fromarray((i_image.numpy() * 255).astype("uint8"))
    background_tasks.add_task(save_prediction_to_gcp, data.filename, preds[0], i_image)
    return {"filename": data.filename, "pred": preds[0], "sigmoid": sigmoid_output.item()}
