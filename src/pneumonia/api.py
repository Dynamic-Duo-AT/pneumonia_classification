import os
from contextlib import asynccontextmanager
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from PIL import Image
from datetime import datetime, timezone

from pneumonia.model import Model


def download_model(gcs_uri: str, local_path: Path) -> None:
    """
    Download a model from Google Cloud Storage to a local path.

    Args:
        gcs_uri: GCS URI of the model (e.g., gs://bucket_name/path/to/model.pth).
        local_path: Local path to save the downloaded model.
    """
    assert gcs_uri.startswith("gs://")

    # Convert gs://bucket/path -> https://storage.googleapis.com/bucket/path
    http_url = "https://storage.googleapis.com/" + gcs_uri[len("gs://") :]

    local_path.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(http_url, timeout=60)
    r.raise_for_status()
    local_path.write_bytes(r.content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, mean, std, size
    load_dotenv()
    print("Loading model")
    model = Model()
    # load model weights
    local_model = Path("/tmp/model.pth")
    download_model("gs://models_pneumonia/model.pth", local_model)
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

def add_to_database(
        now: str,
        filename: str,
        image: Image.Image,
        pred: str,
    ) -> None:
    """Function to add prediction to database."""
    db_path = Path("db/")
    db_path.mkdir(parents=True, exist_ok=True)
    with open(db_path / "prediction_database.csv", "a") as file:
        file.write(f"{now},{filename},{pred}\n")
    
    # Save the image
    image_path = db_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    image.save(image_path / filename)

@app.get("/")
async def read_root():
    """Root endpoint."""
    return {"Welcome":"To the Pneumonia Detection API"}

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

    now = datetime.now(timezone.utc).isoformat()
    # convert image back to PIL for saving
    i_image = Image.fromarray((i_image.numpy() * 255).astype('uint8'))
    background_tasks.add_task(add_to_database,now, data.filename, i_image, preds[0])
              
    return {"filename": data.filename, "pred": preds[0], "sigmoid": sigmoid_output.item()}
