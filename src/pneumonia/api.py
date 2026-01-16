import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from pneumonia.model import Model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device, mean, std, size
    load_dotenv()
    print("Loading model")
    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mean = os.getenv("MEAN_PIXEL")
    std = os.getenv("STD_PIXEL")
    size = os.getenv("SIZE")

    yield

    print("Cleaning up")
    del model, device, mean, std, size


app = FastAPI(lifespan=lifespan)


@app.post("/pred/")
async def pred(data: UploadFile = File(...)):
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
    i_image = (i_image - float(mean)) / float(std)  # [size,size]
    i_image = i_image.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,size,size]

    # pred
    with torch.no_grad():
        output = model(i_image)
        sigmoid_output = torch.sigmoid(output)
        preds_binary = (sigmoid_output > 0.5).float()
        preds = ["Pneumonia" if pred.item() == 1.0 else "Normal" for pred in preds_binary]
    return {"filename": data.filename, "pred": preds[0], "sigmoid": sigmoid_output.item()}
