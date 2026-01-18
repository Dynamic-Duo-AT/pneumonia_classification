FROM python:3.11-slim

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi pydantic uvicorn \
    torch torchvision \
    python-dotenv Pillow python-multipart requests

COPY src ./
COPY .env ./

EXPOSE $PORT

CMD sh -c "uvicorn pneumonia.api:app --port $PORT --host 0.0.0.0 --workers 1"
