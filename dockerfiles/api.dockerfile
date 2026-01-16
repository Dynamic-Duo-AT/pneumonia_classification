FROM python:3.11-slim

WORKDIR /

# Install only what you actually need to build Python wheels (and git if you truly need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps (fewer layers)
RUN pip install --no-cache-dir fastapi pydantic uvicorn torch torchvision python-dotenv Pillow python-multipart

COPY src ./
COPY models ./models
COPY .env ./


# Optional: default port if PORT isn't set by the platform
ENV PORT=8000
EXPOSE 8000

CMD sh -c "uvicorn pneumonia.api:app --host 0.0.0.0 --port ${PORT} --workers 1"
