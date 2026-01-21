#!/usr/bin/env bash
set -euo pipefail

echo "Running dvc pull..."
export DVC_CACHE_DIR="${DVC_CACHE_DIR:-/tmp/dvc-cache}"
uv run dvc pull

echo "Starting API..."
exec uv run uvicorn pneumonia.cloud_drift:app --host 0.0.0.0 --port $PORT --workers 1