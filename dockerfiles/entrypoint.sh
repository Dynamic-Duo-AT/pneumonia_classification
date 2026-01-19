#!/usr/bin/env bash
set -euo pipefail

echo "Running dvc pull..."
# Make sure there's a writable cache (default /tmp is fine)
export DVC_CACHE_DIR="${DVC_CACHE_DIR:-/tmp/dvc-cache}"

# Pull tracked artifacts (data/models/etc.) defined by your DVC files
uv run dvc pull

echo "Starting training..."
exec uv run inv exp1-train "$@"