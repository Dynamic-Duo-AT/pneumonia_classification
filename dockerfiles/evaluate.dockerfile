FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY tasks.py ./tasks.py
COPY models ./models
COPY data ./data
COPY README.md README.md

RUN uv sync --locked --no-cache --no-install-project
ENTRYPOINT ["uv", "run", "inv", "exp1-test"]