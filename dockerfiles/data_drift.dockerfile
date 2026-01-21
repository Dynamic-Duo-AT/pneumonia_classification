FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY .dvc ./.dvc
COPY *.dvc ./
COPY README.md README.md
COPY db ./db

RUN uv sync --locked --no-cache --no-install-project

COPY dockerfiles/entrypoint_drift.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE $PORT
ENTRYPOINT ["/entrypoint.sh"]
