FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY uv.lock pyproject.toml ./
RUN uv sync --locked --no-cache --no-install-project

# Copy only source + configs + tooling (NO data/models)
COPY src ./src
COPY configs ./configs
COPY tasks.py ./tasks.py
COPY README.md ./README.md
COPY .dvc ./.dvc
COPY *.dvc ./

# Entrypoint that runs dvc pull then training
COPY dockerfiles/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]