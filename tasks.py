import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pneumonia"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def exp1_workers(ctx: Context) -> None:
    """Run experiment exp1"""
    ctx.run("uv run python -m pneumonia.train --config-name exp1_workers", echo=True, pty=not WINDOWS)
    ctx.run("uv run python -m pneumonia.evaluate --config-name exp1_workers", echo=True, pty=not WINDOWS)


@task
def exp1(ctx: Context) -> None:
    """Run experiment exp1"""
    ctx.run("uv run python -m pneumonia.train --config-name exp1", echo=True, pty=not WINDOWS)
    ctx.run("uv run python -m pneumonia.evaluate --config-name exp1", echo=True, pty=not WINDOWS)


@task
def exp1_train(ctx: Context) -> None:
    """Run experiment exp1"""
    ctx.run("uv run python -m pneumonia.train --config-name exp1", echo=True, pty=not WINDOWS)


@task
def exp1_test(ctx: Context) -> None:
    """Run experiment exp1"""
    ctx.run("uv run python -m pneumonia.evaluate --config-name exp1", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
