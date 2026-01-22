import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "pneumonia"
PYTHON_VERSION = "3.12"


# Project commands
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
def train(ctx: Context, config_name: str) -> None:
    """Run training with specified config."""
    ctx.run(f"uv run python -m pneumonia.train --config-name {config_name}", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, config_name: str) -> None:
    """Run evaluation with specified config."""
    ctx.run(f"uv run python -m pneumonia.evaluate --config-name {config_name}", echo=True, pty=not WINDOWS)


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
