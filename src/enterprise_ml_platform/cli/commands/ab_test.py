"""CLI utilities for managing experiments."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from enterprise_ml_platform.services.ab_testing import (
    ExperimentConfig,
    ExperimentManager,
)

app = typer.Typer(help="Manage A/B testing experiments.")
_manager = ExperimentManager()


@app.command()
def create(config: Path) -> None:
    """Create an experiment from a JSON config file."""
    data = json.loads(config.read_text())
    cfg = ExperimentConfig(**data)
    asyncio.run(_manager.create_experiment(cfg))


@app.command()
def assign(name: str, session: str) -> None:
    """Get assigned variant for a session."""
    variant = asyncio.run(_manager.get_variant(name, session))
    typer.echo(variant)


@app.command("list")
def list_experiments() -> None:  # pragma: no cover - debug helper
    """List the registered experiments.

    Named ``list_experiments`` rather than ``list``: shadowing the builtin
    made the ``list(...)`` call below recurse into this command instead.
    """
    typer.echo(list(_manager._experiments.keys()))
