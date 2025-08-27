"""Deployment management commands."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Manage model deployments.")


@app.command()
def create(config: Path, environment: str) -> None:
    """Create a new deployment."""
    console.print(f"Created deployment from {config} to {environment}")


@app.command()
def update(deployment: str, config: Path) -> None:
    """Update an existing deployment."""
    console.print(f"Updated {deployment} using {config}")


@app.command()
def rollback(deployment: str, version: str) -> None:
    """Rollback to a previous deployment version."""
    console.print(f"Rolled back {deployment} to {version}")


@app.command()
def scale(deployment: str, replicas: int) -> None:
    """Scale a deployment."""
    console.print(f"Scaled {deployment} to {replicas} replicas")


@app.command()
def delete(deployment: str) -> None:
    """Delete a deployment."""
    console.print(f"Deleted deployment {deployment}")


@app.command()
def logs(deployment: str, lines: int = typer.Option(100, help="Number of log lines")) -> None:
    """View deployment logs."""
    console.print(f"Showing last {lines} lines for {deployment}")


@app.command()
def status(deployment: str) -> None:
    """Check deployment status."""
    console.print(f"Deployment {deployment} is running")
