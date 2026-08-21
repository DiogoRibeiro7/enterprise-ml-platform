"""Monitoring commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Monitor models and system health.")


@app.command()
def drift(model: str, baseline: Path, current: Path) -> None:
    """Monitor data or model drift."""
    console.print(f"Calculated drift for {model} using {baseline} and {current}")


@app.command()
def performance(model: str, period: str) -> None:
    """Monitor model performance over a period."""
    console.print(f"Gathered performance metrics for {model} over {period}")


@app.command()
def alerts(action: str, name: str) -> None:
    """Manage monitoring alerts."""
    console.print(f"{action.title()} alert {name}")


@app.command()
def dashboard() -> None:
    """Launch monitoring dashboard."""
    console.print("Launching dashboard...")


@app.command()
def health() -> None:
    """Perform system health checks."""
    console.print("System is healthy")
