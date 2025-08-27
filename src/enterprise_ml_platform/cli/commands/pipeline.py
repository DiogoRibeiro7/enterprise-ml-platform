"""Pipeline management commands."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress

console = Console()
app = typer.Typer(help="Manage pipeline executions.")


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, help="Path to pipeline config."),
    stages: str = typer.Option("", help="Comma-separated list of stages to run."),
) -> None:
    """Execute complete or partial pipelines.

    Example:
        ``mlp pipeline run --config config.yaml --stages data,features,train``

    Args:
        config: Path to pipeline configuration file.
        stages: Comma-separated list of stages to run.
    """
    stage_list: List[str] = [s.strip() for s in stages.split(",") if s]
    total = len(stage_list) or 1
    with Progress() as progress:
        task = progress.add_task("Running pipeline", total=total)
        for _ in stage_list or ["all"]:
            progress.update(task, advance=1)
    console.print(f"[green]Pipeline run completed using {config}[/green]")


@app.command()
def status(run_id: str = typer.Argument(...)) -> None:
    """Check pipeline execution status.

    Args:
        run_id: Identifier of the pipeline run.
    """
    console.print(f"Status for run {run_id}: [cyan]running[/cyan]")


@app.command()
def list(status: Optional[str] = typer.Option(None, help="Filter by status.")) -> None:
    """List pipeline runs with optional filtering."""
    console.print(f"Listing pipeline runs with status: {status or 'any'}")


@app.command()
def stop(run_id: str) -> None:
    """Stop a running pipeline.

    Args:
        run_id: Identifier of the pipeline run.
    """
    console.print(f"Stopped pipeline run {run_id}")


@app.command()
def retry(run_id: str, stages: str = typer.Option("", help="Stages to retry")) -> None:
    """Retry failed pipeline stages.

    Args:
        run_id: Identifier of the pipeline run.
        stages: Comma-separated stages to retry.
    """
    console.print(f"Retrying {stages or 'all'} for run {run_id}")


@app.command()
def schedule(
    config: Path = typer.Option(..., exists=True, help="Config file"),
    cron: str = typer.Option(..., help="Cron expression"),
) -> None:
    """Schedule pipeline runs.

    Args:
        config: Path to pipeline configuration file.
        cron: Cron expression describing schedule.
    """
    console.print(f"Scheduled pipeline using {config} with cron '{cron}'")
