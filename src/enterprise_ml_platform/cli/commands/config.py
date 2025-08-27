"""Configuration management commands."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
import yaml

from enterprise_ml_platform.core.exceptions import ConfigurationError
from enterprise_ml_platform.utils.config_loader import load_config

console = Console()
app = typer.Typer(help="Manage application configuration.")


@app.command()
def validate(config: Path) -> None:
    """Validate a configuration file.

    Args:
        config: Path to the configuration file.
    """
    try:
        load_config(config)
    except ConfigurationError as exc:  # pragma: no cover - simple wrapper
        console.print(f"[red]Invalid configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]Configuration is valid[/green]")


@app.command()
def init(template: str = typer.Option("basic"), output: Path = typer.Option(Path("config.yaml"))) -> None:
    """Create a configuration file from a template.

    Args:
        template: Template name.
        output: Destination path for the configuration file.
    """
    config = {"template": template, "version": 1}
    output.write_text(yaml.safe_dump(config), encoding="utf-8")
    console.print(f"Wrote configuration template to {output}")
