"""Typer-based command line interface for the Enterprise ML Platform."""
from __future__ import annotations

import typer
from rich.console import Console

from enterprise_ml_platform.cli.commands import (
    config as config_cmd,
    data as data_cmd,
    deploy as deploy_cmd,
    models as models_cmd,
    monitor as monitor_cmd,
    pipeline as pipeline_cmd,
)

console = Console()
app = typer.Typer(help="Command-line interface for the Enterprise ML Platform.")
app.add_typer(pipeline_cmd.app, name="pipeline")
app.add_typer(data_cmd.app, name="data")
app.add_typer(models_cmd.app, name="models")
app.add_typer(deploy_cmd.app, name="deploy")
app.add_typer(monitor_cmd.app, name="monitor")
app.add_typer(config_cmd.app, name="config")


def version_callback(value: bool) -> None:
    """Print version information and exit.

    Args:
        value: Whether the user requested the version.
    """
    if value:
        from importlib.metadata import version

        console.print(f"enterprise-ml-platform v{version('enterprise-ml-platform')}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the application's version and exit.",
    ),
) -> None:
    """Entry point for the ``mlp`` command.

    Args:
        ctx: Typer context object.
        version: Display version information.
    """


if __name__ == "__main__":  # pragma: no cover
    app()
