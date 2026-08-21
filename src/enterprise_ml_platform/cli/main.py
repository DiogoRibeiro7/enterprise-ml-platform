"""Typer-based command line interface for the Enterprise ML Platform."""

from __future__ import annotations

import typer
from rich.console import Console

from enterprise_ml_platform.cli.commands import (
    ab_test as ab_test_cmd,
)
from enterprise_ml_platform.cli.commands import (
    config as config_cmd,
)
from enterprise_ml_platform.cli.commands import (
    data as data_cmd,
)
from enterprise_ml_platform.cli.commands import (
    deploy as deploy_cmd,
)
from enterprise_ml_platform.cli.commands import (
    models as models_cmd,
)
from enterprise_ml_platform.cli.commands import (
    monitor as monitor_cmd,
)
from enterprise_ml_platform.cli.commands import (
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
app.add_typer(ab_test_cmd.app, name="abtest")


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


def run() -> None:
    """Console script entry point.

    The ``main`` below is a Typer *callback*: it takes a context Typer builds
    and injects, so calling it as a console script raises ``TypeError``. The
    Typer app itself is what has to be invoked.
    """
    app()


if __name__ == "__main__":  # pragma: no cover
    run()
