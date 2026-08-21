"""Every entry point declared in ``pyproject.toml`` must resolve.

A console script or plugin pointing at a module that was never written only
fails once someone runs ``pip install .`` and then the command, which is
exactly the first thing a reader of the README does.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import shutil
import subprocess

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    tomllib = pytest.importorskip("tomli")

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared_entry_points() -> list[tuple[str, str, str]]:
    """Return ``(group, name, target)`` for every declared entry point."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = [
        ("console-scripts", n, t) for n, t in project.get("scripts", {}).items()
    ]
    for group, entries in project.get("entry-points", {}).items():
        declared += [(group, n, t) for n, t in entries.items()]
    return sorted(declared)


ENTRY_POINTS = _declared_entry_points()


def test_pyproject_declares_entry_points() -> None:
    """Guard against the parsing above silently finding nothing."""
    assert ENTRY_POINTS, "no entry points found in pyproject.toml"


@pytest.mark.parametrize(
    "group,name,target",
    ENTRY_POINTS,
    ids=[f"{group}:{name}" for group, name, _ in ENTRY_POINTS],
)
def test_entry_point_resolves(group: str, name: str, target: str) -> None:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    assert attr, f"{group}:{name} declares no attribute"
    assert hasattr(module, attr), f"{module_name} has no attribute {attr!r}"


def test_console_scripts_are_callable() -> None:
    """A console script target has to be something the shell can invoke."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    for name, target in project.get("scripts", {}).items():
        module_name, _, attr = target.partition(":")
        entry = getattr(importlib.import_module(module_name), attr)
        assert callable(entry), f"{name} -> {target} is not callable"


def test_console_scripts_take_no_required_arguments() -> None:
    """A console script is called with no arguments, so it must accept none.

    Regression: ``mlp`` pointed at a Typer *callback*, which takes a context
    Typer injects. It resolved and was callable, so a resolution check passed
    -- but running ``mlp`` raised ``TypeError: main() missing 1 required
    positional argument: 'ctx'``.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    for name, target in project.get("scripts", {}).items():
        module_name, _, attr = target.partition(":")
        entry = getattr(importlib.import_module(module_name), attr)
        required = [
            parameter
            for parameter in inspect.signature(entry).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        assert not required, (
            f"console script {name} -> {target} requires "
            f"{[p.name for p in required]}, but is invoked with no arguments"
        )


@pytest.mark.parametrize("script", ["mlp", "enterprise-ml"])
def test_installed_console_script_runs(script: str) -> None:
    """The only check that proves the shell command works end to end."""
    executable = shutil.which(script)
    if executable is None:
        pytest.skip(f"{script} is not installed; run `pip install -e .` first")

    result = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
