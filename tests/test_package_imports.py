"""Every module in the package must import.

A module that never gets imported by a test can hold anything -- including a
literal ``\\n`` written into the source, which is how
``security.audit.audit_logger`` shipped a file that could not be parsed at
all. Walking the package is the cheapest way to keep that from recurring.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil

import pytest

import enterprise_ml_platform

SOURCE_ROOT = pathlib.Path(enterprise_ml_platform.__file__).parent
MODULES = sorted(
    module.name
    for module in pkgutil.walk_packages(
        enterprise_ml_platform.__path__, enterprise_ml_platform.__name__ + "."
    )
)
SOURCE_FILES = sorted(SOURCE_ROOT.rglob("*.py"))


def test_the_package_has_modules_to_check() -> None:
    """Guard against the walk above silently finding nothing."""
    assert len(MODULES) > 100, f"only found {len(MODULES)} modules"


@pytest.mark.parametrize("module_name", MODULES, ids=lambda n: n.split(".", 1)[-1])
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_source_file_parses(path: pathlib.Path) -> None:
    """Catch syntax errors even in modules an import might skip."""
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize(
    "path", SOURCE_FILES, ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_module_has_a_real_docstring(path: pathlib.Path) -> None:
    """A string after ``from __future__`` is a no-op, not a docstring.

    Placed there, it leaves ``module.__doc__`` as None and every documentation
    tool reporting the module as undocumented.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if ast.get_docstring(tree):
        return
    stray = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert not stray, (
        f"{path.name} has a string at line {stray[0].lineno} that was meant to be "
        "the module docstring; move it above the imports"
    )
