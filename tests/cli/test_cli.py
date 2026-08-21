"""Tests for the CLI application."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from enterprise_ml_platform.cli.main import app

runner = CliRunner()


def test_pipeline_run(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("stages: []", encoding="utf-8")
    result = runner.invoke(app, ["pipeline", "run", "--config", str(config)])
    assert result.exit_code == 0
    assert "Pipeline run completed" in result.stdout


def test_model_train(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("params: {}", encoding="utf-8")
    result = runner.invoke(
        app,
        ["models", "train", "--algorithm", "xgboost", "--config", str(config)],
    )
    assert result.exit_code == 0
    assert "training completed" in result.stdout


def test_config_validate(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("key: value", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(config)])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.stdout
