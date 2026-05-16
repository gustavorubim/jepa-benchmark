"""YAML config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from jepa_robotics.config.schema import BenchmarkConfig


def load_config(path: str | Path | None) -> BenchmarkConfig:
    if path is None:
        return BenchmarkConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        msg = f"Config at {config_path} must contain a YAML mapping."
        raise ValueError(msg)
    return BenchmarkConfig.model_validate(raw)


def dump_config(config: BenchmarkConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_yamlable(config.model_dump(mode="python")), handle, sort_keys=False)


def _to_yamlable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_yamlable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_yamlable(item) for item in value]
    return value
