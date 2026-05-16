"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def experiment_id_from_path(path: str | Path) -> str:
    return Path(path).name
