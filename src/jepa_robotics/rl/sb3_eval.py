"""SB3 evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_sb3_model(path: str | Path) -> Any:  # pragma: no cover
    from stable_baselines3 import SAC

    return SAC.load(path)
