"""Generalization perturbation helpers."""

from __future__ import annotations

import numpy as np


def add_observation_noise(
    obs: dict[str, np.ndarray], scale: float, seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        key: value + rng.normal(0.0, scale, size=value.shape).astype(np.float32)
        for key, value in obs.items()
    }
