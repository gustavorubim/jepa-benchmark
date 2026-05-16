"""Candidate action sequence sampling."""

from __future__ import annotations

import numpy as np


def sample_uniform_actions(
    rng: np.random.Generator,
    num_candidates: int,
    horizon: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    return rng.uniform(
        action_low, action_high, size=(num_candidates, horizon, action_low.shape[0])
    ).astype(np.float32)
