"""Trajectory transform helpers."""

from __future__ import annotations

import numpy as np


def stack_state(observation: np.ndarray, achieved_goal: np.ndarray) -> np.ndarray:
    return np.concatenate([observation.ravel(), achieved_goal.ravel()]).astype(np.float32)


def stack_state_with_goal(
    observation: np.ndarray,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [observation.ravel(), achieved_goal.ravel(), desired_goal.ravel()]
    ).astype(np.float32)
