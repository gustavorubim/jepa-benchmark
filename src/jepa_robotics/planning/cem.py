"""Cross-entropy method action search."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cem_optimize(
    rng: np.random.Generator,
    score_fn: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    num_candidates: int,
    num_elites: int,
    iterations: int,
    init_std: float,
    min_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    action_dim = action_low.shape[0]
    mean = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.full((horizon, action_dim), init_std, dtype=np.float32)
    scores = np.zeros(num_candidates, dtype=np.float32)
    candidates = np.zeros((num_candidates, horizon, action_dim), dtype=np.float32)
    for _ in range(iterations):
        candidates = rng.normal(mean, std, size=candidates.shape).astype(np.float32)
        candidates = np.clip(candidates, action_low, action_high)
        scores = score_fn(candidates)
        elite_idx = np.argsort(scores)[-num_elites:]
        elites = candidates[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), min_std)
    best = candidates[int(np.argmax(scores))]
    return best, scores
