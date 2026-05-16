"""Linear probe helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def fit_linear_probe(latents: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    model = Ridge(alpha=1.0).fit(latents, targets)
    preds = model.predict(latents)
    return {
        "r2": float(r2_score(targets, preds)),
        "mse": float(mean_squared_error(targets, preds)),
        "mae": float(mean_absolute_error(targets, preds)),
    }
