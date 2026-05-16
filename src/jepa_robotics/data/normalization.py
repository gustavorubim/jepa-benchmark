"""Array normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunningNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, data: np.ndarray, eps: float = 1e-6) -> RunningNormalizer:
        return cls(mean=data.mean(axis=0), std=data.std(axis=0) + eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std + self.mean).astype(np.float32)
