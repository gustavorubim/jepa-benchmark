"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env: Any, seed: int) -> Any:
    return env.reset(seed=seed)
