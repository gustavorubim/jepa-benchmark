from __future__ import annotations

import random

import numpy as np
import torch

from jepa_robotics.envs.make_env import make_env
from jepa_robotics.utils.seed import seed_env, set_global_seed


def test_set_global_seed_reproducible() -> None:
    set_global_seed(123)
    values_a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_global_seed(123)
    values_b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert values_a == values_b


def test_seed_env_resets() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    obs_a, _ = seed_env(env, 42)
    obs_b, _ = seed_env(env, 42)
    np.testing.assert_allclose(obs_a["observation"], obs_b["observation"])
