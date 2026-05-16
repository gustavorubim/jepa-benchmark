"""Vector environment helpers."""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.vec_env import DummyVecEnv

from jepa_robotics.envs.make_env import make_env


def make_dummy_vec_env(env_id: str, seed: int, max_episode_steps: int | None = None) -> Any:
    return DummyVecEnv([lambda: make_env(env_id, seed, max_episode_steps=max_episode_steps)])
