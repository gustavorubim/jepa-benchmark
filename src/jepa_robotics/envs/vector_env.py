"""Vector environment helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from jepa_robotics.envs.make_env import make_env


def make_dummy_vec_env(env_id: str, seed: int, max_episode_steps: int | None = None) -> Any:
    return DummyVecEnv([lambda: make_env(env_id, seed, max_episode_steps=max_episode_steps)])


def make_env_thunk(
    env_id: str,
    seed: int,
    obs_mode: str = "state",
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
    monitor_file: str | None = None,
) -> Callable[[], Any]:
    """Build a picklable environment thunk for SB3 vector environments."""

    def _init() -> Any:
        env = make_env(
            env_id,
            seed=seed,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            max_episode_steps=max_episode_steps,
        )
        if monitor_file is not None:
            return Monitor(env, filename=monitor_file)
        return env

    return _init


def make_sb3_vec_env(
    env_id: str,
    seed: int,
    n_envs: int,
    vec_env_type: str,
    obs_mode: str = "state",
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
    monitor_dir: str | None = None,
) -> VecEnv:
    """Create an SB3 vector env using the repo's central environment factory."""
    env_count = max(1, n_envs)
    thunks = []
    for index in range(env_count):
        monitor_file = None
        if monitor_dir is not None:
            name = "monitor.csv" if env_count == 1 else f"monitor_{index}.csv"
            monitor_file = f"{monitor_dir}/{name}"
        thunks.append(
            make_env_thunk(
                env_id=env_id,
                seed=seed + index,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                max_episode_steps=max_episode_steps,
                monitor_file=monitor_file,
            )
        )
    if env_count > 1 and vec_env_type == "subproc":
        return SubprocVecEnv(thunks)
    return DummyVecEnv(thunks)
