"""Dataset collection routines."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.replay_buffer import TransitionBuffer
from jepa_robotics.data.storage import save_trajectories_npz
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.utils.seed import set_global_seed


def collect_dataset(
    config: BenchmarkConfig,
    seed: int,
    output_dir: str | Path,
    policy_path: str | Path | None = None,
) -> Path:
    set_global_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(
        config.env.id,
        seed=seed,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.dataset.max_episode_steps,
    )
    policy = _load_policy(policy_path)
    buffer = TransitionBuffer()
    for episode in range(config.dataset.num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        for timestep in range(config.dataset.max_episode_steps):
            action = _predict_action(policy, obs, env.action_space, seed + episode + timestep)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            buffer.append(next_obs, action, float(reward), terminated, truncated, episode, timestep)
            obs = next_obs
            if terminated or truncated:
                break
    arrays = buffer.to_arrays()
    path = out / "trajectories.npz"
    save_trajectories_npz(
        path,
        arrays,
        {
            "env_id": config.env.id,
            "seed": seed,
            "num_episodes": config.dataset.num_episodes,
            "source": config.dataset.source,
            "obs_mode": config.env.obs_mode,
            "created_by": "jepa_robotics",
            "git_commit": "unknown",
        },
    )
    dump_config(config, out / "config_resolved.yaml")
    return path


def _load_policy(policy_path: str | Path | None) -> Any:
    if policy_path is None:
        return None
    path = Path(policy_path)
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("policy.json").decode("utf-8"))
        if metadata.get("policy_type") == "random":
            return None
    except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
        pass
    try:  # pragma: no cover - exercised only in real SB3 runs
        from stable_baselines3 import SAC

        return SAC.load(path)
    except Exception:
        return None


def _predict_action(
    policy: Any, obs: dict[str, np.ndarray], action_space: Any, seed: int
) -> np.ndarray:
    if policy is not None:  # pragma: no cover - depends on SB3 artifact
        action, _ = policy.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    action_space.seed(seed)
    return np.asarray(action_space.sample(), dtype=np.float32)
