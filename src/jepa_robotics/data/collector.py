"""Dataset collection routines."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.replay_buffer import TransitionBuffer
from jepa_robotics.data.storage import save_trajectories_npz
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.envs.vector_env import make_sb3_vec_env
from jepa_robotics.utils.provenance import git_commit
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
    path = out / "trajectories.npz"
    if config.dataset.skip_existing and path.exists():
        return path
    if policy_path is None and config.dataset.n_envs > 1:
        buffer = _collect_random_vectorized(config, seed)
    else:
        buffer = _collect_single_env(config, seed, policy_path)
    arrays = buffer.to_arrays()
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
            "git_commit": git_commit(Path.cwd()),
            "collector_n_envs": config.dataset.n_envs,
            "collector_vec_env_type": config.dataset.vec_env_type,
        },
    )
    dump_config(config, out / "config_resolved.yaml")
    return path


def _collect_single_env(
    config: BenchmarkConfig,
    seed: int,
    policy_path: str | Path | None = None,
) -> TransitionBuffer:
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
    env.close()
    return buffer


def _collect_random_vectorized(config: BenchmarkConfig, seed: int) -> TransitionBuffer:
    n_envs = max(1, min(config.dataset.n_envs, config.dataset.num_episodes))
    vec_env = make_sb3_vec_env(
        env_id=config.env.id,
        seed=seed,
        n_envs=n_envs,
        vec_env_type=config.dataset.vec_env_type,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.dataset.max_episode_steps,
    )
    action_space = vec_env.action_space
    action_space.seed(seed)
    vec_env.reset()
    buffer = TransitionBuffer()
    episode_ids = np.arange(n_envs, dtype=np.int64)
    timestep_ids = np.zeros(n_envs, dtype=np.int64)
    active = episode_ids < config.dataset.num_episodes
    episodes_started = n_envs
    episodes_completed = 0
    while episodes_completed < config.dataset.num_episodes:
        actions = np.stack([action_space.sample() for _ in range(n_envs)]).astype(np.float32)
        raw_next_obs, rewards, dones, infos = vec_env.step(actions)
        next_obs = cast(dict[str, np.ndarray], raw_next_obs)
        for env_index in range(n_envs):
            if not active[env_index]:
                continue
            final_obs = infos[env_index].get("terminal_observation")
            stored_obs = (
                final_obs
                if dones[env_index] and final_obs is not None
                else _slice_obs(next_obs, env_index)
            )
            buffer.append(
                stored_obs,
                actions[env_index],
                float(rewards[env_index]),
                bool(dones[env_index]),
                False,
                int(episode_ids[env_index]),
                int(timestep_ids[env_index]),
            )
            if dones[env_index]:
                episodes_completed += 1
                if episodes_started < config.dataset.num_episodes:
                    episode_ids[env_index] = episodes_started
                    timestep_ids[env_index] = 0
                    episodes_started += 1
                else:
                    active[env_index] = False
            else:
                timestep_ids[env_index] += 1
    vec_env.close()
    return buffer


def _slice_obs(obs: dict[str, np.ndarray], env_index: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[env_index]) for key, value in obs.items()}


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
