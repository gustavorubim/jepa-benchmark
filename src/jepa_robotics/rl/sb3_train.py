"""Stable-Baselines3 training entry points."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.evaluation.rollouts import RandomPolicy, evaluate_policy_rollouts
from jepa_robotics.utils.seed import set_global_seed


def train_rl_baseline(
    config: BenchmarkConfig,
    method: str,
    seed: int,
    output_dir: str | Path,
    smoke_test: bool = False,
) -> Path:
    set_global_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if smoke_test or config.env.id == "ToyGoal-v0":
        return _write_smoke_rl_artifacts(config, method, seed, out)
    return _train_sb3(config, method, seed, out)


def _write_smoke_rl_artifacts(config: BenchmarkConfig, method: str, seed: int, out: Path) -> Path:
    env = make_env(config.env.id, seed=seed, max_episode_steps=config.env.max_episode_steps)
    policy = RandomPolicy(env.action_space, seed=seed)
    rows: list[dict[str, float | int | str]] = []
    for step in [0, max(config.rl.total_timesteps, 1)]:
        metrics = evaluate_policy_rollouts(env, policy, episodes=1, seed=seed + step)
        rows.append(
            {
                "global_step": step,
                "seed": seed,
                "method": method,
                "env_id": config.env.id,
                **metrics,
            }
        )
    pd.DataFrame(rows).to_csv(out / "eval_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "monitor.csv").write_text("r,l,t\n0,0,0\n", encoding="utf-8")
    with zipfile.ZipFile(out / "model.zip", "w") as archive:
        archive.writestr("policy.json", json.dumps({"policy_type": "random", "seed": seed}))
    dump_config(config, out / "config_resolved.yaml")
    (out / "training_stdout.log").write_text("smoke RL artifact generated\n", encoding="utf-8")
    return out


def _train_sb3(
    config: BenchmarkConfig, method: str, seed: int, out: Path
) -> Path:  # pragma: no cover
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(
                    config.env.id,
                    seed=seed,
                    obs_mode=config.env.obs_mode,
                    reward_mode=config.env.reward_mode,
                    max_episode_steps=config.env.max_episode_steps,
                ),
                filename=str(out / "monitor.csv"),
            )
        ]
    )
    algorithm = method.upper()
    if algorithm in {"SAC", "SAC_HER"}:
        kwargs: dict[str, Any] = {
            "policy": config.rl.policy,
            "env": env,
            "learning_rate": config.rl.learning_rate,
            "buffer_size": config.rl.buffer_size,
            "batch_size": config.rl.batch_size,
            "gamma": config.rl.gamma,
            "tau": config.rl.tau,
            "learning_starts": config.rl.learning_starts,
            "train_freq": config.rl.train_freq,
            "gradient_steps": config.rl.gradient_steps,
            "verbose": 0,
            "seed": seed,
        }
        if algorithm == "SAC_HER" or config.rl.replay_buffer_class == "HerReplayBuffer":
            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = config.rl.replay_buffer_kwargs
        model: Any = SAC(**kwargs)
    elif algorithm == "TD3":
        model = TD3(config.rl.policy, env, learning_rate=config.rl.learning_rate, seed=seed)
    else:
        raise ValueError(f"Unsupported RL method: {method}")
    callback = _MetricsEvalCallback(
        config=config,
        method=method,
        seed=seed,
        output_dir=out,
        eval_freq=config.rl.eval_freq,
    )
    assert isinstance(callback, BaseCallback)
    model.learn(total_timesteps=config.rl.total_timesteps, callback=callback, progress_bar=False)
    model.save(out / "model.zip")
    dump_config(config, out / "config_resolved.yaml")
    callback.write_metrics()
    return out


class _MetricsEvalCallback(BaseCallback):  # pragma: no cover - exercised in real runs
    def __init__(
        self,
        config: BenchmarkConfig,
        method: str,
        seed: int,
        output_dir: Path,
        eval_freq: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.method = method
        self.seed = seed
        self.output_dir = output_dir
        self.eval_freq = max(1, eval_freq)
        self.rows: list[dict[str, float | int | str]] = []
        self._last_eval_step = -1

    def _on_training_start(self) -> None:
        self._evaluate_and_store(0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._evaluate_and_store(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._evaluate_and_store(self.num_timesteps)

    def _evaluate_and_store(self, global_step: int) -> None:
        if global_step == self._last_eval_step:
            return
        row = _evaluate_sb3_model(
            model=self.model,
            config=self.config,
            method=self.method,
            seed=self.seed,
            global_step=global_step,
        )
        self.rows.append(row)
        self._last_eval_step = global_step
        self.write_metrics()

    def write_metrics(self) -> None:
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.output_dir / "eval_metrics.csv", index=False)
        frame.to_csv(self.output_dir / "metrics.csv", index=False)


def _evaluate_sb3_model(
    model: Any,
    config: BenchmarkConfig,
    method: str,
    seed: int,
    global_step: int,
) -> dict[str, float | int | str]:  # pragma: no cover - exercised in real runs
    env = make_env(
        config.env.id,
        seed=seed + 10_000 + global_step,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.env.max_episode_steps,
    )
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    for episode in range(config.rl.n_eval_episodes):
        obs, _ = env.reset(seed=seed + 10_000 + global_step + episode)
        total_reward = 0.0
        success = 0.0
        length = 0
        for step in range(config.env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = float(info.get("is_success", 0.0))
            length = step + 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        successes.append(success)
        lengths.append(length)
    env.close()
    return {
        "global_step": global_step,
        "seed": seed,
        "method": method,
        "env_id": config.env.id,
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_success_rate": float(np.mean(successes)),
        "std_success_rate": float(np.std(successes)),
        "mean_episode_length": float(np.mean(lengths)),
    }
