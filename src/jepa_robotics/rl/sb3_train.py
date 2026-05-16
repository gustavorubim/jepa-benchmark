"""Stable-Baselines3 training entry points."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.envs.vector_env import make_sb3_vec_env
from jepa_robotics.evaluation.rollouts import RandomPolicy, evaluate_policy_rollouts
from jepa_robotics.rl.callbacks import EarlyStopState
from jepa_robotics.training.policy_trainer import jepa_feature_extractor_kwargs
from jepa_robotics.utils.seed import set_global_seed


def train_rl_baseline(
    config: BenchmarkConfig,
    method: str,
    seed: int,
    output_dir: str | Path,
    smoke_test: bool = False,
    feature_extractor: str | None = None,
    encoder_checkpoint: str | Path | None = None,
    freeze_encoder: bool = True,
    force: bool = False,
) -> Path:
    set_global_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not force and config.rl.skip_existing and _rl_outputs_complete(out):
        return out
    if smoke_test or config.env.id == "ToyGoal-v0":
        return _write_smoke_rl_artifacts(config, method, seed, out)
    return _train_sb3(
        config,
        method,
        seed,
        out,
        feature_extractor=feature_extractor,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
    )


def _rl_outputs_complete(out: Path) -> bool:
    required = ["model.zip", "metrics.csv", "config_resolved.yaml"]
    return all((out / name).exists() for name in required)


def _write_smoke_rl_artifacts(config: BenchmarkConfig, method: str, seed: int, out: Path) -> Path:
    started = time.perf_counter()
    env = make_env(config.env.id, seed=seed, max_episode_steps=config.env.max_episode_steps)
    policy = RandomPolicy(env.action_space, seed=seed)
    rows: list[dict[str, float | int | str]] = []
    for step in [0, max(config.rl.total_timesteps, 1)]:
        eval_started = time.perf_counter()
        metrics = evaluate_policy_rollouts(env, policy, episodes=1, seed=seed + step)
        eval_seconds = time.perf_counter() - eval_started
        rows.append(
            {
                "global_step": step,
                "seed": seed,
                "method": method,
                "env_id": config.env.id,
                "eval_seconds": eval_seconds,
                "total_runtime_seconds": time.perf_counter() - started,
                "steps_per_second": float(step / max(time.perf_counter() - started, 1e-9)),
                "early_stop_triggered": 0,
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
    config: BenchmarkConfig,
    method: str,
    seed: int,
    out: Path,
    feature_extractor: str | None = None,
    encoder_checkpoint: str | Path | None = None,
    freeze_encoder: bool = True,
) -> Path:  # pragma: no cover
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    env = make_sb3_vec_env(
        env_id=config.env.id,
        seed=seed,
        n_envs=config.rl.n_envs,
        vec_env_type=config.rl.vec_env_type,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.env.max_episode_steps,
        monitor_dir=str(out),
        time_feature_wrapper=config.rl.time_feature_wrapper,
    )
    algorithm = method.upper()
    sb3_algorithm = "SAC" if algorithm in {"SAC", "SAC_HER", "SAC_JEPA"} else algorithm
    sb3_algorithm = "TQC" if algorithm in {"TQC", "TQC_HER"} else sb3_algorithm
    if sb3_algorithm == "SAC":
        kwargs = _off_policy_kwargs(
            config=config,
            env=env,
            seed=seed,
            feature_extractor=feature_extractor,
            encoder_checkpoint=encoder_checkpoint,
            freeze_encoder=freeze_encoder,
        )
        if _uses_her(algorithm, config):
            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = config.rl.replay_buffer_kwargs
        model: Any = SAC(**kwargs)
    elif sb3_algorithm == "TQC":
        from sb3_contrib import TQC

        kwargs = _off_policy_kwargs(
            config=config,
            env=env,
            seed=seed,
            feature_extractor=feature_extractor,
            encoder_checkpoint=encoder_checkpoint,
            freeze_encoder=freeze_encoder,
        )
        kwargs["top_quantiles_to_drop_per_net"] = config.rl.top_quantiles_to_drop_per_net
        if _uses_her(algorithm, config):
            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = config.rl.replay_buffer_kwargs
        model = TQC(**kwargs)
    elif sb3_algorithm == "TD3":
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
    started = time.perf_counter()
    model.learn(total_timesteps=config.rl.total_timesteps, callback=callback, progress_bar=False)
    model.save(out / "model.zip")
    env.close()
    dump_config(config, out / "config_resolved.yaml")
    callback.total_runtime_seconds = time.perf_counter() - started
    callback.write_metrics()
    return out


def _policy_kwargs(
    config: BenchmarkConfig,
    feature_extractor: str | None,
    encoder_checkpoint: str | Path | None,
    freeze_encoder: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config.rl.policy_net_arch:
        kwargs["net_arch"] = config.rl.policy_net_arch
    if config.rl.n_critics is not None:
        kwargs["n_critics"] = config.rl.n_critics
    if feature_extractor is None:
        return kwargs
    if feature_extractor != "jepa":
        raise ValueError(f"Unsupported feature extractor: {feature_extractor}")
    if encoder_checkpoint is None:
        raise ValueError("--encoder-checkpoint is required when --feature-extractor jepa is used.")
    kwargs.update(
        jepa_feature_extractor_kwargs(
            encoder_checkpoint=encoder_checkpoint,
            freeze_encoder=freeze_encoder,
            include_desired_goal=False,
            features_dim=config.jepa.latent_dim,
            hidden_dims=config.jepa.hidden_dims,
            activation=config.jepa.activation,
            layer_norm=config.jepa.layer_norm,
            normalize_output=config.jepa.normalize_latents,
        )
    )
    return kwargs


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
        self._started = time.perf_counter()
        self.total_runtime_seconds = 0.0
        self.eval_env = make_env(
            config.env.id,
            seed=seed + 10_000,
            obs_mode=config.env.obs_mode,
            reward_mode=config.env.reward_mode,
            max_episode_steps=config.env.max_episode_steps,
            time_feature_wrapper=config.rl.time_feature_wrapper,
        )
        self.early_stop = EarlyStopState(
            threshold=config.rl.early_stop_success,
            patience=config.rl.early_stop_patience,
        )

    def _on_training_start(self) -> None:
        self._evaluate_and_store(0)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            row = self._evaluate_and_store(
                self.num_timesteps,
                episodes=max(1, self.config.rl.train_eval_episodes),
                phase="train",
            )
            if row is not None and self.early_stop.observe(float(row["mean_success_rate"])):
                row["early_stop_triggered"] = 1
                self.write_metrics()
                return False
        return True

    def _on_training_end(self) -> None:
        self._evaluate_and_store(
            self.num_timesteps,
            episodes=max(1, self.config.rl.final_eval_episodes),
            phase="final",
            force=True,
        )
        self.eval_env.close()

    def _evaluate_and_store(
        self,
        global_step: int,
        episodes: int | None = None,
        phase: str = "train",
        force: bool = False,
    ) -> dict[str, float | int | str] | None:
        if global_step == self._last_eval_step and not force:
            return None
        eval_started = time.perf_counter()
        row = _evaluate_sb3_model(
            model=self.model,
            env=self.eval_env,
            config=self.config,
            method=self.method,
            seed=self.seed,
            global_step=global_step,
            episodes=episodes or max(1, self.config.rl.train_eval_episodes),
        )
        eval_seconds = time.perf_counter() - eval_started
        elapsed = time.perf_counter() - self._started
        row["eval_phase"] = phase
        row["eval_seconds"] = eval_seconds
        row["total_runtime_seconds"] = elapsed
        row["steps_per_second"] = float(global_step / max(elapsed, 1e-9))
        row["early_stop_triggered"] = int(self.early_stop.should_stop)
        row["n_envs"] = self.config.rl.n_envs
        self.rows.append(row)
        self._last_eval_step = global_step
        self.write_metrics()
        return row

    def write_metrics(self) -> None:
        frame = pd.DataFrame(self.rows)
        if not frame.empty and self.total_runtime_seconds > 0:
            frame.loc[frame.index[-1], "total_runtime_seconds"] = self.total_runtime_seconds
            frame.loc[frame.index[-1], "steps_per_second"] = float(
                frame.iloc[-1]["global_step"] / max(self.total_runtime_seconds, 1e-9)
            )
        frame.to_csv(self.output_dir / "eval_metrics.csv", index=False)
        frame.to_csv(self.output_dir / "metrics.csv", index=False)


def _uses_her(algorithm: str, config: BenchmarkConfig) -> bool:
    return algorithm in {"SAC_HER", "TQC_HER"} or config.rl.replay_buffer_class == "HerReplayBuffer"


def _off_policy_kwargs(
    config: BenchmarkConfig,
    env: Any,
    seed: int,
    feature_extractor: str | None,
    encoder_checkpoint: str | Path | None,
    freeze_encoder: bool,
) -> dict[str, Any]:
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
        "ent_coef": config.rl.ent_coef,
        "use_sde": config.rl.use_sde,
        "sde_sample_freq": config.rl.sde_sample_freq,
        "use_sde_at_warmup": config.rl.use_sde_at_warmup,
        "verbose": 0,
        "seed": seed,
    }
    policy_kwargs = _policy_kwargs(
        config=config,
        feature_extractor=feature_extractor,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
    )
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs
    return kwargs


def _evaluate_sb3_model(
    model: Any,
    env: Any,
    config: BenchmarkConfig,
    method: str,
    seed: int,
    global_step: int,
    episodes: int,
) -> dict[str, float | int | str]:  # pragma: no cover - exercised in real runs
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    for episode in range(episodes):
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
