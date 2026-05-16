"""Evaluate JEPA-MPC or baseline policies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jepa_robotics.config.load import dump_config, load_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.trajectory_dataset import TrajectoryWindowDataset
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.planning.latent_mpc import LatentMPC
from jepa_robotics.training.jepa_trainer import train_jepa_model
from jepa_robotics.utils.device import get_torch_device


def run(
    config_path: str,
    checkpoint: str | None = None,
    dataset: str | None = None,
    seed: int = 0,
    smoke_test: bool = False,
) -> Path:
    config = load_config(config_path)
    if smoke_test:
        config = config.smoke_copy()
    out = Path(config.experiment.output_dir) / "mpc" / "jepa_mpc" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dataset) if dataset else _resolve_dataset_path(config, seed)
    jepa_dir = (
        Path(checkpoint).parent
        if checkpoint
        else Path(config.experiment.output_dir) / "jepa" / "state_jepa" / f"seed_{seed}"
    )
    if not (jepa_dir / "encoder.pt").exists():
        if not dataset_path.exists():
            from jepa_robotics.data.collector import collect_dataset

            collect_dataset(config, seed=seed, output_dir=dataset_path.parent)
        train_jepa_model(config, dataset_path=dataset_path, seed=seed, output_dir=jepa_dir)
    model = _load_model(config, dataset_path, jepa_dir)
    env = make_env(config.env.id, seed=seed, max_episode_steps=config.env.max_episode_steps)
    device = get_torch_device(config.device.preferred)
    action_space = env.action_space
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise TypeError("LatentMPC requires a Box action space with low/high bounds.")
    controller = LatentMPC(
        model=model,
        action_low=np.asarray(action_space.low, dtype=np.float32),
        action_high=np.asarray(action_space.high, dtype=np.float32),
        device=device,
        planner=config.mpc.planner,
        horizon=config.mpc.horizon,
        num_candidates=config.mpc.num_candidates,
        num_elites=config.mpc.num_elites,
        iterations=config.mpc.iterations,
        lambda_action=config.mpc.lambda_action,
        seed=seed,
    )
    rows, diagnostics = _evaluate_mpc(env, controller, config, seed)
    summary = _mpc_summary(diagnostics)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "eval_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "mpc_diagnostics.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "mpc_summary.csv", index=False)
    dump_config(config, out / "config_resolved.yaml")
    (out / "stdout.log").write_text("evaluate completed\n", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    return out


def _load_model(config: BenchmarkConfig, dataset_path: Path, jepa_dir: Path) -> StateJEPA:
    dataset = TrajectoryWindowDataset.from_npz(dataset_path, horizon=config.jepa.max_horizon)
    device = get_torch_device(config.device.preferred)
    model = StateJEPA(
        input_dim=dataset.input_dim,
        action_dim=dataset.action_dim,
        latent_dim=config.jepa.latent_dim,
        hidden_dims=config.jepa.hidden_dims,
        predictor_hidden_dims=config.jepa.predictor_dims,
    ).to(device)
    model.online_encoder.load_state_dict(torch.load(jepa_dir / "encoder.pt", map_location=device))
    model.target_encoder.load_state_dict(
        torch.load(jepa_dir / "target_encoder.pt", map_location=device)
    )
    model.predictor.load_state_dict(torch.load(jepa_dir / "predictor.pt", map_location=device))
    return model


def _evaluate_mpc(
    env: Any,
    controller: LatentMPC,
    config: BenchmarkConfig,
    seed: int,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int]]]:
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    diagnostics: list[dict[str, float | int]] = []
    for episode in range(max(1, config.rl.n_eval_episodes)):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        last_success = 0.0
        for step in range(config.env.max_episode_steps):
            goal_distance_before = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            started = time.perf_counter()
            action = controller.act(obs, obs["desired_goal"])
            planning_time_ms = (time.perf_counter() - started) * 1000.0
            obs, reward, terminated, truncated, info = env.step(action)
            goal_distance_after = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            total += float(reward)
            last_success = float(info.get("is_success", 0.0))
            diagnostics.append(
                {
                    "episode": episode,
                    "step": step,
                    "planning_time_ms": planning_time_ms,
                    "chosen_action_norm": controller.last_diagnostics.selected_action_norm,
                    "selected_sequence_smoothness": (
                        controller.last_diagnostics.selected_sequence_smoothness
                    ),
                    "candidate_score_mean": controller.last_diagnostics.score_mean,
                    "candidate_score_std": controller.last_diagnostics.score_std,
                    "best_candidate_score": controller.last_diagnostics.best_candidate_score,
                    "predicted_goal_latent_distance": (
                        controller.last_diagnostics.latent_distance_to_goal
                    ),
                    "predicted_latent_progress": (
                        -controller.last_diagnostics.latent_distance_to_goal
                    ),
                    "actual_goal_distance_before_step": goal_distance_before,
                    "actual_goal_distance_after_step": goal_distance_after,
                    "actual_goal_progress": goal_distance_before - goal_distance_after,
                    "reward": float(reward),
                    "is_success": last_success,
                }
            )
            if terminated or truncated:
                lengths.append(step + 1)
                break
        rewards.append(total)
        successes.append(last_success)
    rows: list[dict[str, float | int | str]] = [
        {
            "global_step": 0,
            "seed": seed,
            "method": "jepa_mpc",
            "env_id": config.env.id,
            "mean_episode_reward": float(np.mean(rewards)),
            "std_episode_reward": float(np.std(rewards)),
            "mean_success_rate": float(np.mean(successes)),
            "std_success_rate": float(np.std(successes)),
            "mean_episode_length": float(np.mean(lengths or [config.env.max_episode_steps])),
        }
    ]
    return rows, diagnostics


def _mpc_summary(diagnostics: list[dict[str, float | int]]) -> dict[str, float]:
    if not diagnostics:
        return {"predicted_actual_progress_correlation": float("nan")}
    frame = pd.DataFrame(diagnostics)
    predicted_constant = frame["predicted_latent_progress"].nunique() < 2
    actual_constant = frame["actual_goal_progress"].nunique() < 2
    if predicted_constant or actual_constant:
        correlation = float("nan")
    else:
        correlation = float(frame["predicted_latent_progress"].corr(frame["actual_goal_progress"]))
    return {
        "predicted_actual_progress_correlation": correlation,
        "mean_action_norm": float(frame["chosen_action_norm"].mean()),
        "mean_sequence_smoothness": float(frame["selected_sequence_smoothness"].mean()),
        "mean_planning_time_ms": float(frame["planning_time_ms"].mean()),
    }


def _resolve_dataset_path(config: BenchmarkConfig, seed: int) -> Path:
    default = (
        Path(config.experiment.output_dir)
        / "datasets"
        / config.env.id
        / config.dataset.source
        / f"seed_{seed}"
        / "trajectories.npz"
    )
    if default.exists():
        return default
    matches = sorted(
        (Path(config.experiment.output_dir) / "datasets" / config.env.id).glob(
            f"*/seed_{seed}/trajectories.npz"
        )
    )
    return matches[0] if matches else default


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, args.checkpoint, args.dataset, args.seed, args.smoke_test))


if __name__ == "__main__":
    main()
