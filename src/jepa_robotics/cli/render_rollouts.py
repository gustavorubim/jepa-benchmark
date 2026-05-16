"""Render deterministic rollout videos for trained policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import SAC

from jepa_robotics.config.load import load_config
from jepa_robotics.data.storage import load_trajectories_npz
from jepa_robotics.data.trajectory_dataset import TrajectoryWindowDataset
from jepa_robotics.envs.make_env import make_env
from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.planning.latent_mpc import LatentMPC
from jepa_robotics.planning.scoring import make_goal_state_bank
from jepa_robotics.utils.device import get_torch_device


def run(
    experiment: str,
    methods: list[str],
    seed: int = 0,
    episodes: int = 3,
    fps: int = 10,
    budget: int | None = None,
    output_dir: str | None = None,
    overwrite: bool = False,
) -> Path:
    experiment_path = Path(experiment)
    if not output_dir:
        output_dir = str(experiment_path / "videos")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for method in methods:
        if method in {"sac", "sac_jepa"}:
            results.extend(
                _render_sac(
                    experiment=experiment_path,
                    method=method,
                    seed=seed,
                    episodes=episodes,
                    fps=fps,
                    output_root=output_root,
                    overwrite=overwrite,
                )
            )
        elif method == "jepa_mpc":
            results.extend(
                _render_jepa_mpc(
                    experiment=experiment_path,
                    seed=seed,
                    episodes=episodes,
                    fps=fps,
                    budget=budget,
                    output_root=output_root,
                    overwrite=overwrite,
                )
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    return output_root


def _render_sac(
    experiment: Path,
    method: str,
    seed: int,
    episodes: int,
    fps: int,
    output_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    policy_path = _latest_match(
        experiment,
        f"**/rl/{method}/seed_{seed}/model.zip",
    )
    config_path = policy_path.parent / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml for {method} at {policy_path}")
    config = load_config(config_path)

    env = make_env(
        config.env.id,
        seed=seed,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.env.max_episode_steps,
        render_mode="rgb_array",
    )
    model = SAC.load(policy_path)
    method_dir = output_root / method
    method_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    with torch.no_grad():
        for episode in range(max(1, episodes)):
            frames = _collect_sac_frames(
                model,
                env,
                seed=seed + episode,
                config=config,
            )
            path = method_dir / f"episode_{episode + 1:02d}.gif"
            if not overwrite and path.exists():
                continue
            _write_gif(frames=frames, out=path, fps=fps)
            outputs.append(path)

    env.close()
    return outputs


def _collect_sac_frames(
    model: SAC,
    env: Any,
    seed: int,
    config: Any,
) -> list[np.ndarray]:
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = [_to_rgb_frame(env)]
    for _ in range(config.env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _ = env.step(action)
        frames.append(_to_rgb_frame(env))
        if terminated or truncated:
            break
    return frames


def _render_jepa_mpc(
    experiment: Path,
    seed: int,
    episodes: int,
    fps: int,
    budget: int | None,
    output_root: Path,
    overwrite: bool = False,
) -> list[Path]:
    policy_parent = _resolve_jepa_mpc_path(experiment, seed=seed, budget=budget)
    encoder_path = policy_parent / "jepa" / "encoder.pt"
    config_path = policy_parent.parent / "config_resolved.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml for JEPA-MPC at {policy_parent}")
    config = load_config(config_path)
    trajectories_path = policy_parent / "trajectories.npz"
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Missing dataset for MPC budget at {trajectories_path}")

    arrays = load_trajectories_npz(trajectories_path)
    if arrays.episode_ids.size == 0:
        raise ValueError(f"No trajectories found for MPC render at {trajectories_path}")

    dataset = TrajectoryWindowDataset.from_npz(trajectories_path, horizon=config.jepa.max_horizon)
    model = _load_jepa_model(config, dataset, encoder_path.parent)
    env = make_env(
        config.env.id,
        seed=seed,
        obs_mode=config.env.obs_mode,
        reward_mode=config.env.reward_mode,
        max_episode_steps=config.env.max_episode_steps,
        render_mode="rgb_array",
    )
    action_space = env.action_space
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise TypeError("LatentMPC requires a Box action space with low/high bounds.")
    controller = LatentMPC(
        model=model,
        action_low=np.asarray(action_space.low, dtype=np.float32),
        action_high=np.asarray(action_space.high, dtype=np.float32),
        device=get_torch_device(config.device.preferred),
        planner=config.mpc.planner,
        horizon=config.mpc.horizon,
        num_candidates=config.mpc.num_candidates,
        num_elites=config.mpc.num_elites,
        iterations=config.mpc.iterations,
        lambda_action=config.mpc.lambda_action,
        seed=seed,
        goal_bank=make_goal_state_bank(arrays.observations, arrays.achieved_goals),
    )

    budget_label = policy_parent.name
    method_dir = output_root / "jepa_mpc" / budget_label
    method_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with torch.no_grad():
        for episode in range(max(1, episodes)):
            frames = _collect_mpc_frames(controller, env, seed + episode, config)
            path = method_dir / f"episode_{episode + 1:02d}.gif"
            if not overwrite and path.exists():
                continue
            _write_gif(frames=frames, out=path, fps=fps)
            outputs.append(path)
    env.close()
    return outputs


def _collect_mpc_frames(
    controller: LatentMPC,
    env: Any,
    seed: int,
    config: Any,
) -> list[np.ndarray]:
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = [_to_rgb_frame(env)]
    for _step in range(config.env.max_episode_steps):
        action = controller.act(obs, obs["desired_goal"])
        obs, _reward, terminated, truncated, _ = env.step(action)
        frames.append(_to_rgb_frame(env))
        if terminated or truncated:
            break
    return frames


def _resolve_jepa_mpc_path(experiment: Path, seed: int, budget: int | None) -> Path:
    if budget is not None:
        pattern = f"**/mpc/jepa_mpc/seed_{seed}/budget_{budget}"
        candidates = list(experiment.rglob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No matching JEPA-MPC budget={budget} under {experiment}")
        return _pick_newest_path(candidates)

    candidate_roots = list(experiment.rglob(f"**/mpc/jepa_mpc/seed_{seed}/budget_*"))
    if not candidate_roots:
        raise FileNotFoundError(f"No JEPA-MPC budgets found under {experiment}")
    return _pick_newest_path(candidate_roots)


def _pick_newest_path(paths: list[Path], key: Any | None = None) -> Path:
    if key is None:
        return max(paths, key=lambda p: p.stat().st_mtime)
    return max(paths, key=key)


def _load_jepa_model(config: Any, dataset: TrajectoryWindowDataset, jepa_dir: Path) -> StateJEPA:
    device = get_torch_device(config.device.preferred)
    model = StateJEPA(
        input_dim=dataset.input_dim,
        action_dim=dataset.action_dim,
        latent_dim=config.jepa.latent_dim,
        hidden_dims=config.jepa.hidden_dims,
        predictor_hidden_dims=config.jepa.predictor_dims,
    ).to(device)
    model.online_encoder.load_state_dict(torch.load(jepa_dir / "encoder.pt", map_location=device))
    model.target_encoder.load_state_dict(torch.load(jepa_dir / "target_encoder.pt", map_location=device))
    model.predictor.load_state_dict(torch.load(jepa_dir / "predictor.pt", map_location=device))
    return model


def _latest_match(experiment: Path, pattern: str) -> Path:
    matches = list(experiment.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files found under {experiment} for pattern {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _to_rgb_frame(env: Any) -> np.ndarray:
    frame = env.render()
    frame_array = np.asarray(frame, dtype=np.uint8)
    if frame_array.ndim == 2:
        frame_array = np.stack([frame_array] * 3, axis=-1)
    return frame_array


def _write_gif(frames: list[np.ndarray], out: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {out}")
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    frame_duration_ms = max(1, int(1000 / max(1, fps)))
    first, *rest = images
    first.save(
        out,
        save_all=True,
        append_images=rest,
        optimize=False,
        duration=frame_duration_ms,
        loop=0,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Render rollout GIFs from saved checkpoints.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["sac", "sac_jepa", "jepa_mpc"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(
        run(
            experiment=args.experiment,
            methods=args.methods,
            seed=args.seed,
            episodes=args.episodes,
            fps=args.fps,
            budget=args.budget,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
