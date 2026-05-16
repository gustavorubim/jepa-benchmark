"""Versioned trajectory storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jepa_robotics.utils.serialization import write_json

SCHEMA_VERSION = "trajectory_npz_v1"


@dataclass(frozen=True)
class TrajectoryArrays:
    observations: np.ndarray
    achieved_goals: np.ndarray
    desired_goals: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    timestep_ids: np.ndarray

    @property
    def num_transitions(self) -> int:
        return int(self.actions.shape[0])


def save_trajectories_npz(
    path: str | Path,
    arrays: TrajectoryArrays,
    metadata: dict[str, Any],
    images: np.ndarray | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "observations": arrays.observations,
        "achieved_goals": arrays.achieved_goals,
        "desired_goals": arrays.desired_goals,
        "actions": arrays.actions,
        "rewards": arrays.rewards,
        "terminated": arrays.terminated,
        "truncated": arrays.truncated,
        "episode_ids": arrays.episode_ids,
        "timestep_ids": arrays.timestep_ids,
    }
    if images is not None:
        payload["images"] = images
    np.savez_compressed(output, **payload)  # type: ignore[arg-type]
    enriched = {
        "schema_version": SCHEMA_VERSION,
        **metadata,
        "num_transitions": arrays.num_transitions,
    }
    write_json(output.with_name("metadata.json"), enriched)


def load_trajectories_npz(path: str | Path) -> TrajectoryArrays:
    data = np.load(Path(path), allow_pickle=False)
    return TrajectoryArrays(
        observations=data["observations"].astype(np.float32),
        achieved_goals=data["achieved_goals"].astype(np.float32),
        desired_goals=data["desired_goals"].astype(np.float32),
        actions=data["actions"].astype(np.float32),
        rewards=data["rewards"].astype(np.float32),
        terminated=data["terminated"].astype(bool),
        truncated=data["truncated"].astype(bool),
        episode_ids=data["episode_ids"].astype(np.int64),
        timestep_ids=data["timestep_ids"].astype(np.int64),
    )
