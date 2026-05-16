"""Simple in-memory transition buffer for dataset collection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jepa_robotics.data.storage import TrajectoryArrays


@dataclass
class TransitionBuffer:
    observations: list[np.ndarray] = field(default_factory=list)
    achieved_goals: list[np.ndarray] = field(default_factory=list)
    desired_goals: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    timestep_ids: list[int] = field(default_factory=list)

    def append(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: int,
        timestep_id: int,
    ) -> None:
        self.observations.append(np.asarray(obs["observation"], dtype=np.float32))
        self.achieved_goals.append(np.asarray(obs["achieved_goal"], dtype=np.float32))
        self.desired_goals.append(np.asarray(obs["desired_goal"], dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.episode_ids.append(episode_id)
        self.timestep_ids.append(timestep_id)

    def to_arrays(self) -> TrajectoryArrays:
        return TrajectoryArrays(
            observations=np.stack(self.observations).astype(np.float32),
            achieved_goals=np.stack(self.achieved_goals).astype(np.float32),
            desired_goals=np.stack(self.desired_goals).astype(np.float32),
            actions=np.stack(self.actions).astype(np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            terminated=np.asarray(self.terminated, dtype=bool),
            truncated=np.asarray(self.truncated, dtype=bool),
            episode_ids=np.asarray(self.episode_ids, dtype=np.int64),
            timestep_ids=np.asarray(self.timestep_ids, dtype=np.int64),
        )
