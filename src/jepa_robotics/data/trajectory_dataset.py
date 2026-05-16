"""PyTorch dataset for JEPA trajectory windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa_robotics.data.storage import TrajectoryArrays, load_trajectories_npz
from jepa_robotics.data.transforms import stack_state


class TrajectoryWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: TrajectoryArrays, horizon: int) -> None:
        self.arrays = arrays
        self.horizon = horizon
        self.indices = self._valid_indices()

    @classmethod
    def from_npz(cls, path: str | Path, horizon: int) -> TrajectoryWindowDataset:
        return cls(load_trajectories_npz(path), horizon=horizon)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.indices[index]
        states = [
            stack_state(
                self.arrays.observations[start + offset], self.arrays.achieved_goals[start + offset]
            )
            for offset in range(self.horizon + 1)
        ]
        actions = [self.arrays.actions[start + offset] for offset in range(self.horizon)]
        return {
            "states": torch.as_tensor(np.stack(states), dtype=torch.float32),
            "actions": torch.as_tensor(np.stack(actions), dtype=torch.float32),
            "future_states": torch.as_tensor(np.stack(states[1:]), dtype=torch.float32),
        }

    @property
    def input_dim(self) -> int:
        first = stack_state(self.arrays.observations[0], self.arrays.achieved_goals[0])
        return int(first.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.arrays.actions.shape[-1])

    def _valid_indices(self) -> list[int]:
        valid: list[int] = []
        for idx in range(0, self.arrays.num_transitions - self.horizon):
            same_episode = np.all(
                self.arrays.episode_ids[idx : idx + self.horizon + 1]
                == self.arrays.episode_ids[idx]
            )
            if same_episode:
                valid.append(idx)
        if not valid:
            msg = f"No valid windows for horizon {self.horizon}; collect longer episodes."
            raise ValueError(msg)
        return valid
