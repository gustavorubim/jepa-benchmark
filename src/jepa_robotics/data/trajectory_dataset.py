"""PyTorch dataset for JEPA trajectory windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jepa_robotics.data.storage import TrajectoryArrays, load_trajectories_npz
from jepa_robotics.data.transforms import stack_state


class TrajectoryWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        arrays: TrajectoryArrays,
        horizon: int,
        indices: list[int] | np.ndarray | None = None,
    ) -> None:
        self.arrays = arrays
        self.horizon = horizon
        self.indices = (
            [int(index) for index in indices] if indices is not None else self._valid_indices()
        )
        if not self.indices:
            msg = f"No valid windows for horizon {self.horizon}; collect longer episodes."
            raise ValueError(msg)

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
        window_count = self.arrays.num_transitions - self.horizon
        if window_count <= 0:
            return []
        starts = np.arange(window_count, dtype=np.int64)
        episode_windows = np.stack(
            [self.arrays.episode_ids[starts + offset] for offset in range(self.horizon + 1)],
            axis=1,
        )
        same_episode = np.all(episode_windows == episode_windows[:, :1], axis=1)
        terminal_inside_window = np.zeros(window_count, dtype=bool)
        for offset in range(self.horizon):
            terminal_inside_window |= self.arrays.terminated[starts + offset]
            terminal_inside_window |= self.arrays.truncated[starts + offset]
        return starts[same_episode & ~terminal_inside_window].astype(int).tolist()


def split_window_dataset_by_episode(
    dataset: TrajectoryWindowDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[TrajectoryWindowDataset, TrajectoryWindowDataset]:
    """Split windows by episode id so train and validation episodes do not overlap."""
    episode_ids = np.unique(dataset.arrays.episode_ids[dataset.indices])
    if len(episode_ids) < 2:
        return dataset, dataset
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * validation_fraction)
    val_count = min(max(1, val_count), len(shuffled) - 1)
    val_episodes = {int(item) for item in shuffled[:val_count]}
    train_indices = [
        index
        for index in dataset.indices
        if int(dataset.arrays.episode_ids[index]) not in val_episodes
    ]
    val_indices = [
        index for index in dataset.indices if int(dataset.arrays.episode_ids[index]) in val_episodes
    ]
    return (
        TrajectoryWindowDataset(dataset.arrays, dataset.horizon, train_indices),
        TrajectoryWindowDataset(dataset.arrays, dataset.horizon, val_indices),
    )
