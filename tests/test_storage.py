from __future__ import annotations

from pathlib import Path

import numpy as np

from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.collector import collect_dataset
from jepa_robotics.data.normalization import RunningNormalizer
from jepa_robotics.data.replay_buffer import TransitionBuffer
from jepa_robotics.data.storage import load_trajectories_npz, save_trajectories_npz
from jepa_robotics.data.trajectory_dataset import TrajectoryWindowDataset
from jepa_robotics.data.transforms import stack_state_with_goal


def test_replay_storage_and_dataset(tmp_path: Path) -> None:
    buffer = TransitionBuffer()
    for step in range(3):
        obs = {
            "observation": np.asarray([step, step + 1], dtype=np.float32),
            "achieved_goal": np.asarray([step, step + 1], dtype=np.float32),
            "desired_goal": np.ones(2, dtype=np.float32),
        }
        buffer.append(obs, np.ones(2, dtype=np.float32), -1.0, False, step == 2, 0, step)
    arrays = buffer.to_arrays()
    path = tmp_path / "trajectories.npz"
    save_trajectories_npz(path, arrays, {"env_id": "ToyGoal-v0", "seed": 0})
    loaded = load_trajectories_npz(path)
    assert loaded.num_transitions == 3
    dataset = TrajectoryWindowDataset(loaded, horizon=1)
    sample = dataset[0]
    assert sample["states"].shape == (2, 4)
    assert sample["actions"].shape == (1, 2)


def test_normalizer_and_collector(tmp_path: Path) -> None:
    config = BenchmarkConfig().smoke_copy()
    out = tmp_path / "dataset"
    path = collect_dataset(config, seed=0, output_dir=out)
    arrays = load_trajectories_npz(path)
    normalizer = RunningNormalizer.fit(arrays.observations)
    transformed = normalizer.transform(arrays.observations)
    restored = normalizer.inverse_transform(transformed)
    assert restored.shape == arrays.observations.shape
    assert (out / "metadata.json").exists()
    assert (
        stack_state_with_goal(
            arrays.observations[0], arrays.achieved_goals[0], arrays.desired_goals[0]
        ).shape[0]
        == 6
    )
