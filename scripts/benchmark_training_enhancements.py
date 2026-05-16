"""Measure smoke-scale effects of the training performance enhancements."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jepa_robotics.cli.run_suite import run as run_suite
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.collector import collect_dataset
from jepa_robotics.data.storage import TrajectoryArrays
from jepa_robotics.data.trajectory_dataset import TrajectoryWindowDataset
from jepa_robotics.training.jepa_trainer import train_jepa_model


def main() -> None:
    report_dir = Path("reports") / "training_performance_enhancements"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path("outputs") / "training_performance_enhancements"
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = {
        "suite": _measure_suite(output_dir),
        "jepa_smoke": _measure_jepa_smoke(output_dir),
        "window_indexing": _measure_window_indexing(),
    }
    (report_dir / "measurement.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_markdown(report_dir / "measurement.md", measurements)
    print(report_dir / "measurement.md")


def _measure_suite(output_dir: Path) -> dict[str, Any]:
    suite_dir = output_dir / "suite_smoke"
    started = time.perf_counter()
    run_suite(
        phases=["phase1_fetch_reach"],
        seeds=[0, 1],
        max_parallel=2,
        smoke_test=True,
        force=True,
        output_dir=suite_dir,
        command=["benchmark_training_enhancements.py"],
    )
    first_run_seconds = time.perf_counter() - started
    started = time.perf_counter()
    run_suite(
        phases=["phase1_fetch_reach"],
        seeds=[0, 1],
        max_parallel=2,
        smoke_test=True,
        force=False,
        output_dir=suite_dir,
        command=["benchmark_training_enhancements.py"],
    )
    skip_seconds = time.perf_counter() - started
    statuses = [
        json.loads(path.read_text(encoding="utf-8"))["status"]
        for path in sorted((suite_dir / "suite_status").glob("*.json"))
    ]
    return {
        "suite_dir": str(suite_dir),
        "first_run_seconds": first_run_seconds,
        "skip_rerun_seconds": skip_seconds,
        "statuses_after_rerun": statuses,
    }


def _measure_jepa_smoke(output_dir: Path) -> dict[str, Any]:
    config = BenchmarkConfig().smoke_copy()
    dataset_path = collect_dataset(config, seed=0, output_dir=output_dir / "jepa_dataset")
    started = time.perf_counter()
    jepa_dir = train_jepa_model(
        config,
        dataset_path=dataset_path,
        seed=0,
        output_dir=output_dir / "jepa_smoke",
    )
    elapsed = time.perf_counter() - started
    train = pd.read_csv(jepa_dir / "train_metrics.csv")
    val = pd.read_csv(jepa_dir / "val_metrics.csv")
    return {
        "output_dir": str(jepa_dir),
        "elapsed_seconds": elapsed,
        "epochs_recorded": len(train),
        "final_train_loss": float(train.iloc[-1]["train_loss"]),
        "final_val_loss": float(val.iloc[-1]["val_loss"]),
        "has_real_validation_gap": bool(
            abs(float(train.iloc[-1]["train_loss"]) - float(val.iloc[-1]["val_loss"])) > 1e-9
        ),
        "horizon_metric_columns": [
            column for column in train.columns if column.startswith("prediction_mse_h")
        ],
    }


def _measure_window_indexing() -> dict[str, Any]:
    arrays = _synthetic_arrays(num_episodes=2000, episode_length=50)
    horizon = 8
    started = time.perf_counter()
    legacy = _legacy_valid_indices(arrays, horizon)
    legacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dataset = TrajectoryWindowDataset(arrays, horizon=horizon)
    vectorized_seconds = time.perf_counter() - started
    return {
        "transitions": arrays.num_transitions,
        "horizon": horizon,
        "legacy_seconds": legacy_seconds,
        "vectorized_seconds": vectorized_seconds,
        "speedup": legacy_seconds / max(vectorized_seconds, 1e-12),
        "legacy_valid_windows": len(legacy),
        "vectorized_valid_windows": len(dataset.indices),
    }


def _synthetic_arrays(num_episodes: int, episode_length: int) -> TrajectoryArrays:
    transitions = num_episodes * episode_length
    observations = np.zeros((transitions, 2), dtype=np.float32)
    achieved = np.zeros((transitions, 2), dtype=np.float32)
    desired = np.ones((transitions, 2), dtype=np.float32)
    actions = np.zeros((transitions, 2), dtype=np.float32)
    rewards = np.zeros(transitions, dtype=np.float32)
    terminated = np.zeros(transitions, dtype=bool)
    truncated = np.zeros(transitions, dtype=bool)
    episode_ids = np.repeat(np.arange(num_episodes), episode_length)
    timestep_ids = np.tile(np.arange(episode_length), num_episodes)
    terminal_indices = np.arange(episode_length - 1, transitions, episode_length)
    truncated[terminal_indices] = True
    return TrajectoryArrays(
        observations=observations,
        achieved_goals=achieved,
        desired_goals=desired,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        episode_ids=episode_ids,
        timestep_ids=timestep_ids,
    )


def _legacy_valid_indices(arrays: TrajectoryArrays, horizon: int) -> list[int]:
    valid: list[int] = []
    for idx in range(0, arrays.num_transitions - horizon):
        same_episode = np.all(
            arrays.episode_ids[idx : idx + horizon + 1] == arrays.episode_ids[idx]
        )
        if same_episode:
            valid.append(idx)
    return valid


def _write_markdown(path: Path, measurements: dict[str, Any]) -> None:
    suite = measurements["suite"]
    jepa = measurements["jepa_smoke"]
    windows = measurements["window_indexing"]
    text = f"""# Training Performance Enhancement Measurements

These measurements are smoke-scale checks on this machine. They verify that the new
orchestration, idempotency, JEPA validation, and vectorized window indexing paths work and
are measurable without running long Fetch experiments.

| Check | Result |
| --- | ---: |
| Suite first run, 2 smoke seeds | {suite["first_run_seconds"]:.3f} s |
| Suite rerun skip overhead | {suite["skip_rerun_seconds"]:.3f} s |
| JEPA smoke training | {jepa["elapsed_seconds"]:.3f} s |
| JEPA final train loss | {jepa["final_train_loss"]:.6f} |
| JEPA final validation loss | {jepa["final_val_loss"]:.6f} |
| Real validation gap present | {jepa["has_real_validation_gap"]} |
| Window indexing legacy loop | {windows["legacy_seconds"]:.6f} s |
| Window indexing vectorized | {windows["vectorized_seconds"]:.6f} s |
| Window indexing speedup | {windows["speedup"]:.2f}x |

Suite statuses after rerun: `{suite["statuses_after_rerun"]}`.

JEPA horizon metric columns: `{jepa["horizon_metric_columns"]}`.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
