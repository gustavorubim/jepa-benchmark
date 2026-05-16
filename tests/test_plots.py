from __future__ import annotations

from pathlib import Path

import pandas as pd

from jepa_robotics.plotting.generate import REQUIRED_PLOT_BASES, generate_report_artifacts


def test_generate_report_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "suite" / "phase1_fetch_reach" / "rl" / "sac" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        """
experiment:
  name: phase1_fetch_reach
env:
  id: FetchReachDense-v4
  reward_mode: dense
rl:
  total_timesteps: 10
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "global_step": [0, 10],
            "seed": [0, 0],
            "method": ["sac", "sac"],
            "env_id": ["FetchReachDense-v4", "FetchReachDense-v4"],
            "mean_episode_reward": [-2.0, -1.0],
            "mean_success_rate": [0.0, 0.8],
        }
    ).to_csv(run_dir / "metrics.csv", index=False)
    reports = generate_report_artifacts(tmp_path / "outputs" / "suite")
    for base in REQUIRED_PLOT_BASES:
        assert (reports / "plots" / f"{base}.png").exists()
        assert (reports / "plots" / f"{base}.pdf").exists()
    assert (reports / "tables" / "aggregate_metrics.csv").exists()
    assert (reports / "report.md").exists()
    aggregate = pd.read_csv(reports / "tables" / "aggregate_metrics.csv")
    assert {"phase", "env_id", "method", "seed", "source_path"} - set(
        pd.read_csv(reports / "metrics_combined.csv").columns
    ) == set()
    assert aggregate.loc[0, "env_id"] == "FetchReachDense-v4"


def test_optional_plots_require_real_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reports = generate_report_artifacts(tmp_path / "outputs" / "empty")
    assert not (reports / "plots" / "probe_r2_bar.png").exists()
    manifest = pd.read_csv(reports / "artifact_manifest.csv")
    assert "missing" in set(manifest["status"])


def test_optional_plots_use_real_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "outputs" / "suite"
    jepa_dir = root / "state_jepa" / "jepa" / "state_jepa" / "seed_0"
    jepa_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": [0, 1],
            "global_step": [0, 1],
            "seed": [0, 0],
            "method": ["state_jepa", "state_jepa"],
            "split": ["train", "train"],
            "loss": [1.0, 0.5],
            "cosine_similarity_h1": [0.1, 0.2],
            "latent_std_mean": [0.8, 0.9],
            "prediction_mse_h1": [0.4, 0.2],
        }
    ).to_csv(jepa_dir / "train_metrics.csv", index=False)
    mpc_dir = root / "jepa_mpc" / "mpc" / "jepa_mpc" / "seed_0"
    mpc_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "global_step": [0],
            "seed": [0],
            "method": ["jepa_mpc"],
            "env_id": ["FetchReachDense-v4"],
            "mean_episode_reward": [-1.0],
            "mean_success_rate": [0.5],
        }
    ).to_csv(mpc_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "predicted_latent_progress": [0.1, 0.2],
            "actual_goal_progress": [0.0, 0.3],
            "planning_time_ms": [1.0, 2.0],
        }
    ).to_csv(mpc_dir / "mpc_diagnostics.csv", index=False)
    reports = generate_report_artifacts(root)
    manifest = pd.read_csv(reports / "artifact_manifest.csv")
    statuses = dict(zip(manifest["artifact"], manifest["status"], strict=False))
    assert statuses["jepa_loss_curves"] == "real"
    assert statuses["mpc_predicted_vs_actual_progress"] == "real"
