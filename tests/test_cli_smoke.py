from __future__ import annotations

from pathlib import Path

from jepa_robotics import run_suite as run_suite_module
from jepa_robotics import train_autoencoder as train_autoencoder_module
from jepa_robotics import train_jepa_policy as train_jepa_policy_module
from jepa_robotics.cli import (
    analyze,
    collect_dataset,
    evaluate,
    plot,
    run_experiment,
    run_suite,
    train_autoencoder,
    train_jepa,
    train_jepa_policy,
    train_rl,
)


def test_cli_smoke_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    phase_config = str(root / "configs" / "experiments" / "phase1_fetch_reach.yaml")
    jepa_config = str(root / "configs" / "jepa" / "state_jepa.yaml")
    mpc_config = str(root / "configs" / "jepa" / "jepa_mpc.yaml")

    rl_out = train_rl.run(phase_config, smoke_test=True)
    assert (rl_out / "model.zip").exists()
    dataset = collect_dataset.run(phase_config, policy=str(rl_out / "model.zip"), smoke_test=True)
    assert dataset.exists()
    jepa_out = train_jepa.run(jepa_config, dataset=str(dataset), smoke_test=True)
    assert (jepa_out / "encoder.pt").exists()
    train_metrics = jepa_out / "train_metrics.csv"
    val_metrics = jepa_out / "val_metrics.csv"
    assert "prediction_mse_h1" in train_metrics.read_text(encoding="utf-8")
    assert "val_loss" in val_metrics.read_text(encoding="utf-8")
    autoencoder_out = train_autoencoder.run(jepa_config, dataset=str(dataset), smoke_test=True)
    assert (autoencoder_out / "decoder.pt").exists()
    kwargs = train_jepa_policy.run(str(jepa_out / "encoder.pt"))
    assert "features_extractor_kwargs" in kwargs
    eval_out = evaluate.run(
        mpc_config,
        checkpoint=str(jepa_out / "encoder.pt"),
        dataset=str(dataset),
        smoke_test=True,
    )
    assert (eval_out / "metrics.csv").exists()
    reports = plot.run("outputs/smoke")
    assert (reports / "plots" / "sample_efficiency_success.png").exists()
    tables = analyze.run("outputs/smoke")
    assert (tables / "aggregate_metrics.csv").exists()


def test_run_experiment_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    result = run_experiment.run(
        str(root / "configs" / "experiments" / "phase1_fetch_reach.yaml"),
        smoke_test=True,
    )
    assert (result / "metrics.csv").exists()


def test_run_suite_smoke_skips_completed(tmp_path: Path) -> None:
    suite = run_suite.run(
        phases=["phase1_fetch_reach"],
        seeds=[0, 1],
        max_parallel=2,
        smoke_test=True,
        output_dir=tmp_path / "suite",
        command=["test"],
    )
    assert (suite / "suite_metadata.json").exists()
    rerun = run_suite.run(
        phases=["phase1_fetch_reach"],
        seeds=[0, 1],
        max_parallel=2,
        smoke_test=True,
        output_dir=tmp_path / "suite",
        command=["test"],
    )
    statuses = sorted((rerun / "suite_status").glob("*.json"))
    assert statuses
    assert all('"status": "skipped"' in path.read_text(encoding="utf-8") for path in statuses)


def test_compatibility_modules_export_run_functions() -> None:
    assert callable(run_suite_module.run)
    assert callable(train_autoencoder_module.run)
    assert callable(train_jepa_policy_module.run)
