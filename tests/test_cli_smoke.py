from __future__ import annotations

from pathlib import Path

from jepa_robotics.cli import (
    analyze,
    collect_dataset,
    evaluate,
    plot,
    run_experiment,
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
