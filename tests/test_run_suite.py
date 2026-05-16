from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jepa_robotics.cli.run_suite import (
    SuiteChild,
    _expand_phase_presets,
    _outputs_complete,
    _phase_command,
    _phase_config_path,
    _run_child,
    run,
)


def test_phase_config_resolution_and_command_branches() -> None:
    assert _phase_config_path("phase1_fetch_reach", "iteration").exists()
    assert _phase_config_path("autoencoder", "iteration").name == "autoencoder.yaml"
    with pytest.raises(FileNotFoundError):
        _phase_config_path("missing_phase", "iteration")

    phases = [
        "phase1_fetch_reach",
        "phase2_fetch_push_dense",
        "phase3_fetch_push_dense",
        "phase3_fetch_push_sparse",
        "phase3_fetch_push_sparse_tqc",
        "state_jepa",
        "jepa_mpc",
        "autoencoder",
        "jepa_sac",
    ]
    for phase in phases:
        command, required = _phase_command(phase, 0, Path("config.yaml"), Path("out") / phase)
        assert command[0] == sys.executable
        assert required
    with pytest.raises(ValueError):
        _phase_command("unsupported", 0, Path("config.yaml"), Path("out"))


def test_stage1_fetch_reach_preset_expands_to_reach_pipeline() -> None:
    assert _expand_phase_presets(["stage1_fetch_reach"]) == [
        "phase1_fetch_reach",
        "state_jepa",
        "jepa_sac",
        "jepa_mpc",
    ]


def test_outputs_complete_and_run_child_status(tmp_path: Path) -> None:
    required = tmp_path / "done.txt"
    assert not _outputs_complete([required])
    required.write_text("ok", encoding="utf-8")
    assert _outputs_complete([required])

    child_done = tmp_path / "child_done.txt"
    child = SuiteChild(
        phase="phase",
        seed=0,
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path,
        command=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(child_done)!r}).write_text('ok')",
        ],
        required_outputs=[child_done],
        status_path=tmp_path / "status.json",
    )
    status = _run_child(child)
    assert status["status"] == "completed"
    assert (tmp_path / "status.json").exists()


def test_run_suite_defaults_to_single_seed(tmp_path: Path) -> None:
    suite = run(
        phases=["phase1_fetch_reach"],
        seeds=None,
        smoke_test=True,
        output_dir=tmp_path / "suite",
        command=["test"],
    )
    status_files = sorted((suite / "suite_status").glob("*.json"))
    assert [path.name for path in status_files] == ["phase1_fetch_reach_seed_0.json"]
