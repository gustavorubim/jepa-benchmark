from __future__ import annotations

from pathlib import Path

from jepa_robotics.plotting.generate import REQUIRED_PLOT_BASES, generate_report_artifacts


def test_generate_report_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    reports = generate_report_artifacts(tmp_path / "outputs" / "empty")
    for base in REQUIRED_PLOT_BASES:
        assert (reports / "plots" / f"{base}.png").exists()
        assert (reports / "plots" / f"{base}.pdf").exists()
    assert (reports / "tables" / "aggregate_metrics.csv").exists()
    assert (reports / "report.md").exists()
