"""Generate aggregate tables from saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.plotting.generate import load_experiment_metrics
from jepa_robotics.plotting.tables import write_analysis_tables


def run(experiment: str) -> Path:
    experiment_path = Path(experiment)
    reports_dir = Path("reports") / experiment_path.name
    metrics = load_experiment_metrics(experiment_path)
    write_analysis_tables(metrics, reports_dir)
    return reports_dir / "tables"


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    print(run(args.experiment))


if __name__ == "__main__":
    main()
