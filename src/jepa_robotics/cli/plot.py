"""Generate plots from saved experiment logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.plotting.generate import generate_report_artifacts


def run(experiment: str) -> Path:
    return generate_report_artifacts(experiment)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    print(run(args.experiment))


if __name__ == "__main__":
    main()
