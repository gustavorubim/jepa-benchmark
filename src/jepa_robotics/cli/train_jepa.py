"""Train JEPA world models."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.config.load import load_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.collector import collect_dataset
from jepa_robotics.training.jepa_trainer import train_jepa_model


def run(
    config_path: str,
    dataset: str | None = None,
    seed: int = 0,
    smoke_test: bool = False,
) -> Path:
    config = load_config(config_path)
    if smoke_test:
        config = config.smoke_copy()
    dataset_path = Path(dataset) if dataset else _default_dataset_path(config, seed)
    if not dataset_path.exists():
        collect_dataset(config, seed=seed, output_dir=dataset_path.parent)
    out = Path(config.experiment.output_dir) / "jepa" / "state_jepa" / f"seed_{seed}"
    result = train_jepa_model(config, dataset_path=dataset_path, seed=seed, output_dir=out)
    (result / "stdout.log").write_text("train_jepa completed\n", encoding="utf-8")
    (result / "stderr.log").write_text("", encoding="utf-8")
    return result


def _default_dataset_path(config: BenchmarkConfig, seed: int) -> Path:
    cfg = config
    return (
        Path(cfg.experiment.output_dir)
        / "datasets"
        / cfg.env.id
        / cfg.dataset.source
        / f"seed_{seed}"
        / "trajectories.npz"
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, args.dataset, args.seed, args.smoke_test))


if __name__ == "__main__":
    main()
