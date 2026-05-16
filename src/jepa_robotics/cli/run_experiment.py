"""Convenience pipeline runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.cli import collect_dataset, evaluate, plot, train_jepa, train_rl


def run(config: str, seed: int = 0, smoke_test: bool = False) -> Path:
    train_rl.run(config, seed=seed, smoke_test=smoke_test)
    collect_dataset.run(config, seed=seed, smoke_test=smoke_test)
    train_jepa.run(config, seed=seed, smoke_test=smoke_test)
    result = evaluate.run(config, seed=seed, smoke_test=smoke_test)
    plot.run(str(Path(result).parents[2]))
    return result


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, args.seed, args.smoke_test))


if __name__ == "__main__":
    main()
