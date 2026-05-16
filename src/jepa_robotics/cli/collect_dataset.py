"""Collect trajectory datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.config.load import load_config
from jepa_robotics.data.collector import collect_dataset


def run(
    config_path: str,
    policy: str | None = None,
    seed: int = 0,
    smoke_test: bool = False,
) -> Path:
    config = load_config(config_path)
    if smoke_test:
        config = config.smoke_copy()
    out = (
        Path(config.experiment.output_dir)
        / "datasets"
        / config.env.id
        / config.dataset.source
        / f"seed_{seed}"
    )
    result = collect_dataset(config, seed=seed, output_dir=out, policy_path=policy)
    (out / "stdout.log").write_text("collect_dataset completed\n", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.config, args.policy, args.seed, args.smoke_test))


if __name__ == "__main__":
    main()
