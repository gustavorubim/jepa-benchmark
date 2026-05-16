"""Train traditional RL baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.config.load import load_config
from jepa_robotics.rl.sb3_train import train_rl_baseline


def run(config_path: str, method: str = "sac", seed: int = 0, smoke_test: bool = False) -> Path:
    config = load_config(config_path)
    if smoke_test:
        config = config.smoke_copy()
    out = Path(config.experiment.output_dir) / "rl" / method / f"seed_{seed}"
    result = train_rl_baseline(
        config, method=method, seed=seed, output_dir=out, smoke_test=smoke_test
    )
    (result / "stdout.log").write_text("train_rl completed\n", encoding="utf-8")
    (result / "stderr.log").write_text("", encoding="utf-8")
    return result


def run_with_options(
    config_path: str,
    method: str = "sac",
    seed: int = 0,
    smoke_test: bool = False,
    feature_extractor: str | None = None,
    encoder_checkpoint: str | None = None,
    freeze_encoder: bool = True,
    force: bool = False,
) -> Path:
    config = load_config(config_path)
    if smoke_test:
        config = config.smoke_copy()
    out = Path(config.experiment.output_dir) / "rl" / method / f"seed_{seed}"
    result = train_rl_baseline(
        config,
        method=method,
        seed=seed,
        output_dir=out,
        smoke_test=smoke_test,
        feature_extractor=feature_extractor,
        encoder_checkpoint=encoder_checkpoint,
        freeze_encoder=freeze_encoder,
        force=force,
    )
    (result / "stdout.log").write_text("train_rl completed\n", encoding="utf-8")
    (result / "stderr.log").write_text("", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--feature-extractor", choices=["jepa"], default=None)
    parser.add_argument("--encoder-checkpoint", default=None)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--fine-tune-encoder", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(
        run_with_options(
            args.config,
            args.method,
            args.seed,
            args.smoke_test,
            feature_extractor=args.feature_extractor,
            encoder_checkpoint=args.encoder_checkpoint,
            freeze_encoder=not args.fine_tune_encoder,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
