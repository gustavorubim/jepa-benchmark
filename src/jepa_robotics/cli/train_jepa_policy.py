"""Prepare JEPA-pretrained policy settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_robotics.training.policy_trainer import jepa_feature_extractor_kwargs


def run(encoder_checkpoint: str, freeze_encoder: bool = True) -> dict[str, object]:
    return jepa_feature_extractor_kwargs(Path(encoder_checkpoint), freeze_encoder=freeze_encoder)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--fine-tune", action="store_true")
    args = parser.parse_args(argv)
    print(run(args.encoder_checkpoint, freeze_encoder=not args.fine_tune))


if __name__ == "__main__":
    main()
