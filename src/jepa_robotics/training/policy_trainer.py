"""JEPA-pretrained policy configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jepa_robotics.models.policies import JepaFeatureExtractor


def jepa_feature_extractor_kwargs(
    encoder_checkpoint: str | Path,
    freeze_encoder: bool,
    include_desired_goal: bool = False,
    append_desired_goal: bool = True,
    features_dim: int = 128,
    hidden_dims: list[int] | None = None,
    activation: str = "silu",
    layer_norm: bool = True,
    normalize_output: bool = True,
) -> dict[str, Any]:
    return {
        "features_extractor_class": JepaFeatureExtractor,
        "features_extractor_kwargs": {
            "encoder_checkpoint": str(encoder_checkpoint),
            "freeze_encoder": freeze_encoder,
            "include_desired_goal": include_desired_goal,
            "append_desired_goal": append_desired_goal,
            "features_dim": features_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "layer_norm": layer_norm,
            "normalize_output": normalize_output,
        },
    }
