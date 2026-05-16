"""MLP construction helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "silu",
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    activation_cls: type[nn.Module] = nn.SiLU if activation == "silu" else nn.ReLU
    for hidden in hidden_dims:
        layers.append(nn.Linear(current, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(activation_cls())
        current = hidden
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)
