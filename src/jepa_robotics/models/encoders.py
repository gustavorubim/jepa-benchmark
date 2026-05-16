"""State encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_robotics.models.mlp import build_mlp


class StateEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        layer_norm: bool = True,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        self.net = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [256, 256],
            output_dim=latent_dim,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        if self.normalize_output:
            return F.normalize(z, dim=-1)
        return z
