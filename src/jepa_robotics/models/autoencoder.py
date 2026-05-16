"""Reconstruction-based latent dynamics baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from jepa_robotics.models.encoders import StateEncoder
from jepa_robotics.models.mlp import build_mlp
from jepa_robotics.models.predictors import ActionConditionedPredictor


class AutoencoderDynamics(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            normalize_output=False,
        )
        self.decoder = build_mlp(latent_dim, hidden_dims or [256, 256], input_dim)
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            normalize_output=False,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(state)
        reconstruction = self.decoder(z)
        predicted_next = self.decoder(self.predictor(z, action))
        return reconstruction, predicted_next
