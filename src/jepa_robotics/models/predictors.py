"""Action-conditioned latent predictors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_robotics.models.mlp import build_mlp


class ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 4,
        hidden_dims: list[int] | None = None,
        activation: str = "silu",
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.normalize_output = normalize_output
        self.net = build_mlp(
            input_dim=latent_dim + action_dim,
            hidden_dims=hidden_dims or [256, 256],
            output_dim=latent_dim,
            activation=activation,
            layer_norm=True,
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pred = self.net(torch.cat([z, action], dim=-1))
        if self.normalize_output:
            return F.normalize(pred, dim=-1)
        return pred

    def rollout(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds: list[torch.Tensor] = []
        current = z
        for step in range(actions.shape[1]):
            current = self(current, actions[:, step])
            preds.append(current)
        return torch.stack(preds, dim=1)
