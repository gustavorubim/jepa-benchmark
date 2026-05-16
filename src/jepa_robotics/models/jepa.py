"""State JEPA model."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from jepa_robotics.models.encoders import StateEncoder
from jepa_robotics.models.predictors import ActionConditionedPredictor
from jepa_robotics.training.ema import copy_params, update_ema
from jepa_robotics.training.losses import jepa_prediction_loss


class StateJEPA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        predictor_hidden_dims: list[int] | None = None,
        activation: str = "silu",
        layer_norm: bool = True,
        normalize_latents: bool = True,
    ) -> None:
        super().__init__()
        self.online_encoder = StateEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            normalize_output=normalize_latents,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        copy_params(self.online_encoder, self.target_encoder)
        self.predictor = ActionConditionedPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=predictor_hidden_dims or hidden_dims,
            activation=activation,
            normalize_output=normalize_latents,
        )

    def update_target(self, momentum: float) -> None:
        update_ema(self.online_encoder, self.target_encoder, momentum)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(states)

    def forward_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        variance_gamma: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z0 = self.online_encoder(states[:, 0])
        predictions = self.predictor.rollout(z0, actions)
        with torch.no_grad():
            target_shape = states[:, 1:].shape
            target_flat = states[:, 1:].reshape(-1, target_shape[-1])
            targets = self.target_encoder(target_flat).reshape(
                target_shape[0], target_shape[1], predictions.shape[-1]
            )
        online_flat = states.reshape(-1, states.shape[-1])
        online_latents = self.online_encoder(online_flat).reshape(
            states.shape[0], states.shape[1], predictions.shape[-1]
        )
        return jepa_prediction_loss(
            predictions=predictions,
            targets=targets,
            online_latents=online_latents,
            lambda_var=lambda_var,
            lambda_cov=lambda_cov,
            variance_gamma=variance_gamma,
        )
