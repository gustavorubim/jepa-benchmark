"""Autoencoder baseline train-step helper."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jepa_robotics.models.autoencoder import AutoencoderDynamics


def autoencoder_dynamics_loss(
    model: AutoencoderDynamics,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    reconstruction, predicted_next = model(state, action)
    return F.mse_loss(reconstruction, state) + beta * F.mse_loss(predicted_next, next_state)
