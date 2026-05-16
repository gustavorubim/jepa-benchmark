from __future__ import annotations

import torch

from jepa_robotics.training.losses import (
    cosine_distance,
    covariance_regularization,
    jepa_prediction_loss,
    variance_regularization,
)


def test_losses_are_finite() -> None:
    predictions = torch.randn(4, 2, 8)
    targets = torch.randn(4, 2, 8)
    latents = torch.randn(4, 3, 8)
    loss, metrics = jepa_prediction_loss(predictions, targets, latents)
    assert torch.isfinite(loss)
    assert metrics["latent_std_mean"] > 0
    assert variance_regularization(latents.reshape(-1, 8)) >= 0
    assert covariance_regularization(latents.reshape(-1, 8)) >= 0
    assert cosine_distance(predictions, targets).shape == (4, 2)
