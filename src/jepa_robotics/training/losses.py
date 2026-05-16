"""JEPA losses and collapse diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 2.0 - 2.0 * F.cosine_similarity(pred, target, dim=-1)


def variance_regularization(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std) ** 2)


def covariance_regularization(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0)
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[-1]


def jepa_prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    online_latents: torch.Tensor,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.04,
    variance_gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    base = cosine_distance(F.normalize(predictions, dim=-1), F.normalize(targets, dim=-1)).mean()
    flat_latents = online_latents.reshape(-1, online_latents.shape[-1])
    var_loss = variance_regularization(flat_latents, variance_gamma)
    cov_loss = covariance_regularization(flat_latents)
    total = base + lambda_var * var_loss + lambda_cov * cov_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "jepa_loss": float(base.detach().cpu()),
        "variance_loss": float(var_loss.detach().cpu()),
        "covariance_loss": float(cov_loss.detach().cpu()),
        "latent_std_mean": float(flat_latents.std(dim=0).mean().detach().cpu()),
        "latent_std_min": float(flat_latents.std(dim=0).min().detach().cpu()),
        "latent_norm_mean": float(flat_latents.norm(dim=-1).mean().detach().cpu()),
    }
    return total, metrics
