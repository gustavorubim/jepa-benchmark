from __future__ import annotations

import torch

from jepa_robotics.models.encoders import StateEncoder
from jepa_robotics.training.ema import copy_params, update_ema


def test_copy_and_update_ema() -> None:
    source = StateEncoder(4, latent_dim=2, hidden_dims=[4], normalize_output=False)
    target = StateEncoder(4, latent_dim=2, hidden_dims=[4], normalize_output=False)
    copy_params(source, target)
    before = [param.clone() for param in target.parameters()]
    with torch.no_grad():
        for param in source.parameters():
            param.add_(1.0)
    update_ema(source, target, momentum=0.5)
    after = list(target.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after, strict=True))
    assert all(not param.requires_grad for param in target.parameters())
