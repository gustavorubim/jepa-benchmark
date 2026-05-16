"""EMA target-network updates."""

from __future__ import annotations

import torch.nn as nn


def copy_params(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad_(False)


def update_ema(source: nn.Module, target: nn.Module, momentum: float) -> None:
    source_params = dict(source.named_parameters())
    for name, target_param in target.named_parameters():
        target_param.data.mul_(momentum).add_(source_params[name].data, alpha=1.0 - momentum)
