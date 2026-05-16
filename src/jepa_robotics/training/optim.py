"""Optimizer helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def make_adamw(
    parameters: Iterable[torch.Tensor],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
