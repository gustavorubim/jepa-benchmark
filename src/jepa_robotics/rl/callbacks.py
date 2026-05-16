"""Callback placeholders kept separate from CLI code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSchedule:
    eval_freq: int
    n_eval_episodes: int
