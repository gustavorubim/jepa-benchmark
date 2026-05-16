"""Reusable callback state for RL training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalSchedule:
    eval_freq: int
    n_eval_episodes: int


@dataclass
class EarlyStopState:
    """Track consecutive successful evaluations for early stopping."""

    threshold: float | None
    patience: int = 3
    consecutive_successes: int = 0
    should_stop: bool = False

    def observe(self, success_rate: float) -> bool:
        if self.threshold is None:
            return False
        if success_rate >= self.threshold:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        self.should_stop = self.consecutive_successes >= max(1, self.patience)
        return self.should_stop
