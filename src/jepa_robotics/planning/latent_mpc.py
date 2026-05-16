"""Latent MPC controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.planning.action_sampling import sample_uniform_actions
from jepa_robotics.planning.cem import cem_optimize
from jepa_robotics.planning.scoring import score_action_sequences


@dataclass(frozen=True)
class MPCDiagnostics:
    best_candidate_score: float
    score_mean: float
    score_std: float
    selected_action_norm: float


class LatentMPC:
    def __init__(
        self,
        model: StateJEPA,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
        planner: str = "random",
        horizon: int = 8,
        num_candidates: int = 512,
        num_elites: int = 64,
        iterations: int = 4,
        lambda_action: float = 0.001,
        seed: int = 0,
    ) -> None:
        self.model = model.to(device).eval()
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.device = device
        self.planner = planner
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.iterations = iterations
        self.lambda_action = lambda_action
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics = MPCDiagnostics(0.0, 0.0, 0.0, 0.0)

    def act(self, obs: dict[str, np.ndarray], goal: np.ndarray | None = None) -> np.ndarray:
        del goal
        if self.planner == "cem":
            sequence, scores = cem_optimize(
                rng=self.rng,
                score_fn=lambda candidates: score_action_sequences(
                    self.model, obs, candidates, self.lambda_action, self.device
                ),
                horizon=self.horizon,
                action_low=self.action_low,
                action_high=self.action_high,
                num_candidates=self.num_candidates,
                num_elites=min(self.num_elites, self.num_candidates),
                iterations=self.iterations,
                init_std=0.7,
                min_std=0.05,
            )
        else:
            candidates = sample_uniform_actions(
                self.rng, self.num_candidates, self.horizon, self.action_low, self.action_high
            )
            scores = score_action_sequences(
                self.model, obs, candidates, self.lambda_action, self.device
            )
            sequence = candidates[int(np.argmax(scores))]
        action = sequence[0].astype(np.float32)
        self.last_diagnostics = MPCDiagnostics(
            best_candidate_score=float(np.max(scores)),
            score_mean=float(np.mean(scores)),
            score_std=float(np.std(scores)),
            selected_action_norm=float(np.linalg.norm(action)),
        )
        return action
