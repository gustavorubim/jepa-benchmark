"""Latent planner scoring helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from jepa_robotics.data.transforms import stack_state
from jepa_robotics.models.jepa import StateJEPA


def make_goal_proxy(obs: dict[str, np.ndarray]) -> np.ndarray:
    observation = np.asarray(obs["observation"], dtype=np.float32).copy()
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    return stack_state(observation, desired[: achieved.shape[0]])


def score_action_sequences(
    model: StateJEPA,
    obs: dict[str, np.ndarray],
    action_sequences: np.ndarray,
    lambda_action: float,
    device: torch.device,
) -> np.ndarray:
    state = stack_state(obs["observation"], obs["achieved_goal"])
    goal_proxy = make_goal_proxy(obs)
    with torch.no_grad():
        z0 = model.online_encoder(
            torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        )
        actions = torch.as_tensor(action_sequences, dtype=torch.float32, device=device)
        z0_batch = z0.repeat(actions.shape[0], 1)
        preds = model.predictor.rollout(z0_batch, actions)
        z_final = F.normalize(preds[:, -1], dim=-1)
        z_goal = model.target_encoder(
            torch.as_tensor(goal_proxy, dtype=torch.float32, device=device).unsqueeze(0)
        )
        z_goal = F.normalize(z_goal, dim=-1)
        latent_cost = (z_final - z_goal).pow(2).sum(dim=-1)
        action_cost = (
            torch.as_tensor(action_sequences, dtype=torch.float32, device=device)
            .pow(2)
            .sum(dim=(1, 2))
        )
        score = -(latent_cost + lambda_action * action_cost)
    return score.cpu().numpy().astype(np.float32)
