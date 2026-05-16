"""Latent planner scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from jepa_robotics.data.transforms import stack_state
from jepa_robotics.models.jepa import StateJEPA


@dataclass(frozen=True)
class GoalStateBank:
    states: np.ndarray
    achieved_goals: np.ndarray


@dataclass(frozen=True)
class GoalContext:
    z0: torch.Tensor
    z_goal: torch.Tensor
    goal_source: str
    nearest_goal_distance: float


def make_goal_proxy(obs: dict[str, np.ndarray]) -> np.ndarray:
    observation = np.asarray(obs["observation"], dtype=np.float32).copy()
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    return stack_state(observation, desired[: achieved.shape[0]])


def make_goal_state_bank(
    observations: np.ndarray,
    achieved_goals: np.ndarray,
) -> GoalStateBank:
    states = np.stack(
        [
            stack_state(observation, achieved_goal)
            for observation, achieved_goal in zip(observations, achieved_goals, strict=True)
        ]
    ).astype(np.float32)
    return GoalStateBank(states=states, achieved_goals=achieved_goals.astype(np.float32))


def score_action_sequences(
    model: StateJEPA,
    obs: dict[str, np.ndarray],
    action_sequences: np.ndarray,
    lambda_action: float,
    device: torch.device,
    goal_bank: GoalStateBank | None = None,
) -> np.ndarray:
    context = latent_goal_context(model, obs, device, goal_bank=goal_bank)
    return score_action_sequences_from_latents(
        model, context.z0, context.z_goal, action_sequences, lambda_action, device
    )


def score_action_sequences_from_latents(
    model: StateJEPA,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    action_sequences: np.ndarray,
    lambda_action: float,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        actions = torch.as_tensor(action_sequences, dtype=torch.float32, device=device)
        z0_batch = z0.repeat(actions.shape[0], 1)
        preds = model.predictor.rollout(z0_batch, actions)
        z_final = F.normalize(preds[:, -1], dim=-1)
        latent_cost = (z_final - z_goal).pow(2).sum(dim=-1)
        action_cost = actions.pow(2).sum(dim=(1, 2))
        score = -(latent_cost + lambda_action * action_cost)
    return score.cpu().numpy().astype(np.float32)


def latent_goal_context(
    model: StateJEPA,
    obs: dict[str, np.ndarray],
    device: torch.device,
    goal_bank: GoalStateBank | None = None,
) -> GoalContext:
    state = stack_state(obs["observation"], obs["achieved_goal"])
    goal_state, goal_source, nearest_goal_distance = _goal_state(obs, goal_bank)
    with torch.no_grad():
        z0 = model.online_encoder(
            torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        )
        z_goal = model.target_encoder(
            torch.as_tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)
        )
    return GoalContext(
        z0=z0,
        z_goal=F.normalize(z_goal, dim=-1),
        goal_source=goal_source,
        nearest_goal_distance=nearest_goal_distance,
    )


def _goal_state(
    obs: dict[str, np.ndarray],
    goal_bank: GoalStateBank | None,
) -> tuple[np.ndarray, str, float]:
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    if goal_bank is None or len(goal_bank.states) == 0:
        proxy = make_goal_proxy(obs)
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        return proxy, "fake_state_proxy", float(np.linalg.norm(desired - achieved))
    distances = np.linalg.norm(goal_bank.achieved_goals - desired, axis=1)
    index = int(np.argmin(distances))
    return goal_bank.states[index], "nearest_dataset_state", float(distances[index])
