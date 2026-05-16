from __future__ import annotations

import numpy as np
import torch

from jepa_robotics.envs.make_env import make_env
from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.planning.action_sampling import sample_uniform_actions
from jepa_robotics.planning.cem import cem_optimize
from jepa_robotics.planning.latent_mpc import LatentMPC
from jepa_robotics.planning.scoring import (
    make_goal_proxy,
    make_goal_state_bank,
    score_action_sequences,
)


def test_latent_mpc_action_and_scoring() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    obs, _ = env.reset(seed=0)
    model = StateJEPA(input_dim=4, action_dim=2, latent_dim=8, hidden_dims=[16])
    controller = LatentMPC(
        model,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        device=torch.device("cpu"),
        planner="random",
        horizon=2,
        num_candidates=4,
        seed=0,
    )
    action = controller.act(obs)
    assert action.shape == (2,)
    candidates = sample_uniform_actions(
        np.random.default_rng(0),
        4,
        2,
        np.asarray([-1.0, -1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )
    scores = score_action_sequences(model, obs, candidates, 0.001, torch.device("cpu"))
    assert scores.shape == (4,)
    assert make_goal_proxy(obs).shape == (4,)


def test_latent_mpc_uses_dataset_goal_bank() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    obs, _ = env.reset(seed=0)
    model = StateJEPA(input_dim=4, action_dim=2, latent_dim=8, hidden_dims=[16])
    bank = make_goal_state_bank(
        observations=np.stack([obs["observation"], obs["desired_goal"]]),
        achieved_goals=np.stack([obs["achieved_goal"], obs["desired_goal"]]),
    )
    controller = LatentMPC(
        model,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        device=torch.device("cpu"),
        planner="random",
        horizon=2,
        num_candidates=4,
        seed=0,
        goal_bank=bank,
    )
    controller.act(obs)
    assert controller.last_diagnostics.goal_source == "nearest_dataset_state"
    assert controller.last_diagnostics.nearest_goal_distance == 0.0


def test_cem_optimizer() -> None:
    best, scores = cem_optimize(
        rng=np.random.default_rng(0),
        score_fn=lambda candidates: -np.square(candidates).sum(axis=(1, 2)),
        horizon=2,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        num_candidates=8,
        num_elites=2,
        iterations=2,
        init_std=0.5,
        min_std=0.01,
    )
    assert best.shape == (2, 1)
    assert scores.shape == (8,)
