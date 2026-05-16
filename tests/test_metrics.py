from __future__ import annotations

import numpy as np
import pandas as pd

from jepa_robotics.envs.make_env import make_env
from jepa_robotics.evaluation.generalization import add_observation_noise
from jepa_robotics.evaluation.metrics import (
    aggregate_learning_metrics,
    area_under_curve,
    mean_standard_error,
)
from jepa_robotics.evaluation.probes import fit_linear_probe
from jepa_robotics.evaluation.rollouts import RandomPolicy, evaluate_policy_rollouts


def test_metric_aggregation() -> None:
    df = pd.DataFrame(
        {
            "global_step": [0, 5, 10],
            "seed": [0, 0, 0],
            "method": ["sac", "sac", "sac"],
            "mean_success_rate": [0.0, 0.4, 0.8],
            "mean_episode_reward": [-2.0, -1.0, 0.0],
        }
    )
    aggregate = aggregate_learning_metrics(df)
    assert aggregate.loc[0, "steps_to_success_50"] == 10
    assert area_under_curve(np.asarray([0, 1]), np.asarray([0, 1])) == 0.5
    assert mean_standard_error(aggregate, "auc_success").loc[0, "count"] == 1


def test_rollout_probe_and_noise() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    policy = RandomPolicy(env.action_space, seed=0)
    metrics = evaluate_policy_rollouts(env, policy, episodes=1, seed=0)
    assert "mean_success_rate" in metrics
    probe = fit_linear_probe(np.eye(4), np.eye(4))
    assert probe["mse"] < 0.1
    obs, _ = env.reset(seed=0)
    noisy = add_observation_noise(obs, 0.1, seed=0)
    assert noisy["observation"].shape == obs["observation"].shape
