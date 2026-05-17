from __future__ import annotations

import torch

from jepa_robotics.envs.make_env import make_env
from jepa_robotics.models.autoencoder import AutoencoderDynamics
from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.models.policies import JepaFeatureExtractor
from jepa_robotics.training.autoencoder_trainer import autoencoder_dynamics_loss


def test_jepa_shapes_and_gradient_flow() -> None:
    model = StateJEPA(input_dim=4, action_dim=2, latent_dim=8, hidden_dims=[16])
    states = torch.randn(3, 3, 4)
    actions = torch.randn(3, 2, 2)
    loss, metrics = model.forward_loss(states, actions)
    assert torch.isfinite(loss)
    loss.backward()
    assert metrics["loss"] > 0
    assert any(param.grad is not None for param in model.online_encoder.parameters())
    assert any(param.grad is not None for param in model.predictor.parameters())
    assert all(param.grad is None for param in model.target_encoder.parameters())
    rollout = model.predictor.rollout(model.online_encoder(states[:, 0]), actions)
    assert rollout.shape == (3, 2, 8)


def test_autoencoder_loss() -> None:
    model = AutoencoderDynamics(input_dim=4, action_dim=2, latent_dim=8)
    state = torch.randn(2, 4)
    action = torch.randn(2, 2)
    loss = autoencoder_dynamics_loss(model, state, action, torch.randn(2, 4))
    assert torch.isfinite(loss)


def test_jepa_feature_extractor_forward() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    extractor = JepaFeatureExtractor(
        env.observation_space,
        features_dim=8,
        hidden_dims=[16],
        include_desired_goal=False,
    )
    obs = {
        "observation": torch.zeros(2, 2),
        "achieved_goal": torch.zeros(2, 2),
        "desired_goal": torch.ones(2, 2),
    }
    features = extractor(obs)
    assert features.shape == (2, 8)


def test_jepa_feature_extractor_can_append_goal_without_checkpoint_dim_change() -> None:
    env = make_env("ToyGoal-v0", seed=0)
    extractor = JepaFeatureExtractor(
        env.observation_space,
        features_dim=8,
        hidden_dims=[16],
        include_desired_goal=False,
        append_desired_goal=True,
    )
    obs = {
        "observation": torch.zeros(2, 2),
        "achieved_goal": torch.zeros(2, 2),
        "desired_goal": torch.ones(2, 2),
    }
    features = extractor(obs)
    assert features.shape == (2, 10)
    assert torch.allclose(features[:, -2:], torch.ones(2, 2))
