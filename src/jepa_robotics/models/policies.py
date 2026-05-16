"""SB3 policy feature extractors backed by a JEPA encoder."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from jepa_robotics.models.encoders import StateEncoder


class JepaFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space[Any],
        encoder_checkpoint: str | None = None,
        features_dim: int = 128,
        freeze_encoder: bool = True,
        include_desired_goal: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)
        dict_space = observation_space
        if not isinstance(dict_space, gym.spaces.Dict):
            raise TypeError("JepaFeatureExtractor requires a Dict observation space.")
        obs_shape = dict_space["observation"].shape
        achieved_shape = dict_space["achieved_goal"].shape
        goal_shape = dict_space["desired_goal"].shape
        assert obs_shape is not None
        assert achieved_shape is not None
        assert goal_shape is not None
        obs_dim = int(obs_shape[0])
        achieved_dim = int(achieved_shape[0])
        goal_dim = int(goal_shape[0]) if include_desired_goal else 0
        self.include_desired_goal = include_desired_goal
        self.encoder = StateEncoder(obs_dim + achieved_dim + goal_dim, latent_dim=features_dim)
        if encoder_checkpoint is not None:
            self.encoder.load_state_dict(th.load(encoder_checkpoint, map_location="cpu"))
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        pieces = [observations["observation"], observations["achieved_goal"]]
        if self.include_desired_goal:
            pieces.append(observations["desired_goal"])
        return self.encoder(th.cat(pieces, dim=-1))
