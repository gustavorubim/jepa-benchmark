"""Observation conversion helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def flatten_goal_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["observation"], dtype=np.float32).ravel(),
            np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
            np.asarray(obs["desired_goal"], dtype=np.float32).ravel(),
        ]
    ).astype(np.float32)


def jepa_state_observation(obs: dict[str, np.ndarray], include_goal: bool = False) -> np.ndarray:
    pieces = [
        np.asarray(obs["observation"], dtype=np.float32).ravel(),
        np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
    ]
    if include_goal:
        pieces.append(np.asarray(obs["desired_goal"], dtype=np.float32).ravel())
    return np.concatenate(pieces).astype(np.float32)


class VisualObservationWrapper(gym.ObservationWrapper[Any, dict[str, Any], dict[str, Any]]):
    """Attach RGB render output while preserving the original state dict."""

    def __init__(self, env: gym.Env[Any, Any], image_size: int = 84) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
                "state": env.observation_space,
            }
        )

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        rendered: Any = self.env.render()
        image = np.asarray(
            rendered
            if rendered is not None
            else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        )
        resized = np.resize(image, (self.image_size, self.image_size, 3)).astype(np.uint8)
        return {"image": resized, "state": observation}
