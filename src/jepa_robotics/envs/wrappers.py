"""Generic wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class InfoRecorder(gym.Wrapper[Any, Any, Any, Any]):
    """Record the most recent info dictionary for lightweight smoke tests."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self.last_info: dict[str, Any] = {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info)
        return obs, float(reward), terminated, truncated, info
