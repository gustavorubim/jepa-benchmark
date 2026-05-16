"""Optional shaped reward wrappers for ablations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ReachShapedReward(gym.Wrapper[Any, Any, Any, Any]):
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = -float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        info["shaped_reward"] = reward
        return obs, reward, terminated, truncated, info


class PushShapedReward(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(self, env: gym.Env[Any, Any], alpha: float = 0.1) -> None:
        super().__init__(env)
        self.alpha = alpha

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired = np.asarray(obs["desired_goal"], dtype=np.float32)
        state = np.asarray(obs["observation"], dtype=np.float32)
        gripper = state[: achieved.shape[0]]
        reward = -float(np.linalg.norm(achieved - desired)) - self.alpha * float(
            np.linalg.norm(gripper - achieved)
        )
        info["shaped_reward"] = reward
        return obs, reward, terminated, truncated, info
