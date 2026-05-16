"""Rollout helpers used by CLIs and tests."""

from __future__ import annotations

from typing import Protocol

import gymnasium as gym
import numpy as np


class Policy(Protocol):
    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


class RandomPolicy:
    def __init__(self, action_space: gym.Space[np.ndarray], seed: int) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        del obs
        return np.asarray(self.action_space.sample(), dtype=np.float32)


def evaluate_policy_rollouts(
    env: gym.Env,
    policy: Policy,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    rewards: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        last_info: dict[str, float] = {}
        for step in range(1_000):
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            last_info = info
            if terminated or truncated:
                lengths.append(step + 1)
                break
        rewards.append(total)
        successes.append(float(last_info.get("is_success", 0.0)))
    return {
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_success_rate": float(np.mean(successes)),
        "std_success_rate": float(np.std(successes)),
        "mean_episode_length": float(np.mean(lengths)),
    }
