"""Environment factory with a deterministic smoke environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import error as gym_error
from gymnasium import spaces


class ToyGoalEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Tiny goal-conditioned environment used for smoke tests."""

    def __init__(self, max_episode_steps: int = 5, seed: int = 0) -> None:
        super().__init__()
        self.metadata = {"render_modes": ["rgb_array"]}
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(2, dtype=np.float32)
        self._goal = np.ones(2, dtype=np.float32)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        del options
        self._step = 0
        self._state = self._rng.uniform(-0.25, 0.25, size=2).astype(np.float32)
        self._goal = self._rng.uniform(0.4, 0.8, size=2).astype(np.float32)
        return self._obs(), {"is_success": self._success()}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_space = self.action_space
        assert isinstance(action_space, spaces.Box)
        clipped = np.clip(action, action_space.low, action_space.high).astype(np.float32)
        self._state = (self._state + 0.15 * clipped).astype(np.float32)
        self._step += 1
        distance = float(np.linalg.norm(self._state - self._goal))
        reward = -distance
        terminated = distance < 0.05
        truncated = self._step >= self.max_episode_steps
        return self._obs(), reward, terminated, truncated, {"is_success": float(terminated)}

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        del info
        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def render(self) -> Any:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        point = np.clip(((self._state + 1.0) * 12).astype(int), 0, 31)
        goal = np.clip(((self._goal + 1.0) * 12).astype(int), 0, 31)
        image[point[0], point[1]] = [255, 255, 255]
        image[goal[0], goal[1]] = [0, 255, 0]
        return image

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "observation": self._state.copy(),
            "achieved_goal": self._state.copy(),
            "desired_goal": self._goal.copy(),
        }

    def _success(self) -> float:
        return float(np.linalg.norm(self._state - self._goal) < 0.05)


def make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
    obs_mode: str = "state",
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> gym.Env[Any, Any]:
    """Create a robotics environment or the built-in smoke-test environment."""
    if env_id == "ToyGoal-v0":
        return ToyGoalEnv(max_episode_steps=max_episode_steps or 5, seed=seed)

    try:
        import gymnasium_robotics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise RuntimeError(
            "gymnasium-robotics is required for Fetch environments. Run `uv sync` first."
        ) from exc

    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    resolved_env_id = resolve_env_id(env_id)
    env = gym.make(resolved_env_id, max_episode_steps=max_episode_steps, **kwargs)
    env.reset(seed=seed)
    del reward_mode
    if obs_mode == "visual":
        from jepa_robotics.envs.observation_wrappers import VisualObservationWrapper

        return VisualObservationWrapper(env)
    if obs_mode != "state":
        raise ValueError("obs_mode must be 'state' or 'visual'")
    return env


def resolve_env_id(env_id: str) -> str:
    """Resolve spec-era Fetch v3 IDs to the installed Gymnasium Robotics version."""
    if env_id.startswith("Fetch"):
        import gymnasium_robotics  # noqa: F401

    try:
        gym.spec(env_id)
        return env_id
    except gym_error.DeprecatedEnv:
        if env_id.endswith("-v3") and env_id.startswith("Fetch"):
            return f"{env_id[:-2]}v4"
        raise
