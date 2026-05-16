from __future__ import annotations

import numpy as np

from jepa_robotics.envs.make_env import make_env, resolve_env_id
from jepa_robotics.envs.observation_wrappers import flatten_goal_observation, jepa_state_observation
from jepa_robotics.envs.reward_wrappers import ReachShapedReward
from jepa_robotics.envs.vector_env import make_sb3_vec_env
from jepa_robotics.envs.wrappers import InfoRecorder


def test_toy_goal_env_step() -> None:
    env = make_env("ToyGoal-v0", seed=0, max_episode_steps=2)
    obs, info = env.reset(seed=0)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    assert "is_success" in info
    flat = flatten_goal_observation(obs)
    state = jepa_state_observation(obs)
    assert flat.shape[0] == 6
    assert state.shape[0] == 4
    next_obs, reward, _terminated, truncated, _info = env.step(np.ones(2, dtype=np.float32))
    assert isinstance(reward, float)
    assert next_obs["observation"].shape == (2,)
    _obs, _reward, _terminated, truncated, _info = env.step(np.ones(2, dtype=np.float32))
    assert truncated


def test_wrappers_record_info_and_shape_reward() -> None:
    env = InfoRecorder(ReachShapedReward(make_env("ToyGoal-v0", seed=0)))
    obs, _ = env.reset(seed=0)
    _obs, reward, _terminated, _truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward <= 0
    assert "shaped_reward" in info
    assert env.last_info == info
    assert obs["achieved_goal"].shape == (2,)


def test_fetch_v3_ids_resolve_to_current_version() -> None:
    assert resolve_env_id("FetchReachDense-v3") == "FetchReachDense-v4"


def test_dummy_vector_env_supports_multiple_toy_envs(tmp_path) -> None:
    vec_env = make_sb3_vec_env(
        "ToyGoal-v0",
        seed=0,
        n_envs=2,
        vec_env_type="dummy",
        max_episode_steps=2,
        monitor_dir=str(tmp_path),
    )
    obs = vec_env.reset()
    assert isinstance(obs, dict)
    assert obs["observation"].shape == (2, 2)
    vec_env.close()
