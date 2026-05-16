from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from jepa_robotics import (
    analyze,
    collect_dataset,
    evaluate,
    plot,
    run_experiment,
    train_jepa,
    train_rl,
)
from jepa_robotics.config.load import load_config
from jepa_robotics.envs.make_env import ToyGoalEnv, make_env
from jepa_robotics.envs.observation_wrappers import VisualObservationWrapper, jepa_state_observation
from jepa_robotics.envs.reward_wrappers import PushShapedReward
from jepa_robotics.envs.vector_env import make_dummy_vec_env
from jepa_robotics.models.policies import JepaFeatureExtractor
from jepa_robotics.rl.callbacks import EvalSchedule
from jepa_robotics.rl.her import default_her_kwargs
from jepa_robotics.utils.logging import log_status
from jepa_robotics.utils.paths import ensure_dir, experiment_id_from_path
from jepa_robotics.utils.serialization import read_json, write_json, write_jsonl, write_metrics_csv
from jepa_robotics.utils.timers import elapsed_timer


def test_paths_serialization_logging_and_timer(tmp_path: Path) -> None:
    directory = ensure_dir(tmp_path / "nested")
    assert directory.exists()
    assert experiment_id_from_path(tmp_path / "outputs" / "smoke") == "smoke"
    write_json(directory / "data.json", {"a": 1})
    assert read_json(directory / "data.json") == {"a": 1}
    write_jsonl(directory / "rows.jsonl", [{"a": 1}, {"a": 2}])
    assert (directory / "rows.jsonl").read_text(encoding="utf-8").count("\n") == 2
    write_metrics_csv(directory / "metrics.csv", [{"metric": "x", "value": 1.0}])
    assert (directory / "metrics.csv").exists()
    with elapsed_timer() as elapsed:
        assert elapsed() >= 0.0
    log_status("utility smoke")


def test_optional_env_wrappers_and_vector_env() -> None:
    env = PushShapedReward(ToyGoalEnv(seed=0))
    obs, _ = env.reset(seed=0)
    _obs, reward, _terminated, _truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward <= 0.0
    assert "shaped_reward" in info
    assert jepa_state_observation(obs, include_goal=True).shape == (6,)

    visual = VisualObservationWrapper(ToyGoalEnv(seed=0), image_size=8)
    visual_obs, _ = visual.reset(seed=0)
    assert visual_obs["image"].shape == (8, 8, 3)

    vec = make_dummy_vec_env("ToyGoal-v0", seed=0, max_episode_steps=2)
    reset = vec.reset()
    assert "observation" in reset
    vec.close()


def test_policy_feature_extractor_and_small_helpers() -> None:
    observation_space = gym.spaces.Dict(
        {
            "observation": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "achieved_goal": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "desired_goal": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
        }
    )
    extractor = JepaFeatureExtractor(observation_space, features_dim=4, include_desired_goal=True)
    output = extractor(
        {
            "observation": torch.zeros(2, 2),
            "achieved_goal": torch.zeros(2, 2),
            "desired_goal": torch.ones(2, 2),
        }
    )
    assert output.shape == (2, 4)
    assert default_her_kwargs()["n_sampled_goal"] == 4
    assert EvalSchedule(eval_freq=10, n_eval_episodes=2).eval_freq == 10
    assert load_config(None).experiment.name == "smoke"
    assert callable(train_rl.run)
    assert callable(collect_dataset.run)
    assert callable(train_jepa.run)
    assert callable(evaluate.run)
    assert callable(plot.run)
    assert callable(analyze.run)
    assert callable(run_experiment.run)


def test_make_env_invalid_obs_mode() -> None:
    env = make_env("ToyGoal-v0", seed=0, obs_mode="bad")
    assert isinstance(env, ToyGoalEnv)
