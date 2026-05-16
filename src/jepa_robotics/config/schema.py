"""Pydantic schemas for experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperimentSection(BaseModel):
    name: str = "smoke"
    output_dir: Path = Path("outputs/smoke")
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])


class EnvSection(BaseModel):
    id: str = "FetchReachDense-v3"
    obs_mode: Literal["state", "visual"] = "state"
    reward_mode: str | None = None
    render_mode: str | None = None
    max_episode_steps: int = 50
    image_size: int = 84
    success_threshold: float = 0.05


class RLSection(BaseModel):
    algorithm: str = "SAC"
    policy: str = "MultiInputPolicy"
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    train_eval_episodes: int = 5
    final_eval_episodes: int = 20
    n_envs: int = 1
    vec_env_type: Literal["dummy", "subproc"] = "dummy"
    early_stop_success: float | None = None
    early_stop_patience: int = 3
    skip_existing: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.05
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    replay_buffer_class: str | None = None
    replay_buffer_kwargs: dict[str, Any] = Field(default_factory=dict)


class DatasetSection(BaseModel):
    source: Literal["random", "rl_policy", "mixture"] = "random"
    num_episodes: int = 1_000
    max_episode_steps: int = 50
    save_format: Literal["npz"] = "npz"
    random_fraction: float = 0.3
    n_envs: int = 1
    vec_env_type: Literal["dummy", "subproc"] = "dummy"
    skip_existing: bool = True


class JEPASection(BaseModel):
    latent_dim: int = 128
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    predictor_hidden_dims: list[int] | None = None
    activation: Literal["relu", "silu"] = "silu"
    layer_norm: bool = True
    normalize_latents: bool = True
    prediction_horizons: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    ema_momentum: float = 0.996
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    variance_gamma: float = 1.0
    gradient_clip_norm: float = 10.0
    validation_fraction: float = 0.2
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    compile_model: bool = False
    amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0

    @property
    def max_horizon(self) -> int:
        return max(self.prediction_horizons)

    @property
    def predictor_dims(self) -> list[int]:
        return self.predictor_hidden_dims or self.hidden_dims


class MPCSection(BaseModel):
    planner: Literal["random", "cem"] = "cem"
    horizon: int = 8
    num_candidates: int = 512
    num_elites: int = 64
    iterations: int = 4
    init_std: float = 0.7
    min_std: float = 0.05
    lambda_action: float = 0.001
    action_low: list[float] = Field(default_factory=lambda: [-1.0, -1.0, -1.0, -1.0])
    action_high: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


class AutoencoderSection(BaseModel):
    latent_dim: int = 128
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 256])
    batch_size: int = 512
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    beta_dynamics: float = 1.0
    validation_fraction: float = 0.2
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0


class DeviceSection(BaseModel):
    preferred: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class GeneralizationSection(BaseModel):
    perturbations: list[str] = Field(default_factory=list)
    levels: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])


class BenchmarkConfig(BaseModel):
    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    env: EnvSection = Field(default_factory=EnvSection)
    rl: RLSection = Field(default_factory=RLSection)
    dataset: DatasetSection = Field(default_factory=DatasetSection)
    jepa: JEPASection = Field(default_factory=JEPASection)
    mpc: MPCSection = Field(default_factory=MPCSection)
    autoencoder: AutoencoderSection = Field(default_factory=AutoencoderSection)
    device: DeviceSection = Field(default_factory=DeviceSection)
    generalization: GeneralizationSection = Field(default_factory=GeneralizationSection)

    def smoke_copy(self) -> BenchmarkConfig:
        data = self.model_dump(mode="python")
        data["experiment"]["name"] = "smoke"
        data["experiment"]["output_dir"] = Path("outputs/smoke")
        data["experiment"]["seeds"] = [0]
        data["env"]["id"] = "ToyGoal-v0"
        data["env"]["max_episode_steps"] = 5
        data["rl"]["total_timesteps"] = 5
        data["rl"]["eval_freq"] = 5
        data["rl"]["n_eval_episodes"] = 1
        data["rl"]["train_eval_episodes"] = 1
        data["rl"]["final_eval_episodes"] = 1
        data["rl"]["n_envs"] = 1
        data["rl"]["batch_size"] = 2
        data["dataset"]["num_episodes"] = 3
        data["dataset"]["max_episode_steps"] = 5
        data["dataset"]["n_envs"] = 1
        data["jepa"]["latent_dim"] = 8
        data["jepa"]["hidden_dims"] = [16]
        data["jepa"]["prediction_horizons"] = [1, 2]
        data["jepa"]["batch_size"] = 4
        data["jepa"]["epochs"] = 2
        data["jepa"]["validation_fraction"] = 0.34
        data["jepa"]["num_workers"] = 0
        data["jepa"]["pin_memory"] = False
        data["jepa"]["persistent_workers"] = False
        data["jepa"]["compile_model"] = False
        data["jepa"]["amp"] = False
        data["autoencoder"]["latent_dim"] = 8
        data["autoencoder"]["hidden_dims"] = [16]
        data["autoencoder"]["batch_size"] = 4
        data["autoencoder"]["epochs"] = 2
        data["autoencoder"]["validation_fraction"] = 0.34
        data["mpc"]["planner"] = "random"
        data["mpc"]["horizon"] = 2
        data["mpc"]["num_candidates"] = 8
        data["mpc"]["num_elites"] = 2
        data["mpc"]["iterations"] = 1
        data["mpc"]["action_low"] = [-1.0, -1.0]
        data["mpc"]["action_high"] = [1.0, 1.0]
        data["device"]["preferred"] = "cpu"
        return BenchmarkConfig.model_validate(data)
