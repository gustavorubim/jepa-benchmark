"""JEPA training loop."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.normalization import RunningNormalizer
from jepa_robotics.data.trajectory_dataset import TrajectoryWindowDataset
from jepa_robotics.models.jepa import StateJEPA
from jepa_robotics.training.optim import make_adamw
from jepa_robotics.utils.device import get_torch_device


def train_jepa_model(
    config: BenchmarkConfig,
    dataset_path: str | Path,
    seed: int,
    output_dir: str | Path,
) -> Path:
    dataset = TrajectoryWindowDataset.from_npz(dataset_path, horizon=config.jepa.max_horizon)
    device = get_torch_device(config.device.preferred)
    model = StateJEPA(
        input_dim=dataset.input_dim,
        action_dim=dataset.action_dim,
        latent_dim=config.jepa.latent_dim,
        hidden_dims=config.jepa.hidden_dims,
        predictor_hidden_dims=config.jepa.predictor_dims,
        activation=config.jepa.activation,
        layer_norm=config.jepa.layer_norm,
        normalize_latents=config.jepa.normalize_latents,
    ).to(device)
    optimizer = make_adamw(
        model.parameters(),
        learning_rate=config.jepa.learning_rate,
        weight_decay=config.jepa.weight_decay,
    )
    loader = DataLoader(dataset, batch_size=config.jepa.batch_size, shuffle=True)
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(config.jepa.epochs):
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.forward_loss(
                states,
                actions,
                lambda_var=config.jepa.lambda_var,
                lambda_cov=config.jepa.lambda_cov,
                variance_gamma=config.jepa.variance_gamma,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.jepa.gradient_clip_norm)
            optimizer.step()
            model.update_target(config.jepa.ema_momentum)
        rows.append(
            {
                "epoch": epoch,
                "seed": seed,
                "method": "state_jepa",
                "train_loss": metrics["loss"],
                "val_loss": metrics["loss"],
                "cosine_similarity_h1": 1.0 - metrics["jepa_loss"] / 2.0,
                "cosine_similarity_h2": 1.0 - metrics["jepa_loss"] / 2.0,
                "latent_std_mean": metrics["latent_std_mean"],
                "latent_std_min": metrics["latent_std_min"],
                "latent_norm_mean": metrics["latent_norm_mean"],
                "learning_rate": config.jepa.learning_rate,
            }
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.online_encoder.state_dict(), out / "encoder.pt")
    torch.save(model.target_encoder.state_dict(), out / "target_encoder.pt")
    torch.save(model.predictor.state_dict(), out / "predictor.pt")
    normalizer = RunningNormalizer.fit(dataset.arrays.observations)
    with (out / "normalizer.pkl").open("wb") as handle:
        pickle.dump({"schema_version": "normalizer_v1", "normalizer": normalizer}, handle)
    pd.DataFrame(rows).to_csv(out / "train_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "val_metrics.csv", index=False)
    dump_config(config, out / "config_resolved.yaml")
    return out
