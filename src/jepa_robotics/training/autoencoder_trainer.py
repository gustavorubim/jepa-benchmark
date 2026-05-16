"""Autoencoder baseline training helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.trajectory_dataset import (
    TrajectoryWindowDataset,
    split_window_dataset_by_episode,
)
from jepa_robotics.models.autoencoder import AutoencoderDynamics
from jepa_robotics.training.optim import make_adamw
from jepa_robotics.utils.device import get_torch_device


def autoencoder_dynamics_loss(
    model: AutoencoderDynamics,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    reconstruction, predicted_next = model(state, action)
    return F.mse_loss(reconstruction, state) + beta * F.mse_loss(predicted_next, next_state)


def train_autoencoder_model(
    config: BenchmarkConfig,
    dataset_path: str | Path,
    seed: int,
    output_dir: str | Path,
) -> Path:
    dataset = TrajectoryWindowDataset.from_npz(dataset_path, horizon=1)
    train_dataset, val_dataset = split_window_dataset_by_episode(
        dataset,
        validation_fraction=config.autoencoder.validation_fraction,
        seed=seed,
    )
    device = get_torch_device(config.device.preferred)
    model = AutoencoderDynamics(
        input_dim=dataset.input_dim,
        action_dim=dataset.action_dim,
        latent_dim=config.autoencoder.latent_dim,
        hidden_dims=config.autoencoder.hidden_dims,
    ).to(device)
    optimizer = make_adamw(
        model.parameters(),
        learning_rate=config.autoencoder.learning_rate,
        weight_decay=config.autoencoder.weight_decay,
    )
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = _make_loader(val_dataset, config, shuffle=False)
    train_rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(config.autoencoder.epochs):
        train_metrics = _run_epoch(model, train_loader, device, config, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, config, None)
        train_rows.append(_row(epoch, seed, "train", train_metrics, config))
        val_rows.append(_row(epoch, seed, "val", val_metrics, config))
        val_loss = val_metrics["loss"]
        if val_loss < best_val - config.autoencoder.early_stop_min_delta:
            best_val = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            config.autoencoder.early_stop_patience is not None
            and stale_epochs >= config.autoencoder.early_stop_patience
        ):
            break
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out / "encoder.pt")
    torch.save(model.decoder.state_dict(), out / "decoder.pt")
    torch.save(model.predictor.state_dict(), out / "predictor.pt")
    pd.DataFrame(train_rows).to_csv(out / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "val_metrics.csv", index=False)
    dump_config(config, out / "config_resolved.yaml")
    return out


def _make_loader(
    dataset: TrajectoryWindowDataset, config: BenchmarkConfig, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    num_workers = max(0, config.autoencoder.num_workers)
    return DataLoader(
        dataset,
        batch_size=config.autoencoder.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.autoencoder.pin_memory,
        persistent_workers=config.autoencoder.persistent_workers and num_workers > 0,
    )


def _run_epoch(
    model: AutoencoderDynamics,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: BenchmarkConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction_mse": 0.0, "dynamics_mse": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            state = states[:, 0]
            next_state = states[:, 1]
            action = actions[:, 0]
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            reconstruction, predicted_next = model(state, action)
            reconstruction_mse = F.mse_loss(reconstruction, state)
            dynamics_mse = F.mse_loss(predicted_next, next_state)
            loss = reconstruction_mse + config.autoencoder.beta_dynamics * dynamics_mse
            if training:
                assert optimizer is not None
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["reconstruction_mse"] += float(reconstruction_mse.detach().cpu())
            totals["dynamics_mse"] += float(dynamics_mse.detach().cpu())
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _row(
    epoch: int,
    seed: int,
    split: str,
    metrics: dict[str, float],
    config: BenchmarkConfig,
) -> dict[str, float | int | str]:
    prefix = "train" if split == "train" else "val"
    return {
        "epoch": epoch,
        "global_step": epoch,
        "seed": seed,
        "method": "autoencoder",
        "split": split,
        f"{prefix}_loss": metrics["loss"],
        "loss": metrics["loss"],
        "reconstruction_mse": metrics["reconstruction_mse"],
        "dynamics_mse": metrics["dynamics_mse"],
        "learning_rate": config.autoencoder.learning_rate,
    }
