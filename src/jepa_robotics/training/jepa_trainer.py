"""JEPA training loop."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from jepa_robotics.config.load import dump_config
from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.data.normalization import RunningNormalizer
from jepa_robotics.data.trajectory_dataset import (
    TrajectoryWindowDataset,
    split_window_dataset_by_episode,
)
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
    train_dataset, val_dataset = split_window_dataset_by_episode(
        dataset,
        validation_fraction=config.jepa.validation_fraction,
        seed=seed,
    )
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
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = _make_loader(val_dataset, config, shuffle=False)
    forward_model: Any = torch.compile(model) if config.jepa.compile_model else model
    train_rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(config.jepa.epochs):
        train_metrics = _run_epoch(
            model=model,
            forward_model=forward_model,
            loader=train_loader,
            device=device,
            config=config,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model=model,
            forward_model=forward_model,
            loader=val_loader,
            device=device,
            config=config,
            optimizer=None,
        )
        train_rows.append(_metrics_row(epoch, seed, "train", train_metrics, config))
        val_rows.append(_metrics_row(epoch, seed, "val", val_metrics, config))
        val_loss = float(val_metrics["loss"])
        if val_loss < best_val - config.jepa.early_stop_min_delta:
            best_val = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            config.jepa.early_stop_patience is not None
            and stale_epochs >= config.jepa.early_stop_patience
        ):
            break
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.online_encoder.state_dict(), out / "encoder.pt")
    torch.save(model.target_encoder.state_dict(), out / "target_encoder.pt")
    torch.save(model.predictor.state_dict(), out / "predictor.pt")
    normalizer = RunningNormalizer.fit(dataset.arrays.observations)
    with (out / "normalizer.pkl").open("wb") as handle:
        pickle.dump({"schema_version": "normalizer_v1", "normalizer": normalizer}, handle)
    pd.DataFrame(train_rows).to_csv(out / "train_metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "val_metrics.csv", index=False)
    dump_config(config, out / "config_resolved.yaml")
    return out


def _make_loader(
    dataset: TrajectoryWindowDataset, config: BenchmarkConfig, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    num_workers = max(0, config.jepa.num_workers)
    return DataLoader(
        dataset,
        batch_size=config.jepa.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.jepa.pin_memory,
        persistent_workers=config.jepa.persistent_workers and num_workers > 0,
    )


def _run_epoch(
    model: StateJEPA,
    forward_model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: BenchmarkConfig,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    grad_context = nullcontext() if training else torch.no_grad()
    with grad_context:
        for batch in loader:
            states = batch["states"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, config):
                loss, metrics = forward_model.forward_loss(
                    states,
                    actions,
                    lambda_var=config.jepa.lambda_var,
                    lambda_cov=config.jepa.lambda_cov,
                    variance_gamma=config.jepa.variance_gamma,
                )
            if training:
                assert optimizer is not None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.jepa.gradient_clip_norm)
                optimizer.step()
                model.update_target(config.jepa.ema_momentum)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@contextmanager
def _amp_context(device: torch.device, config: BenchmarkConfig) -> Iterator[None]:
    if not config.jepa.amp:
        yield
        return
    dtype = torch.float16 if config.jepa.amp_dtype == "float16" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def _metrics_row(
    epoch: int,
    seed: int,
    split: str,
    metrics: dict[str, float],
    config: BenchmarkConfig,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "epoch": epoch,
        "global_step": epoch,
        "seed": seed,
        "method": "state_jepa",
        "split": split,
        "learning_rate": config.jepa.learning_rate,
    }
    prefix = "train" if split == "train" else "val"
    row[f"{prefix}_loss"] = metrics["loss"]
    row.update(metrics)
    return row
