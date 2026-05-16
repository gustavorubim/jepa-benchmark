"""Metric aggregation for experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def steps_to_threshold(df: pd.DataFrame, threshold: float) -> float:
    reached = df[df["mean_success_rate"] >= threshold].sort_values("global_step")
    if reached.empty:
        return float("nan")
    return float(reached.iloc[0]["global_step"])


def area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def aggregate_learning_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for (method, seed), group in df.groupby(["method", "seed"], dropna=False):
        ordered = group.sort_values("global_step")
        steps = ordered["global_step"].to_numpy(dtype=float)
        success = ordered["mean_success_rate"].to_numpy(dtype=float)
        reward = ordered["mean_episode_reward"].to_numpy(dtype=float)
        rows.append(
            {
                "method": str(method),
                "seed": int(seed),
                "steps_to_success_25": steps_to_threshold(ordered, 0.25),
                "steps_to_success_50": steps_to_threshold(ordered, 0.50),
                "steps_to_success_75": steps_to_threshold(ordered, 0.75),
                "auc_success": area_under_curve(steps, success),
                "auc_reward": area_under_curve(steps, reward),
                "final_success_rate": float(success[-1]),
                "final_mean_reward": float(reward[-1]),
            }
        )
    return pd.DataFrame(rows)


def mean_standard_error(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    grouped = df.groupby("method")[value_column]
    return grouped.agg(mean="mean", sem="sem", count="count").reset_index()
