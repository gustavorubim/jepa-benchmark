"""Metric aggregation for experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

AGGREGATE_KEYS = ["phase", "env_id", "method", "seed"]


def steps_to_threshold(
    df: pd.DataFrame, threshold: float, step_column: str = "global_step"
) -> float:
    reached = df[df["mean_success_rate"] >= threshold].sort_values(step_column)
    if reached.empty:
        return float("nan")
    return float(reached.iloc[0][step_column])


def area_under_curve(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def aggregate_learning_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = _with_aggregate_columns(df)
    rows: list[dict[str, float | str | int | None]] = []
    for keys, group in metrics.groupby(AGGREGATE_KEYS, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_data = dict(zip(AGGREGATE_KEYS, key_values, strict=True))
        ordered = group.sort_values("global_step")
        step_column = (
            "environment_interactions" if "environment_interactions" in ordered else "global_step"
        )
        steps = ordered[step_column].to_numpy(dtype=float)
        success = ordered["mean_success_rate"].to_numpy(dtype=float)
        reward = ordered["mean_episode_reward"].to_numpy(dtype=float)
        source_paths = (
            sorted(ordered["source_path"].dropna().astype(str).unique())
            if "source_path" in ordered
            else []
        )
        rows.append(
            {
                **key_data,
                "comparison_group": _first_present(ordered, "comparison_group"),
                "reward_mode": _first_present(ordered, "reward_mode"),
                "total_steps": float(ordered["total_steps"].max())
                if "total_steps" in ordered
                else float(steps[-1]),
                "total_environment_interactions": float(ordered[step_column].max()),
                "configured_environment_budget": float(
                    ordered["configured_environment_budget"].max()
                )
                if "configured_environment_budget" in ordered
                else float(ordered[step_column].max()),
                "source_path": ";".join(source_paths),
                "steps_to_success_25": steps_to_threshold(ordered, 0.25, step_column),
                "steps_to_success_50": steps_to_threshold(ordered, 0.50, step_column),
                "steps_to_success_75": steps_to_threshold(ordered, 0.75, step_column),
                "auc_success": area_under_curve(steps, success),
                "auc_reward": area_under_curve(steps, reward),
                "final_success_rate": float(success[-1]),
                "final_mean_reward": float(reward[-1]),
            }
        )
    return pd.DataFrame(rows)


def mean_standard_error(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    group_columns = [column for column in ["phase", "env_id", "method"] if column in df.columns]
    grouped = df.groupby(group_columns or ["method"])[value_column]
    return grouped.agg(mean="mean", sem="sem", count="count").reset_index()


def _with_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    metrics = df.copy()
    for column, default in {
        "phase": "unknown",
        "env_id": "unknown",
        "method": "unknown",
        "seed": 0,
    }.items():
        if column not in metrics:
            metrics[column] = default
    return metrics


def _first_present(df: pd.DataFrame, column: str) -> str:
    if column not in df:
        return ""
    values = df[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else ""
