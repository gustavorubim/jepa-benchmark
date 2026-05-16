"""Analysis table generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from jepa_robotics.evaluation.metrics import AGGREGATE_KEYS, aggregate_learning_metrics


def write_analysis_tables(metrics: pd.DataFrame, reports_dir: str | Path) -> dict[str, Path]:
    tables_dir = Path(reports_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_learning_metrics(metrics)
    aggregate = _with_normalized_auc(aggregate, metrics)
    aggregate_path = tables_dir / "aggregate_metrics.csv"
    threshold_path = tables_dir / "threshold_metrics.csv"
    tests_path = tables_dir / "statistical_tests.csv"
    summary_path = tables_dir / "summary_statistics.csv"
    robust_summary_path = tables_dir / "robust_summary.csv"
    threshold_fraction_path = tables_dir / "threshold_fractions.csv"
    if aggregate.empty:
        aggregate.to_csv(aggregate_path, index=False)
        pd.DataFrame().to_csv(threshold_path, index=False)
        pd.DataFrame().to_csv(summary_path, index=False)
        pd.DataFrame().to_csv(robust_summary_path, index=False)
        pd.DataFrame().to_csv(threshold_fraction_path, index=False)
        pd.DataFrame().to_csv(tests_path, index=False)
        return {
            "aggregate": aggregate_path,
            "thresholds": threshold_path,
            "tests": tests_path,
            "summary": summary_path,
            "robust_summary": robust_summary_path,
            "threshold_fractions": threshold_fraction_path,
        }
    aggregate.to_csv(aggregate_path, index=False)
    threshold_columns = [
        column
        for column in [
            "phase",
            "comparison_group",
            "env_id",
            "method",
            "seed",
            "reward_mode",
            "total_steps",
            "total_environment_interactions",
            "configured_environment_budget",
            "steps_to_success_25",
            "steps_to_success_50",
            "steps_to_success_75",
        ]
        if column in aggregate.columns
    ]
    aggregate[threshold_columns].to_csv(threshold_path, index=False)
    _summary_statistics(aggregate).to_csv(summary_path, index=False)
    _robust_summary(aggregate).to_csv(robust_summary_path, index=False)
    _threshold_fractions(aggregate).to_csv(threshold_fraction_path, index=False)
    _comparison_diagnostics(aggregate).to_csv(tests_path, index=False)
    return {
        "aggregate": aggregate_path,
        "thresholds": threshold_path,
        "tests": tests_path,
        "summary": summary_path,
        "robust_summary": robust_summary_path,
        "threshold_fractions": threshold_fraction_path,
    }


def _with_normalized_auc(aggregate: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        normalized = aggregate.copy()
        normalized["auc_success_normalized"] = []
        return normalized
    normalized = aggregate.copy()
    if normalized.empty:
        normalized["auc_success_normalized"] = []
        return normalized
    metric_frame = metrics.copy()
    for column in AGGREGATE_KEYS:
        if column not in metric_frame:
            metric_frame[column] = "unknown" if column != "seed" else 0
    step_column = (
        "environment_interactions" if "environment_interactions" in metric_frame else "global_step"
    )
    max_steps = metric_frame.groupby(AGGREGATE_KEYS, dropna=False)[step_column].max()
    values: list[float] = []
    for row in normalized.itertuples(index=False):
        key = tuple(getattr(row, column) for column in AGGREGATE_KEYS)
        max_step = float(getattr(row, "configured_environment_budget", max_steps.get(key, 0.0)))
        values.append(float(row.auc_success / max_step) if max_step > 0 else 0.0)
    normalized["auc_success_normalized"] = values
    return normalized


def _summary_statistics(aggregate: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc_success",
        "auc_success_normalized",
        "final_success_rate",
        "final_mean_reward",
        "steps_to_success_25",
        "steps_to_success_50",
        "steps_to_success_75",
    ]
    rows: list[dict[str, float | int | str]] = []
    if aggregate.empty:
        return pd.DataFrame(rows)
    group_columns = [
        column
        for column in ["comparison_group", "phase", "env_id", "method"]
        if column in aggregate.columns
    ]
    for keys, group in aggregate.groupby(group_columns or ["method"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_data = dict(zip(group_columns or ["method"], key_values, strict=True))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            ci_low, ci_high = _bootstrap_ci(values)
            rows.append(
                {
                    **key_data,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "se": float(stats.sem(values)) if len(values) > 1 else 0.0,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "n": len(values),
                }
            )
    return pd.DataFrame(rows)


def _threshold_fractions(aggregate: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "success_25": "steps_to_success_25",
        "success_50": "steps_to_success_50",
        "success_75": "steps_to_success_75",
    }
    rows: list[dict[str, float | int | str]] = []
    if aggregate.empty:
        return pd.DataFrame(rows)
    group_columns = [
        column
        for column in ["comparison_group", "phase", "env_id", "method"]
        if column in aggregate.columns
    ]
    for keys, group in aggregate.groupby(group_columns or ["method"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_data = dict(zip(group_columns or ["method"], key_values, strict=True))
        for label, column in thresholds.items():
            reached = group[column].notna()
            rows.append(
                {
                    **key_data,
                    "threshold": label,
                    "seeds_reached": int(reached.sum()),
                    "total_seeds": len(group),
                    "fraction_reached": float(reached.mean()) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, iterations: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if len(values) else float("nan")
        return value, value
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _robust_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "auc_success_normalized",
        "final_success_rate",
        "final_mean_reward",
        "steps_to_success_25",
        "steps_to_success_50",
        "steps_to_success_75",
    ]
    rows: list[dict[str, float | int | str]] = []
    group_columns = [
        column
        for column in ["comparison_group", "phase", "env_id", "method"]
        if column in aggregate.columns
    ]
    for keys, group in aggregate.groupby(group_columns or ["method"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_data = dict(zip(group_columns or ["method"], key_values, strict=True))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            iqm_low, iqm_high = _bootstrap_iqm_ci(values)
            rows.append(
                {
                    **key_data,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "iqm": _interquartile_mean(values),
                    "iqm_ci_low": iqm_low,
                    "iqm_ci_high": iqm_high,
                    "n": len(values),
                    "claim_ready_n": len(values) >= 5,
                }
            )
    return pd.DataFrame(rows)


def _interquartile_mean(values: np.ndarray) -> float:
    ordered = np.sort(values.astype(float))
    if len(ordered) < 4:
        return float(np.mean(ordered))
    trim = int(np.floor(0.25 * len(ordered)))
    trimmed = ordered[trim : len(ordered) - trim]
    return float(np.mean(trimmed)) if len(trimmed) else float(np.mean(ordered))


def _bootstrap_iqm_ci(values: np.ndarray, iterations: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        value = _interquartile_mean(values)
        return value, value
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    iqms = np.asarray([_interquartile_mean(sample) for sample in samples], dtype=float)
    low, high = np.percentile(iqms, [2.5, 97.5])
    return float(low), float(high)


def _comparison_diagnostics(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str | bool]] = []
    if aggregate.empty:
        return pd.DataFrame(
            [
                {
                    "phase": "none",
                    "env_id": "none",
                    "methods": "",
                    "n_methods": 0,
                    "min_seeds_per_method": 0,
                    "matched_total_steps": False,
                    "claim_ready": False,
                    "status": "no metrics",
                }
            ]
        )
    context_columns = [
        column for column in ["comparison_group", "env_id"] if column in aggregate.columns
    ]
    for keys, context in aggregate.groupby(context_columns or ["env_id"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        context_data = dict(zip(context_columns or ["env_id"], key_values, strict=True))
        methods = sorted(context["method"].astype(str).unique())
        seed_counts = context.groupby("method")["seed"].nunique()
        budget_column = (
            "configured_environment_budget"
            if "configured_environment_budget" in context
            else "total_environment_interactions"
            if "total_environment_interactions" in context
            else "total_steps"
        )
        budgets = sorted(context[budget_column].dropna().astype(float).unique())
        matched_budget = len(budgets) == 1
        min_seeds = int(seed_counts.min()) if not seed_counts.empty else 0
        claim_ready = len(methods) >= 2 and min_seeds >= 5 and matched_budget
        if claim_ready:
            status = "eligible for multi-seed descriptive comparison"
        elif len(methods) < 2:
            status = "single method only"
        elif min_seeds < 5:
            status = "underpowered: fewer than five seeds per method"
        else:
            status = "unmatched budgets"
        rows.append(
            {
                **context_data,
                "methods": ",".join(methods),
                "n_methods": len(methods),
                "min_seeds_per_method": min_seeds,
                "matched_total_steps": matched_budget,
                "claim_ready": claim_ready,
                "status": status,
            }
        )
    return pd.DataFrame(rows)
