"""Analysis table generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from jepa_robotics.evaluation.metrics import aggregate_learning_metrics


def write_analysis_tables(metrics: pd.DataFrame, reports_dir: str | Path) -> dict[str, Path]:
    tables_dir = Path(reports_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_learning_metrics(metrics)
    aggregate = _with_normalized_auc(aggregate, metrics)
    aggregate_path = tables_dir / "aggregate_metrics.csv"
    threshold_path = tables_dir / "threshold_metrics.csv"
    tests_path = tables_dir / "statistical_tests.csv"
    summary_path = tables_dir / "summary_statistics.csv"
    threshold_fraction_path = tables_dir / "threshold_fractions.csv"
    aggregate.to_csv(aggregate_path, index=False)
    aggregate[
        [
            "method",
            "seed",
            "steps_to_success_25",
            "steps_to_success_50",
            "steps_to_success_75",
        ]
    ].to_csv(threshold_path, index=False)
    _summary_statistics(aggregate).to_csv(summary_path, index=False)
    _threshold_fractions(aggregate).to_csv(threshold_fraction_path, index=False)
    tests = _paired_tests(aggregate)
    tests.to_csv(tests_path, index=False)
    return {
        "aggregate": aggregate_path,
        "thresholds": threshold_path,
        "tests": tests_path,
        "summary": summary_path,
        "threshold_fractions": threshold_fraction_path,
    }


def _with_normalized_auc(aggregate: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    max_steps = metrics.groupby(["method", "seed"], dropna=False)["global_step"].max()
    normalized = aggregate.copy()
    values: list[float] = []
    for row in normalized.itertuples(index=False):
        max_step = float(max_steps.get((row.method, row.seed), 0.0))
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
    for method, group in aggregate.groupby("method", dropna=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            ci_low, ci_high = _bootstrap_ci(values)
            rows.append(
                {
                    "method": str(method),
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
    for method, group in aggregate.groupby("method", dropna=False):
        for label, column in thresholds.items():
            reached = group[column].notna()
            rows.append(
                {
                    "method": str(method),
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


def _paired_tests(aggregate: pd.DataFrame) -> pd.DataFrame:
    methods = sorted(aggregate["method"].unique())
    rows: list[dict[str, float | str]] = []
    if len(methods) < 2:
        return pd.DataFrame(
            [{"method_a": methods[0] if methods else "none", "method_b": "none", "effect": 0.0}]
        )
    baseline = methods[0]
    for method in methods[1:]:
        joined = aggregate[aggregate["method"].isin([baseline, method])].pivot(
            index="seed", columns="method", values="auc_success"
        )
        joined = joined.dropna()
        if len(joined) >= 2:
            diff = joined[method] - joined[baseline]
            t_stat, p_value = stats.ttest_1samp(diff, 0.0)
            effect = float(diff.mean())
            try:
                wilcoxon = stats.wilcoxon(diff)
                wilcoxon_stat = float(wilcoxon.statistic)
                wilcoxon_p = float(wilcoxon.pvalue)
            except ValueError:
                wilcoxon_stat = np.nan
                wilcoxon_p = np.nan
            pooled_std = float(diff.std(ddof=1))
            cohens_d = float(diff.mean() / pooled_std) if pooled_std > 0 else np.nan
        else:
            t_stat, p_value, effect = np.nan, np.nan, np.nan
            wilcoxon_stat, wilcoxon_p, cohens_d = np.nan, np.nan, np.nan
        rows.append(
            {
                "method_a": baseline,
                "method_b": method,
                "metric": "auc_success",
                "effect": effect,
                "cohens_d": cohens_d,
                "paired_t_stat": float(t_stat),
                "paired_t_p_value": float(p_value),
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)
