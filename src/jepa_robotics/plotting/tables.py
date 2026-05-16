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
    aggregate_path = tables_dir / "aggregate_metrics.csv"
    threshold_path = tables_dir / "threshold_metrics.csv"
    tests_path = tables_dir / "statistical_tests.csv"
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
    tests = _paired_tests(aggregate)
    tests.to_csv(tests_path, index=False)
    return {"aggregate": aggregate_path, "thresholds": threshold_path, "tests": tests_path}


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
        else:
            t_stat, p_value, effect = np.nan, np.nan, np.nan
        rows.append(
            {
                "method_a": baseline,
                "method_b": method,
                "metric": "auc_success",
                "effect": effect,
                "paired_t_stat": float(t_stat),
                "paired_t_p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)
