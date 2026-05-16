"""Generate aggregate tables from saved metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jepa_robotics.plotting.generate import load_experiment_metrics
from jepa_robotics.plotting.tables import write_analysis_tables


def run(experiment: str) -> Path:
    experiment_path = Path(experiment)
    reports_dir = Path("reports") / experiment_path.name
    metrics = load_experiment_metrics(experiment_path)
    paths = write_analysis_tables(metrics, reports_dir)
    _write_report(experiment_path, reports_dir, paths)
    return reports_dir / "tables"


def _write_report(experiment_path: Path, reports_dir: Path, paths: dict[str, Path]) -> None:
    aggregate = pd.read_csv(paths["aggregate"])
    summary = pd.read_csv(paths["summary"])
    thresholds = pd.read_csv(paths["threshold_fractions"])
    best = aggregate.sort_values("auc_success_normalized", ascending=False).head(1)
    if best.empty:
        verdict = "No metrics were found; run an experiment before drawing conclusions."
    else:
        row = best.iloc[0]
        verdict = (
            f"Best observed method by normalized success AUC is `{row['method']}` "
            f"on seed `{row['seed']}`."
        )
    report = f"""# Experiment Analysis

## Setup

- Experiment path: `{experiment_path}`
- Tables path: `{reports_dir / "tables"}`

## Direct Verdict

{verdict}

## Result Tables

- `aggregate_metrics.csv`: per-method/per-seed final reward, final success, AUC,
  normalized AUC, and threshold steps.
- `summary_statistics.csv`: mean, SE, and bootstrap 95% CI by method.
- `threshold_fractions.csv`: fraction of seeds reaching 25%, 50%, and 75% success.
- `statistical_tests.csv`: paired t-test, Wilcoxon fallback, and effect-size columns
  where enough paired seeds exist.

## Caveats

The report is generated from saved CSV logs only. Single-seed smoke runs are pipeline checks,
not publishable evidence. Use confirm or full mode for multi-seed claims.

## Snapshot

Top aggregate row:

```text
{best.to_string(index=False) if not best.empty else "no data"}
```

Summary rows: {len(summary)}

Threshold-fraction rows: {len(thresholds)}
"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    print(run(args.experiment))


if __name__ == "__main__":
    main()
