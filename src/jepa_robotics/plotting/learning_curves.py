"""Learning curve plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(
    df: pd.DataFrame,
    y_column: str,
    ylabel: str,
    title: str,
    output_base: str | Path,
    group_columns: list[str] | None = None,
    x_column: str = "global_step",
) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    resolved_groups = group_columns or ["method"]
    resolved_groups = [column for column in resolved_groups if column in df.columns]
    for keys, group in df.groupby(resolved_groups or ["method"], dropna=False):
        summary = group.groupby(x_column)[y_column].agg(["mean", "sem"]).reset_index()
        x = summary[x_column].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
        key_values = keys if isinstance(keys, tuple) else (keys,)
        label = " / ".join(str(value) for value in key_values)
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - sem, y + sem, alpha=0.2)
    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
