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
) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in df.groupby("method"):
        summary = group.groupby("global_step")[y_column].agg(["mean", "sem"]).reset_index()
        x = summary["global_step"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, label=str(method))
        ax.fill_between(x, y - sem, y + sem, alpha=0.2)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
