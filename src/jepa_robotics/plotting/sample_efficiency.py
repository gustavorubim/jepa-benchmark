"""Sample-efficiency plots and tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, output_base: str | Path) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    summary = df.groupby(x)[y].mean().reset_index()
    ax.bar(summary[x].astype(str), summary[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
