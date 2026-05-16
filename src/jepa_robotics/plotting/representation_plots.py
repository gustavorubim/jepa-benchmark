"""Representation diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_probe_bars(df: pd.DataFrame, metric: str, output_base: str | Path) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["variable"].astype(str), df[metric])
    ax.set_xlabel("probe target")
    ax.set_ylabel(metric)
    ax.set_title(f"Linear probe {metric}")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
