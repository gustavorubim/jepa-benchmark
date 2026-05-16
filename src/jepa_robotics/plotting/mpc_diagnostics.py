"""MPC diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, output_base: str | Path) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df[x], df[y], alpha=0.8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_histogram(df: pd.DataFrame, column: str, title: str, output_base: str | Path) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df[column], bins=min(10, max(1, len(df))))
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
