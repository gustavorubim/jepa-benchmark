"""Sample-efficiency plots and tables."""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LONG_LABEL_THRESHOLD = 24
WRAP_WIDTH = 34


def plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    output_base: str | Path,
    label_columns: list[str] | None = None,
) -> None:
    output = Path(output_base)
    output.parent.mkdir(parents=True, exist_ok=True)
    if label_columns:
        labels = df[label_columns].astype(str).agg(" / ".join, axis=1)
        summary = pd.DataFrame({"label": labels, y: df[y]}).groupby("label")[y].mean().reset_index()
        x_label = " / ".join(label_columns)
    else:
        summary = df.groupby(x, dropna=False)[y].mean().reset_index()
        summary = summary.rename(columns={x: "label"})
        x_label = x
    summary = summary.dropna(subset=[y]).copy()
    if summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No finite values for {y}", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output.with_suffix(".png"))
        fig.savefig(output.with_suffix(".pdf"))
        plt.close(fig)
        return
    summary["label"] = summary["label"].astype(str).map(_readable_bar_label)
    use_horizontal = len(summary) > 4 or summary["label"].map(len).max() > LONG_LABEL_THRESHOLD
    height = max(4.0, 0.75 * len(summary) + 1.8)
    if use_horizontal:
        fig, ax = plt.subplots(figsize=(8, height))
        summary = summary.sort_values(y, ascending=True)
        ax.barh(summary["label"], summary[y])
        ax.set_xlabel(y)
        ax.set_ylabel(x_label)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(summary["label"], summary[y])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"))
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _readable_bar_label(label: str) -> str:
    parts = [part.strip() for part in label.split("|")]
    if len(parts) == 3:
        group, env_id, method = parts
        label = f"{method} | {env_id} | {group}"
    return "\n".join(wrap(label, width=WRAP_WIDTH, break_long_words=False))
