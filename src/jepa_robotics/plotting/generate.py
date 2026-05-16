"""Generate artifact-driven reports from saved experiment logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jepa_robotics.config.load import load_config
from jepa_robotics.plotting.learning_curves import plot_learning_curve
from jepa_robotics.plotting.mpc_diagnostics import plot_histogram, plot_scatter
from jepa_robotics.plotting.representation_plots import plot_probe_bars
from jepa_robotics.plotting.sample_efficiency import plot_bar
from jepa_robotics.plotting.tables import write_analysis_tables

PRETRAINING_BUDGET_METHODS: frozenset[str] = frozenset(
    {
        "jepa_mpc",
        "jepa_sac",
        "sac_jepa",
        "state_jepa",
        "autoencoder",
        "state_autoencoder",
        "sac_ae",
        "ae_mpc",
    }
)
PRETRAINING_BUDGET_PHASES: frozenset[str] = frozenset(
    {
        "jepa_mpc",
        "jepa_sac",
        "state_jepa",
        "autoencoder",
    }
)
POLICY_SOURCED_DATASETS: frozenset[str] = frozenset({"rl_policy", "mixture"})

CORE_PLOT_BASES = [
    "sample_efficiency_success",
    "sample_efficiency_reward",
    "steps_to_threshold_bar",
    "auc_success_bar",
    "final_success_boxplot",
    "reward_boxplot",
]

OPTIONAL_PLOT_BASES = [
    "jepa_loss_curves",
    "jepa_horizon_cosine_similarity",
    "latent_std_over_time",
    "prediction_error_by_horizon",
    "probe_r2_bar",
    "probe_mse_bar",
    "mpc_predicted_vs_actual_progress",
    "mpc_planning_time_histogram",
    "generalization_success_heatmap",
    "generalization_reward_heatmap",
]

REQUIRED_PLOT_BASES = CORE_PLOT_BASES


def load_experiment_metrics(experiment_path: str | Path) -> pd.DataFrame:
    path = Path(experiment_path)
    frames: list[pd.DataFrame] = []
    for csv_path in _metric_csv_paths(path):
        frame = pd.read_csv(csv_path)
        if not {"global_step", "method", "seed"}.issubset(frame.columns):
            continue
        frames.append(_enrich_metric_frame(frame, csv_path, path))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        duplicate_columns = [column for column in combined.columns if column != "source_path"]
        return combined.drop_duplicates(subset=duplicate_columns)
    return pd.DataFrame(
        columns=[
            "phase",
            "env_id",
            "method",
            "seed",
            "reward_mode",
            "total_steps",
            "source_path",
            "global_step",
            "mean_episode_reward",
            "mean_success_rate",
        ]
    )


def generate_report_artifacts(experiment_path: str | Path) -> Path:
    experiment = Path(experiment_path)
    experiment_id = experiment.name
    reports_dir = Path("reports") / experiment_id
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    _clear_known_plots(plots_dir)

    manifest: list[dict[str, str]] = []
    metrics = load_experiment_metrics(experiment)
    metrics.to_csv(reports_dir / "metrics_combined.csv", index=False)
    aggregate_paths = write_analysis_tables(metrics, reports_dir)
    aggregate = pd.read_csv(aggregate_paths["aggregate"])
    aggregate = _with_comparison_label(aggregate)
    metrics = _with_comparison_label(metrics)

    _plot_core_metrics(metrics, aggregate, plots_dir, manifest)
    _plot_jepa_diagnostics(experiment, plots_dir, manifest)
    _plot_probe_outputs(experiment, plots_dir, manifest)
    _plot_mpc_outputs(experiment, plots_dir, manifest)
    _plot_generalization_outputs(experiment, plots_dir, manifest)

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(reports_dir / "artifact_manifest.csv", index=False)
    _write_report(reports_dir, experiment_id, metrics, aggregate, manifest_frame)
    return reports_dir


def _metric_csv_paths(path: Path) -> list[Path]:
    paths: list[Path] = []
    for csv_path in sorted(path.rglob("metrics.csv")) + sorted(path.rglob("eval_metrics.csv")):
        if csv_path.name == "eval_metrics.csv" and (csv_path.parent / "metrics.csv").exists():
            continue
        paths.append(csv_path)
    return paths


def _enrich_metric_frame(
    frame: pd.DataFrame, csv_path: Path, experiment_root: Path
) -> pd.DataFrame:
    enriched = frame.copy()
    rel = _relative_to(csv_path, experiment_root)
    phase = rel.parts[0] if rel.parts else "unknown"
    metadata = _load_metric_metadata(csv_path, experiment_root, phase, enriched)
    enriched["phase"] = enriched.get("phase", phase)
    comparison_group = metadata.get("comparison_group") or phase
    enriched["comparison_group"] = enriched.get("comparison_group", comparison_group)
    enriched["env_id"] = enriched.get("env_id", metadata.get("env_id", "unknown"))
    enriched["reward_mode"] = enriched.get("reward_mode", metadata.get("reward_mode", ""))
    enriched["total_steps"] = enriched.get(
        "total_steps", metadata.get("total_steps", enriched["global_step"].max())
    )
    dataset_source = str(metadata.get("dataset_source", "") or "")
    enriched["dataset_source"] = enriched.get("dataset_source", dataset_source)
    pretraining_env_steps = int(metadata.get("pretraining_env_steps", 0) or 0)
    policy_source_steps = int(metadata.get("policy_source_steps", 0) or 0)
    enriched["pretraining_env_steps"] = enriched.get("pretraining_env_steps", pretraining_env_steps)
    enriched["policy_source_steps"] = enriched.get("policy_source_steps", policy_source_steps)
    method = str(enriched["method"].dropna().iloc[0]) if "method" in enriched else ""
    if _uses_pretraining_budget(phase, method, dataset_source):
        offset = pretraining_env_steps + policy_source_steps
    else:
        offset = 0
    configured_budget = float(metadata.get("total_steps", enriched["global_step"].max())) + float(
        offset
    )
    enriched["configured_environment_budget"] = enriched.get(
        "configured_environment_budget", configured_budget
    )
    enriched["environment_interactions"] = enriched.get(
        "environment_interactions", enriched["global_step"].astype(float) + float(offset)
    )
    enriched["source_path"] = str(rel)
    return enriched


def _load_metric_metadata(
    csv_path: Path, experiment_root: Path, phase: str, frame: pd.DataFrame
) -> dict[str, Any]:
    config_path = _nearest_config(csv_path, experiment_root)
    config = _config_metadata(config_path)
    if not config:
        seed = int(frame["seed"].iloc[0]) if "seed" in frame and not frame.empty else 0
        config = _config_metadata(experiment_root / "suite_configs" / f"{phase}_seed_{seed}.yaml")
    dataset_metadata = _dataset_metadata(experiment_root)
    return {
        "env_id": config.get("env_id", frame["env_id"].iloc[0] if "env_id" in frame else "unknown"),
        "reward_mode": config.get("reward_mode", ""),
        "total_steps": _metric_total_steps(config, phase, frame),
        "dataset_source": config.get("dataset_source", ""),
        "pretraining_env_steps": config.get("pretraining_env_steps", 0),
        "policy_source_steps": dataset_metadata.get("policy_source_steps", 0),
        "comparison_group": config.get("comparison_group") or phase,
    }


def _dataset_metadata(experiment_root: Path) -> dict[str, Any]:
    total: dict[str, Any] = {"policy_source_steps": 0}
    for path in experiment_root.rglob("metadata.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") != "trajectory_npz_v1":
            continue
        total["policy_source_steps"] = max(
            int(total.get("policy_source_steps", 0)),
            int(payload.get("policy_source_steps", 0) or 0),
        )
    return total


def _nearest_config(csv_path: Path, experiment_root: Path) -> Path | None:
    for parent in [csv_path.parent, *csv_path.parents]:
        candidate = parent / "config_resolved.yaml"
        if candidate.exists():
            return candidate
        if parent == experiment_root:
            break
    return None


def _config_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    config = load_config(path)
    return {
        "env_id": config.env.id,
        "comparison_group": config.experiment.comparison_group,
        "reward_mode": config.env.reward_mode or "",
        "total_steps": config.rl.total_timesteps,
        "eval_interaction_budget": config.rl.n_eval_episodes * config.env.max_episode_steps,
        "dataset_source": config.dataset.source,
        "pretraining_env_steps": config.dataset.num_episodes * config.dataset.max_episode_steps,
    }


def _metric_total_steps(config: dict[str, Any], phase: str, frame: pd.DataFrame) -> float:
    if phase == "jepa_mpc":
        return float(config.get("eval_interaction_budget", frame["global_step"].max()))
    return float(config.get("total_steps", frame["global_step"].max()))


def _uses_pretraining_budget(phase: str, method: str, dataset_source: str) -> bool:
    if not dataset_source:
        return False
    method_lower = method.strip().lower()
    phase_lower = phase.strip().lower()
    return method_lower in PRETRAINING_BUDGET_METHODS or phase_lower in PRETRAINING_BUDGET_PHASES


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _with_comparison_label(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    if labeled.empty:
        labeled["comparison"] = []
        return labeled
    for column in ["comparison_group", "phase", "env_id", "method"]:
        if column not in labeled:
            labeled[column] = "unknown"
    labeled["comparison"] = (
        labeled["comparison_group"].astype(str)
        + " | "
        + labeled["env_id"].astype(str)
        + " | "
        + labeled["method"].astype(str)
    )
    return labeled


def _plot_core_metrics(
    metrics: pd.DataFrame, aggregate: pd.DataFrame, plots_dir: Path, manifest: list[dict[str, str]]
) -> None:
    if metrics.empty:
        for base in CORE_PLOT_BASES:
            _record(manifest, base, "missing", "No learning metrics were found.")
        return
    plot_learning_curve(
        metrics,
        "mean_success_rate",
        "mean success rate",
        "Sample efficiency: success",
        plots_dir / "sample_efficiency_success",
        group_columns=["comparison"],
        x_column="environment_interactions",
    )
    _record(manifest, "sample_efficiency_success", "real", "Generated from metrics.csv.")
    plot_learning_curve(
        metrics,
        "mean_episode_reward",
        "mean episode reward",
        "Sample efficiency: reward",
        plots_dir / "sample_efficiency_reward",
        group_columns=["comparison"],
        x_column="environment_interactions",
    )
    _record(manifest, "sample_efficiency_reward", "real", "Generated from metrics.csv.")
    if aggregate.empty:
        for base in CORE_PLOT_BASES[2:]:
            _record(manifest, base, "missing", "No aggregate metrics were produced.")
        return
    for base, column, title in [
        ("steps_to_threshold_bar", "steps_to_success_50", "Steps to 50% success"),
        ("auc_success_bar", "auc_success", "AUC success"),
        ("final_success_boxplot", "final_success_rate", "Final success"),
        ("reward_boxplot", "final_mean_reward", "Final reward"),
    ]:
        plot_bar(aggregate, "comparison", column, title, plots_dir / base)
        _record(manifest, base, "real", "Generated from aggregate_metrics.csv.")


def _plot_jepa_diagnostics(
    experiment: Path, plots_dir: Path, manifest: list[dict[str, str]]
) -> None:
    jepa = _load_jepa_metrics(experiment)
    if jepa.empty:
        for base in [
            "jepa_loss_curves",
            "jepa_horizon_cosine_similarity",
            "latent_std_over_time",
            "prediction_error_by_horizon",
        ]:
            _record(manifest, base, "missing", "No JEPA train_metrics.csv/val_metrics.csv found.")
        return
    jepa["series"] = jepa["phase"].astype(str) + " | seed " + jepa["seed"].astype(str)
    if "split" in jepa:
        jepa["loss_series"] = jepa["series"] + " | " + jepa["split"].astype(str)
    loss_column = "loss" if "loss" in jepa else "train_loss"
    if loss_column in jepa:
        plot_learning_curve(
            jepa,
            loss_column,
            "loss",
            "JEPA loss",
            plots_dir / "jepa_loss_curves",
            group_columns=["loss_series"],
        )
        _record(manifest, "jepa_loss_curves", "real", "Generated from JEPA metric CSVs.")
    _plot_optional_curve(
        jepa,
        "cosine_similarity_h1",
        "cosine similarity",
        "JEPA horizon cosine",
        plots_dir / "jepa_horizon_cosine_similarity",
        manifest,
    )
    _plot_optional_curve(
        jepa,
        "latent_std_mean",
        "latent std",
        "Latent std over time",
        plots_dir / "latent_std_over_time",
        manifest,
    )
    horizon = _jepa_horizon_errors(jepa)
    if horizon.empty:
        _record(
            manifest,
            "prediction_error_by_horizon",
            "missing",
            "No prediction_mse_h* columns found in JEPA metrics.",
        )
    else:
        plot_bar(
            horizon,
            "horizon",
            "prediction_error",
            "Prediction error by horizon",
            plots_dir / "prediction_error_by_horizon",
        )
        _record(
            manifest,
            "prediction_error_by_horizon",
            "real",
            "Generated from JEPA prediction_mse_h* columns.",
        )


def _load_jepa_metrics(experiment: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(experiment.rglob("train_metrics.csv")) + sorted(
        experiment.rglob("val_metrics.csv")
    ):
        frame = pd.read_csv(csv_path)
        if "method" not in frame or not frame["method"].astype(str).str.contains("jepa").any():
            continue
        rel = _relative_to(csv_path, experiment)
        frame["phase"] = rel.parts[0] if rel.parts else "unknown"
        frame["source_path"] = str(rel)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_optional_curve(
    frame: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output_base: Path,
    manifest: list[dict[str, str]],
) -> None:
    base = output_base.name
    if column not in frame:
        _record(manifest, base, "missing", f"No `{column}` column found.")
        return
    plot_learning_curve(frame, column, ylabel, title, output_base, group_columns=["series"])
    _record(manifest, base, "real", f"Generated from `{column}`.")


def _jepa_horizon_errors(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in frame.columns:
        if not column.startswith("prediction_mse_h"):
            continue
        rows.append(
            {
                "horizon": column.removeprefix("prediction_mse_"),
                "prediction_error": float(frame[column].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_probe_outputs(experiment: Path, plots_dir: Path, manifest: list[dict[str, str]]) -> None:
    probe = _load_optional_csv(experiment, ["*probe*.csv", "*probes*.csv"])
    if probe.empty or not {"variable", "r2", "mse"}.issubset(probe.columns):
        _record(manifest, "probe_r2_bar", "missing", "No probe CSV with variable/r2/mse found.")
        _record(manifest, "probe_mse_bar", "missing", "No probe CSV with variable/r2/mse found.")
        return
    plot_probe_bars(probe, "r2", plots_dir / "probe_r2_bar")
    plot_probe_bars(probe, "mse", plots_dir / "probe_mse_bar")
    _record(manifest, "probe_r2_bar", "real", "Generated from probe CSV.")
    _record(manifest, "probe_mse_bar", "real", "Generated from probe CSV.")


def _plot_mpc_outputs(experiment: Path, plots_dir: Path, manifest: list[dict[str, str]]) -> None:
    mpc = _load_optional_csv(experiment, ["mpc_diagnostics.csv"])
    if mpc.empty:
        _record(
            manifest, "mpc_predicted_vs_actual_progress", "missing", "No mpc_diagnostics.csv found."
        )
        _record(manifest, "mpc_planning_time_histogram", "missing", "No mpc_diagnostics.csv found.")
        return
    if {"predicted_latent_progress", "actual_goal_progress"}.issubset(mpc.columns):
        plot_scatter(
            mpc,
            "predicted_latent_progress",
            "actual_goal_progress",
            "MPC predicted vs actual progress",
            plots_dir / "mpc_predicted_vs_actual_progress",
        )
        _record(
            manifest,
            "mpc_predicted_vs_actual_progress",
            "real",
            "Generated from mpc_diagnostics.csv.",
        )
    else:
        _record(
            manifest,
            "mpc_predicted_vs_actual_progress",
            "missing",
            "MPC diagnostics lack predicted/actual progress columns.",
        )
    if "planning_time_ms" in mpc:
        plot_histogram(
            mpc,
            "planning_time_ms",
            "MPC planning time",
            plots_dir / "mpc_planning_time_histogram",
        )
        _record(
            manifest,
            "mpc_planning_time_histogram",
            "real",
            "Generated from mpc_diagnostics.csv.",
        )
    else:
        _record(
            manifest,
            "mpc_planning_time_histogram",
            "missing",
            "MPC diagnostics lack planning_time_ms.",
        )


def _plot_generalization_outputs(
    experiment: Path, plots_dir: Path, manifest: list[dict[str, str]]
) -> None:
    generalization = _load_optional_csv(experiment, ["*generalization*.csv"])
    required = {"perturbation", "level", "mean_success_rate", "mean_episode_reward"}
    if generalization.empty or not required.issubset(generalization.columns):
        _record(
            manifest,
            "generalization_success_heatmap",
            "missing",
            "No generalization CSV with perturbation/level/success/reward found.",
        )
        _record(
            manifest,
            "generalization_reward_heatmap",
            "missing",
            "No generalization CSV with perturbation/level/success/reward found.",
        )
        return
    _plot_heatmap(
        generalization,
        value_column="mean_success_rate",
        output_base=plots_dir / "generalization_success_heatmap",
        title="Generalization success",
    )
    _plot_heatmap(
        generalization,
        value_column="mean_episode_reward",
        output_base=plots_dir / "generalization_reward_heatmap",
        title="Generalization reward",
    )
    _record(
        manifest, "generalization_success_heatmap", "real", "Generated from generalization CSV."
    )
    _record(manifest, "generalization_reward_heatmap", "real", "Generated from generalization CSV.")


def _load_optional_csv(experiment: Path, patterns: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in patterns:
        for csv_path in sorted(experiment.rglob(pattern)):
            frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_heatmap(df: pd.DataFrame, value_column: str, output_base: Path, title: str) -> None:
    pivot = (
        df.pivot_table(index="perturbation", columns="level", values=value_column, aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), [str(item) for item in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [str(item) for item in pivot.index])
    ax.set_xlabel("level")
    ax.set_ylabel("perturbation")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _record(manifest: list[dict[str, str]], artifact: str, status: str, detail: str) -> None:
    manifest.append({"artifact": artifact, "status": status, "detail": detail})


def _clear_known_plots(plots_dir: Path) -> None:
    for base in [*CORE_PLOT_BASES, *OPTIONAL_PLOT_BASES]:
        for suffix in [".png", ".pdf"]:
            path = plots_dir / f"{base}{suffix}"
            if path.exists():
                path.unlink()


def _write_report(
    reports_dir: Path,
    experiment_id: str,
    metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    real = manifest[manifest["status"] == "real"] if not manifest.empty else pd.DataFrame()
    missing = manifest[manifest["status"] == "missing"] if not manifest.empty else pd.DataFrame()
    best = (
        aggregate.sort_values("auc_success_normalized", ascending=False).head(1)
        if "auc_success_normalized" in aggregate
        else pd.DataFrame()
    )
    verdict = _claim_verdict(aggregate)
    report = f"""# Experiment Report

## Setup
- Experiment: `{experiment_id}`
- Metrics rows: {len(metrics)}
- Aggregate rows: {len(aggregate)}
- Generated from saved CSV logs only.

## Direct Verdict
{verdict}

## Artifact Status
- Real plots/tables: {len(real)}
- Missing or not-run diagnostics: {len(missing)}
- Smoke placeholders: 0

## Main Tables
- `metrics_combined.csv` preserves phase, comparison group, environment, method,
  seed, reward mode, total steps, and source path.
- `tables/aggregate_metrics.csv` groups by phase, env_id, method, and seed.
- `tables/robust_summary.csv` reports mean, median, IQM, and bootstrap IQM CIs.
- `artifact_manifest.csv` records whether each optional diagnostic was real or missing.
- `tables/statistical_tests.csv` is now a comparison-readiness diagnostic, not a
  p-value table.

## Interpretation Rules
- Do not compare methods across different environments or phases as one row.
- Only compare methods across phases when they share an explicit `comparison_group`.
- Use `environment_interactions` for sample-efficiency plots; JEPA-based methods
  include configured dataset collection cost as an x-axis offset.
- Single-seed runs are pipeline checks, not publishable performance evidence.
- Missing diagnostics mean the underlying probe/generalization/MPC/Jepa artifact
  was not produced; the report does not substitute synthetic data.

## Top Aggregate Row
```text
{best.to_string(index=False) if not best.empty else "no data"}
```
"""
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "report.md").write_text(report, encoding="utf-8")


def _budgets_close(budgets: list[float], rtol: float = 0.01) -> bool:
    if len(budgets) <= 1:
        return len(budgets) == 1
    reference = float(budgets[0])
    if reference == 0:
        return all(abs(value) <= 1.0 for value in budgets)
    return all(abs(value - reference) / abs(reference) <= rtol for value in budgets)


def _claim_verdict(aggregate: pd.DataFrame) -> str:
    if aggregate.empty:
        return "No learning metrics were found; no performance claim is supported."
    eligible: list[str] = []
    context_columns = [
        column for column in ["comparison_group", "env_id"] if column in aggregate.columns
    ]
    for keys, group in aggregate.groupby(context_columns or ["env_id"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        context = dict(zip(context_columns or ["env_id"], key_values, strict=True))
        methods = group["method"].nunique()
        seed_counts = group.groupby("method")["seed"].nunique()
        budget_column = (
            "configured_environment_budget"
            if "configured_environment_budget" in group
            else "total_environment_interactions"
            if "total_environment_interactions" in group
            else "total_steps"
        )
        budgets = (
            sorted(group[budget_column].dropna().astype(float).unique())
            if budget_column in group
            else []
        )
        if (
            methods >= 2
            and not seed_counts.empty
            and seed_counts.min() >= 5
            and _budgets_close(budgets)
        ):
            eligible.append("/".join(str(value) for value in context.values()))
    if not eligible:
        return (
            "No winner is reported. The current evidence is unmatched, single-seed, "
            "cross-environment, or budget-inconsistent; it is useful for pipeline validation only."
        )
    return (
        "Matched multi-seed contexts exist for descriptive comparison: "
        + ", ".join(sorted(eligible))
        + ". Inspect stratified summaries before making a performance claim."
    )
