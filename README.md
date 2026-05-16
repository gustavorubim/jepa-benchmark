# JEPA Robotics Benchmark

This repository compares classic model-free reinforcement learning baselines with JEPA-style latent predictive world models on goal-conditioned robotics tasks. The first maintained path is state-based Fetch reaching/pushing, with a deterministic `ToyGoal-v0` smoke path for fast CI and local validation.

## Install

Use `uv` for all environment management:

```bash
uv sync
```

The project targets Python `>=3.11,<3.13` and uses a local `.venv` created by `uv`.

## Quickstart Smoke Run

These commands run on CPU without CUDA/MPS and use tiny budgets:

```bash
uv run python -m jepa_robotics.cli.run_suite \
  --mode iteration \
  --phases phase1_fetch_reach \
  --max-parallel 2 \
  --smoke-test

uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.collect_dataset --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.train_jepa --config configs/jepa/state_jepa.yaml --smoke-test
uv run python -m jepa_robotics.cli.train_autoencoder --config configs/jepa/autoencoder.yaml --smoke-test
uv run python -m jepa_robotics.cli.evaluate --config configs/jepa/jepa_mpc.yaml --smoke-test
uv run python -m jepa_robotics.cli.plot --experiment outputs/smoke
uv run python -m jepa_robotics.cli.analyze --experiment outputs/smoke
```

The top-level module aliases also work for the core commands, for example:

```bash
uv run python -m jepa_robotics.train_rl --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.train_jepa --config configs/jepa/state_jepa.yaml --smoke-test
uv run python -m jepa_robotics.evaluate --config configs/jepa/jepa_mpc.yaml --smoke-test
uv run python -m jepa_robotics.plot --experiment outputs/smoke
```

## Experiment Suites and Runtime Tiers

The suite driver launches one subprocess per `(phase, seed)` pair, caps concurrency with
`--max-parallel`, records suite metadata, and skips completed children by default:

```bash
uv run python -m jepa_robotics.cli.run_suite \
  --mode iteration \
  --phases phase1_fetch_reach phase3_fetch_push_sparse \
  --max-parallel 2
```

Suites default to `--seeds 0`. Pass explicit seeds, for example `--seeds 0 1 2`,
when you want confirm or publishable multi-seed evidence. Use `--force` to rerun completed
outputs. Status and provenance files are written under:

```text
outputs/<mode>_suite/
  suite_metadata.json
  suite_status/<phase>_seed_<seed>.json
  suite_configs/<phase>_seed_<seed>.yaml
```

Runtime tiers live in:

```text
configs/experiments/iteration/
configs/experiments/confirm/
configs/experiments/full/
```

Iteration configs are for local development, confirm configs are for stronger single-machine
evidence, and full configs are resumable multi-seed runs for reporting.

## RL Baselines

Dense Fetch tasks use SAC. Sparse goal-conditioned Fetch tasks use SAC with HER replay:

```bash
uv run python -m jepa_robotics.cli.train_rl \
  --config configs/experiments/phase1_fetch_reach.yaml \
  --method sac \
  --seed 0

uv run python -m jepa_robotics.cli.train_rl \
  --config configs/experiments/phase3_fetch_push_sparse.yaml \
  --method sac_her \
  --seed 0
```

RL outputs are written under:

```text
outputs/<experiment>/rl/<method>/seed_<seed>/
  model.zip
  monitor.csv
  eval_metrics.csv
  metrics.csv
  config_resolved.yaml
  stdout.log
  stderr.log
```

RL configs support speed controls:

```yaml
rl:
  n_envs: 2
  vec_env_type: dummy
  train_eval_episodes: 5
  final_eval_episodes: 20
  early_stop_success: 0.95
  early_stop_patience: 3
  skip_existing: true
```

Training-time evaluations use `train_eval_episodes`; the final row uses
`final_eval_episodes`. `metrics.csv` records wall-clock timing, eval seconds, steps/sec,
and early-stop state.

## Dataset Collection

Collect trajectories from a random, RL, or mixture policy:

```bash
uv run python -m jepa_robotics.cli.collect_dataset \
  --config configs/experiments/phase1_fetch_reach.yaml \
  --policy outputs/phase1_fetch_reach/rl/sac/seed_0/model.zip \
  --seed 0
```

Datasets use a documented compressed NPZ schema plus `metadata.json`:

```text
observations, achieved_goals, desired_goals, actions, rewards,
terminated, truncated, episode_ids, timestep_ids
```

Random-policy collection can use multiple vectorized workers:

```yaml
dataset:
  n_envs: 4
  vec_env_type: dummy
  skip_existing: true
```

Window generation for JEPA training is vectorized and rejects windows that cross episode,
terminal, or truncation boundaries.

## Train JEPA

```bash
uv run python -m jepa_robotics.cli.train_jepa \
  --config configs/jepa/state_jepa.yaml \
  --dataset outputs/phase1_fetch_reach/datasets/FetchReachDense-v3/mixture/seed_0/trajectories.npz \
  --seed 0
```

JEPA outputs:

```text
encoder.pt
target_encoder.pt
predictor.pt
normalizer.pkl
train_metrics.csv
val_metrics.csv
config_resolved.yaml
```

`normalizer.pkl` stores `schema_version=normalizer_v1` and a `RunningNormalizer` object.
`train_metrics.csv` and `val_metrics.csv` are split by held-out episode IDs and include
per-horizon cosine similarity and prediction MSE columns such as
`cosine_similarity_h1` and `prediction_mse_h4`.

JEPA configs also expose DataLoader and optional accelerator knobs:

```yaml
jepa:
  validation_fraction: 0.2
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  compile_model: false
  amp: false
```

AMP and compile are off by default.

## Train Autoencoder Baseline

```bash
uv run python -m jepa_robotics.cli.train_autoencoder \
  --config configs/jepa/autoencoder.yaml \
  --dataset outputs/state_jepa/datasets/FetchReachDense-v3/random/seed_0/trajectories.npz \
  --seed 0
```

Autoencoder outputs:

```text
encoder.pt
decoder.pt
predictor.pt
train_metrics.csv
val_metrics.csv
config_resolved.yaml
```

## Evaluate JEPA-MPC

```bash
uv run python -m jepa_robotics.cli.evaluate \
  --config configs/jepa/jepa_mpc.yaml \
  --checkpoint outputs/state_jepa/jepa/state_jepa/seed_0/encoder.pt \
  --dataset outputs/state_jepa/datasets/FetchReachDense-v3/random/seed_0/trajectories.npz \
  --seed 0
```

The planner supports random shooting and CEM. It logs planning time, selected action norm, predicted latent distance, actual goal distance, reward, and success metrics.
It also records action smoothness, candidate score summary statistics, actual goal progress,
and predicted-vs-actual progress correlation in `mpc_diagnostics.csv` and
`mpc_summary.csv`.

## JEPA-Pretrained SAC

SAC can be run with a frozen or fine-tuned JEPA encoder feature extractor:

```bash
uv run python -m jepa_robotics.cli.train_rl \
  --config configs/experiments/phase1_fetch_reach.yaml \
  --method sac_jepa \
  --feature-extractor jepa \
  --encoder-checkpoint outputs/state_jepa/jepa/state_jepa/seed_0/encoder.pt \
  --freeze-encoder \
  --seed 0
```

Use `--fine-tune-encoder` instead of `--freeze-encoder` to allow policy gradients to update
the encoder.

## Plot and Analyze

Plots are generated from saved CSV logs, never from live training objects:

```bash
uv run python -m jepa_robotics.cli.plot --experiment outputs/phase1_fetch_reach
uv run python -m jepa_robotics.cli.analyze --experiment outputs/phase1_fetch_reach
```

Reports are written under:

```text
reports/<experiment>/
  report.md
  metrics_combined.csv
  plots/*.png
  plots/*.pdf
  tables/aggregate_metrics.csv
  tables/threshold_metrics.csv
  tables/statistical_tests.csv
```

`analyze` writes:

```text
tables/summary_statistics.csv
tables/threshold_fractions.csv
report.md
```

The summary table includes mean, SE, and bootstrap 95% CI by method. Statistical tests include
paired t-test, Wilcoxon fallback, and effect-size columns where enough paired seeds are present.

## Configs

All runs are config-driven. Key configs live in:

```text
configs/env/
configs/rl/
configs/jepa/
configs/experiments/
```

Each run copies the resolved config into its output directory and does not mutate the source YAML.

## Device Handling

All PyTorch code uses:

```python
get_torch_device(preferred: str = "auto")
```

`auto` tries CUDA, then MPS, then CPU. Explicit `cuda` or `mps` requests fail clearly if unavailable.

## Quality Gates

Run the required gate before considering changes complete:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
uv run mypy src tests
```

Apply formatting with:

```bash
uv run ruff format .
```

## Full Experiment Scripts

The shell scripts preserve the intended real-run workflow:

```bash
bash scripts/train_baselines.sh
bash scripts/train_jepa.sh
bash scripts/make_all_plots.sh
```

## Performance Measurement

Smoke-scale measurements for this enhancement work can be regenerated with:

```bash
uv run python scripts/benchmark_training_enhancements.py
```

The script writes:

```text
reports/training_performance_enhancements/measurement.md
reports/training_performance_enhancements/measurement.json
```

These are local smoke measurements. Use the suite driver in confirm or full mode for real Fetch
runtime and quality claims.

## Troubleshooting

MuJoCo and Gymnasium Robotics: run `uv sync` and confirm the Fetch env can be imported with `uv run python -c "import gymnasium_robotics"`.

Current Gymnasium Robotics releases deprecate the spec-era Fetch `-v3` IDs in favor of `-v4`. The environment factory accepts the repo configs as written and resolves Fetch `-v3` IDs to installed `-v4` environments at runtime.

CUDA: use `device.preferred: cuda` only when `torch.cuda.is_available()` is true.

MPS: use `device.preferred: mps` only on Apple Silicon with PyTorch MPS support. Use `auto` for CUDA -> MPS -> CPU selection.

Long training times: start with `--smoke-test`, then `configs/experiments/phase0_reacher_debug.yaml`, then the real phase configs.
