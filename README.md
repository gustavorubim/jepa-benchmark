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
uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.collect_dataset --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.train_jepa --config configs/jepa/state_jepa.yaml --smoke-test
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

## Evaluate JEPA-MPC

```bash
uv run python -m jepa_robotics.cli.evaluate \
  --config configs/jepa/jepa_mpc.yaml \
  --checkpoint outputs/state_jepa/jepa/state_jepa/seed_0/encoder.pt \
  --dataset outputs/state_jepa/datasets/FetchReachDense-v3/random/seed_0/trajectories.npz \
  --seed 0
```

The planner supports random shooting and CEM. It logs planning time, selected action norm, predicted latent distance, actual goal distance, reward, and success metrics.

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

## Troubleshooting

MuJoCo and Gymnasium Robotics: run `uv sync` and confirm the Fetch env can be imported with `uv run python -c "import gymnasium_robotics"`.

Current Gymnasium Robotics releases deprecate the spec-era Fetch `-v3` IDs in favor of `-v4`. The environment factory accepts the repo configs as written and resolves Fetch `-v3` IDs to installed `-v4` environments at runtime.

CUDA: use `device.preferred: cuda` only when `torch.cuda.is_available()` is true.

MPS: use `device.preferred: mps` only on Apple Silicon with PyTorch MPS support. Use `auto` for CUDA -> MPS -> CPU selection.

Long training times: start with `--smoke-test`, then `configs/experiments/phase0_reacher_debug.yaml`, then the real phase configs.
