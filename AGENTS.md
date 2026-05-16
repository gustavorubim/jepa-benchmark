# AGENTS.md

# Coding Agent Instructions for `jepa-robotics`

This file defines standing instructions for any coding agent working on this repository. It complements `spec.md`, which is the primary project specification.

## 1. Core Mission

Build and maintain a modular research codebase for comparing classic reinforcement learning baselines against JEPA-style latent predictive world models on simulated robotics tasks.

The codebase must remain:

- reproducible,
- modular,
- testable,
- easy to run with `uv`,
- compatible with CPU, CUDA, and Apple Silicon MPS,
- documented well enough for a new researcher or coding agent to continue the work.

Do not optimize only for getting a single experiment to run. Optimize for a maintainable research system.

---

## 2. Required Quality Gate for Every Task

For every code change, documentation change, refactor, experiment script, or configuration update, the agent must ensure the following are clean before considering the task complete:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
```

When the task involves type-sensitive code, also run:

```bash
uv run mypy src tests
```

If formatting is not clean, apply:

```bash
uv run ruff format .
```

Then rerun the quality gate.

A task is not complete unless:

1. Ruff lint passes.
2. Ruff formatter check passes.
3. Tests pass.
4. Coverage is at least 90% line coverage.
5. README documentation is updated when behavior, commands, architecture, configs, outputs, or experiment workflow changed.
6. Any new CLI, config, model, dataset format, plotting output, or experiment artifact is documented.

If any gate cannot be run in the current environment, state exactly which command could not be run and why. Do not claim the gate passed unless it actually ran and passed.

---

## 3. Package and Environment Rules

Use `uv` only for dependency and environment management.

Do not introduce `pip install` instructions except as an explanation of what `uv` replaces. Preferred commands:

```bash
uv sync
uv add <package>
uv add --dev <package>
uv run <command>
```

The repo should use:

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- a local `.venv` created by `uv sync`

Do not commit local machine artifacts such as experiment outputs, checkpoints, wandb caches, MuJoCo temporary files, or `.venv`.

---

## 4. Architecture Principles

Follow the architecture in `spec.md`. Keep responsibilities separated:

```text
src/jepa_robotics/
  cli/          command-line entry points
  config/       config schemas and loading
  envs/         Gymnasium Robotics environment creation and wrappers
  rl/           Stable-Baselines3 training and evaluation
  data/         trajectory storage, replay datasets, normalization
  models/       encoders, predictors, JEPA modules, autoencoders, policies
  training/     trainers, losses, optimizers, EMA utilities
  planning/     latent MPC, CEM, action sampling, scoring
  evaluation/   rollouts, metrics, probes, generalization tests
  plotting/     publication-quality plots and tables
  utils/        device, seed, logging, filesystem helpers
```

Do not put large amounts of logic inside CLI files. CLI files should parse configs, call library functions, and exit with clear status.

Prefer small pure functions with tests over large stateful classes. Use classes when they represent models, trainers, or reusable experiment components.

---

## 5. Device Compatibility Rules

All PyTorch code must use the central device utility:

```python
def get_torch_device(preferred: str = "auto") -> torch.device:
    ...
```

Required behavior:

- `auto`: CUDA first, MPS second, CPU last.
- `cuda`: require CUDA and fail clearly if unavailable.
- `mps`: require MPS and fail clearly if unavailable.
- `cpu`: always use CPU.

Do not scatter direct `torch.device("cuda")` or `torch.device("mps")` construction across the codebase.

All tensors created inside trainers, losses, planners, and evaluation code must be created on or moved to the configured device.

Tests must not require CUDA or MPS. Device-specific tests should mock availability where practical.

---

## 6. Reinforcement Learning Rules

Use Stable-Baselines3 for traditional RL baselines.

Primary baselines:

- SAC for dense reward Fetch environments.
- SAC + HER replay for sparse goal-conditioned Fetch environments.
- Optional TD3 or PPO only after SAC/SAC+HER is stable.

The RL module must save:

- trained model checkpoints,
- evaluation logs,
- success rates,
- episode rewards,
- environment steps,
- final config snapshot,
- random seed,
- dependency metadata when practical.

All baseline training must be reproducible from a config file.

Do not hardcode environment names, network sizes, seed values, output directories, or training budgets inside training functions. Put them in configs with reasonable defaults.

---

## 7. JEPA Model Rules

The JEPA implementation should start with state observations before moving to images.

Required components:

1. Online encoder.
2. EMA target encoder.
3. Action-conditioned predictor.
4. Latent normalization.
5. JEPA loss using future target latents with stop-gradient.
6. Configurable prediction horizon.
7. Metrics for latent prediction error and representation quality.

The target encoder must be updated only through EMA, never through direct optimizer gradients.

Tests must verify:

- output shapes,
- gradient flow into online encoder and predictor,
- no gradient flow into target encoder,
- EMA update behavior,
- loss value is finite,
- batched sequence rollout works.

---

## 8. Latent MPC Rules

Latent MPC should be implemented as an optional controller that uses a trained JEPA model.

The planner should support:

- random shooting,
- optional CEM,
- configurable planning horizon,
- configurable number of candidate action sequences,
- action bounds from the environment,
- scoring by latent distance to goal latent,
- optional action smoothness penalty.

The planner should execute only the first action, then replan at the next step.

All planner metrics should be logged:

- best candidate score,
- score distribution,
- selected action norm,
- latent distance to goal,
- realized reward,
- success flag.

---

## 9. Data and Artifact Rules

Use explicit, versioned formats for datasets and experiment outputs.

Recommended output structure:

```text
runs/
  <experiment_name>/
    config.yaml
    metrics.jsonl
    summary.json
    checkpoints/
    plots/
    datasets/
```

Do not rely on undocumented pickle blobs for long-term artifacts. If pickle or PyTorch serialization is used, document the exact schema and version.

Trajectory datasets should include at least:

- observations,
- achieved goals,
- desired goals,
- actions,
- rewards,
- dones,
- truncations,
- infos needed for success metrics,
- environment name,
- seed,
- collection policy metadata.

---

## 10. Plotting and Analysis Requirements

Every experiment comparison should produce plots suitable for analytical comparison.

Required plots:

1. Episode reward vs environment steps.
2. Success rate vs environment steps.
3. Sample efficiency to reach fixed success thresholds.
4. Area under learning curve.
5. Final reward and success rate by seed.
6. Confidence intervals or standard error bands across seeds.
7. JEPA train and validation loss curves.
8. JEPA latent prediction error by horizon.
9. Linear probe performance from latent state to object or goal variables.
10. MPC candidate score distribution.
11. MPC latent distance to goal over time.
12. Generalization performance under environment shifts.
13. Wall-clock training time by method.
14. Parameter count by method.
15. Ablation comparison for latent dimension, horizon, and dataset size.

Plots must be reproducible from saved logs. Plot functions must be testable with synthetic logs and should not require completed long-running experiments.

Use clear axis labels, titles, legends, and units. Save figures in at least PNG, preferably also SVG or PDF.

---

## 11. Testing Rules

Maintain at least 90% line coverage.

Tests should be fast by default. Long-running training tests are not allowed in the default test suite.

Use small synthetic examples and tiny configs for smoke tests.

Recommended test layout:

```text
tests/
  test_device.py
  test_seed.py
  test_config.py
  test_envs.py
  test_storage.py
  test_jepa_model.py
  test_jepa_loss.py
  test_ema.py
  test_latent_mpc.py
  test_metrics.py
  test_plots.py
  test_cli_smoke.py
```

For CLI smoke tests:

- use temporary directories,
- use tiny networks,
- use 2 to 5 environment steps,
- assert that expected output files are created,
- do not require GPU,
- do not require internet.

---

## 12. Documentation Rules

Keep `README.md` up to date at all times.

Update the README whenever any of the following changes:

- setup instructions,
- dependencies,
- CLI commands,
- configs,
- experiment workflow,
- output file structure,
- plotting commands,
- model architecture,
- evaluation metrics,
- known limitations,
- troubleshooting steps.

The README should include:

1. Project summary.
2. Installation with `uv`.
3. Quickstart commands.
4. How to run RL baselines.
5. How to collect datasets.
6. How to train JEPA.
7. How to run JEPA-MPC evaluation.
8. How to generate plots.
9. How to run tests and quality gates.
10. Description of output artifacts.
11. Troubleshooting for MuJoCo, Gymnasium Robotics, CUDA, and MPS.

Do not leave stale commands in the README.

---

## 13. Style Rules

Use clear, typed Python.

Preferred conventions:

- type annotations for public functions,
- dataclasses or Pydantic models for configs,
- pathlib instead of raw string paths,
- explicit error messages,
- structured logging,
- JSONL for metrics logs,
- YAML for experiment configs,
- small functions with clear tests.

Avoid:

- hidden global mutable state,
- hardcoded absolute paths,
- notebook-only workflows,
- untested plotting code,
- duplicated device logic,
- large CLI scripts with embedded training logic,
- silent failure when an optional dependency is missing.

---

## 14. Config Rules

All experiments must be driven by config files.

A config should define:

- environment name,
- observation mode,
- reward mode,
- model architecture,
- optimizer settings,
- training budget,
- evaluation frequency,
- random seeds,
- output directory,
- device preference,
- logging options.

When a run starts, copy the fully resolved config to the run directory.

Never mutate the original config file during a run.

---

## 15. Experiment Integrity Rules

Do not compare methods using different training budgets unless the difference is explicit and justified.

For fair comparisons, record and report:

- environment steps,
- gradient steps,
- wall-clock time,
- number of parameters,
- dataset size,
- random seed,
- reward type,
- observation type,
- evaluation protocol.

When reporting results, aggregate across seeds. Do not present a single lucky seed as representative.

---

## 16. Completion Checklist

Before declaring any task complete, verify:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
```

And, when applicable:

```bash
uv run mypy src tests
```

Also verify:

- README is updated.
- Config examples still work.
- New files are included in tests where appropriate.
- Generated artifacts are ignored by git unless intentionally versioned.
- Public functions have useful docstrings when their behavior is non-obvious.
- Errors are clear and actionable.

