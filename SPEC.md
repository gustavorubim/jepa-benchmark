# spec.md

# Project: Small-Scale JEPA World Model for Goal-Conditioned Robotic Manipulation

## 1. Objective

Build a modular research codebase that compares:

1. **Traditional model-free RL** using Stable-Baselines3.
2. **JEPA-style latent predictive world modeling** using PyTorch.
3. Optional ablations using autoencoder dynamics and latent MPC.

The core benchmark is robotic goal-reaching/pushing in existing simulation environments, with primary focus on:

- `FetchReachDense-v4` as the sanity-check environment.
- `FetchPushDense-v4` as the main dense-reward experiment.
- `FetchPushDense-v4` as the primary contact-control Phase 3 target.
- `FetchPush-v4` as a sparse-reward stress test, not the primary benchmark gate.
- Optional later extension: `FetchPickAndPlaceDense-v4`.

The project should answer:

> Can a small JEPA-style latent predictive model improve sample efficiency, representation quality, or generalization compared with classic RL baselines on simulated robotics tasks?

This is not intended to claim novelty over large-scale V-JEPA-style robot planners. It is intended to be a controlled, reproducible, small-scale benchmark.

---

## 2. Non-Negotiable Engineering Requirements

### 2.1 Package and environment management

Use `uv` as the package manager.

The repo must support:

```bash
uv sync
uv run pytest
uv run python -m jepa_robotics.train_rl ...
uv run python -m jepa_robotics.train_jepa ...
uv run python -m jepa_robotics.evaluate ...
uv run python -m jepa_robotics.plot ...
```

Use a standard `pyproject.toml` and `uv.lock`.

The project should create and use a `.venv` automatically through `uv sync`.

### 2.2 Python version

Use:

```toml
requires-python = ">=3.11,<3.13"
```

Python 3.11 is preferred for maximum compatibility with MuJoCo, Gymnasium Robotics, Stable-Baselines3, and PyTorch.

### 2.3 Hardware support

The code must run on:

- CPU
- CUDA
- Apple Silicon MPS

Implement a single utility function:

```python
def get_torch_device(preferred: str = "auto") -> torch.device:
    ...
```

Behavior:

1. If `preferred == "cuda"`, require CUDA or fail clearly.
2. If `preferred == "mps"`, require MPS or fail clearly.
3. If `preferred == "cpu"`, use CPU.
4. If `preferred == "auto"`:
   - use CUDA if available,
   - else use MPS if available,
   - else use CPU.

All PyTorch modules, tensors, losses, and evaluation code must use this device utility.

### 2.4 Reproducibility

Implement:

```python
def set_global_seed(seed: int) -> None:
    ...
```

It must seed:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA if available
- Gymnasium environments at reset time

Every experiment config must include a seed.

Default seeds:

```yaml
seeds: [0, 1, 2, 3, 4]
```

### 2.5 Testing

The project must target **at least 90% line coverage**.

Use:

- `pytest`
- `pytest-cov`
- `ruff`
- `mypy` or `pyright`, preferably `mypy`

CI-style command:

```bash
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
```

Minimum test categories:

1. Unit tests for device selection.
2. Unit tests for config loading.
3. Unit tests for environment wrappers.
4. Unit tests for replay/dataset serialization.
5. Unit tests for JEPA loss shapes and gradient flow.
6. Unit tests for EMA target encoder update.
7. Unit tests for latent rollout predictor.
8. Unit tests for metric aggregation.
9. Unit tests for plotting functions using synthetic logs.
10. Smoke tests for each CLI entry point using very small runs.

Long training jobs should not run in unit tests. Use smoke-test settings with tiny networks and 2 to 5 environment steps.

---

## 3. Recommended Repository Layout

```text
jepa-robotics/
  README.md
  spec.md
  pyproject.toml
  uv.lock
  .python-version
  .gitignore

  configs/
    env/
      fetch_reach_dense.yaml
      fetch_push_dense.yaml
      fetch_push_sparse.yaml
    rl/
      sac.yaml
      sac_her.yaml
      td3.yaml
    jepa/
      state_jepa.yaml
      visual_jepa.yaml
      jepa_mpc.yaml
    experiments/
      phase0_reacher_debug.yaml
      phase1_fetch_reach.yaml
      phase2_fetch_push_dense.yaml
      phase3_fetch_push_dense.yaml
      phase3_fetch_push_sparse.yaml
      phase4_generalization.yaml

  src/
    jepa_robotics/
      __init__.py

      cli/
        train_rl.py
        collect_dataset.py
        train_jepa.py
        train_jepa_policy.py
        evaluate.py
        plot.py
        run_experiment.py

      config/
        schema.py
        load.py

      envs/
        make_env.py
        wrappers.py
        reward_wrappers.py
        observation_wrappers.py
        vector_env.py

      rl/
        sb3_train.py
        sb3_eval.py
        callbacks.py
        her.py

      data/
        replay_buffer.py
        trajectory_dataset.py
        storage.py
        normalization.py
        transforms.py

      models/
        encoders.py
        predictors.py
        jepa.py
        autoencoder.py
        policies.py
        mlp.py

      training/
        jepa_trainer.py
        autoencoder_trainer.py
        policy_trainer.py
        losses.py
        optim.py
        ema.py

      planning/
        latent_mpc.py
        cem.py
        action_sampling.py
        scoring.py

      evaluation/
        rollouts.py
        metrics.py
        probes.py
        generalization.py

      plotting/
        learning_curves.py
        sample_efficiency.py
        representation_plots.py
        mpc_diagnostics.py
        tables.py

      utils/
        device.py
        seed.py
        logging.py
        paths.py
        timers.py
        serialization.py

  tests/
    test_device.py
    test_seed.py
    test_config.py
    test_make_env.py
    test_wrappers.py
    test_dataset.py
    test_storage.py
    test_normalization.py
    test_encoders.py
    test_predictors.py
    test_jepa_loss.py
    test_ema.py
    test_latent_mpc.py
    test_metrics.py
    test_plots.py
    test_cli_smoke.py

  scripts/
    install.sh
    lint.sh
    test.sh
    train_baselines.sh
    train_jepa.sh
    make_all_plots.sh

  outputs/
    .gitkeep

  reports/
    .gitkeep
```

---

## 4. Dependencies

### 4.1 Core dependencies

Add approximately the following:

```bash
uv add numpy pandas scipy matplotlib seaborn tqdm rich pydantic pyyaml typer
uv add torch torchvision
uv add gymnasium gymnasium-robotics mujoco
uv add stable-baselines3 tensorboard
uv add scikit-learn
```

### 4.2 Development dependencies

```bash
uv add --dev pytest pytest-cov ruff mypy pre-commit
```

### 4.3 Optional dependencies

```bash
uv add wandb
uv add imageio opencv-python
```

Weights & Biases should be optional. The code must work without it.

---

## 5. Environment Design

### 5.1 Main environments

The code should support at least these environments:

```text
FetchReachDense-v4
FetchPushDense-v4
FetchPush-v4
```

The environment factory should accept:

```python
make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
    obs_mode: str = "state",
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
)
```

### 5.2 Observation modes

Support two observation modes:

#### State mode

Uses the environment's dictionary observation:

```python
{
  "observation": np.ndarray,
  "achieved_goal": np.ndarray,
  "desired_goal": np.ndarray,
}
```

Flattened model input:

```text
x_t = concat(observation, achieved_goal, desired_goal)
```

For JEPA world modeling, maintain access to the decomposed pieces.

#### Visual mode, optional but architecturally supported

Uses RGB rendering:

```python
{
  "image": np.ndarray[H, W, 3],
  "state": original_env_observation_dict,
}
```

Default image size:

```yaml
image_size: 84
```

Visual mode is an extension. State-based JEPA must work first.

---

## 6. Reward Definitions and Success Criteria

### 6.1 Environment rewards

Use the native rewards for primary environment evaluation.

Dense environments:

- FetchReachDense-v4
- FetchPushDense-v4

Sparse environments:

- FetchPush-v4

Do not redefine the official evaluation reward unless the experiment explicitly says so.

### 6.2 Auxiliary shaped rewards

For exploratory training and ablations, implement optional reward wrappers.

These wrappers must be clearly labeled and never mixed with official evaluation metrics.

#### Reach shaped reward

For reaching:

```text
r_reach = - || achieved_goal - desired_goal ||_2
```

Success:

```text
success = 1[ || achieved_goal - desired_goal ||_2 < success_threshold ]
```

Default:

```yaml
success_threshold: 0.05
```

#### Push shaped reward

For pushing:

```text
d_obj_goal = || object_position - desired_goal ||_2
d_grip_obj = || gripper_position - object_position ||_2

r_push = - d_obj_goal - alpha * d_grip_obj
```

Default:

```yaml
alpha: 0.1
success_threshold: 0.05
```

Use this only for ablation and debugging.

### 6.3 JEPA control reward / scoring function

For latent MPC, do not call it a reward unless it is used for RL. Use "score" or "cost."

Goal-latent cost:

```text
cost = || normalize(z_pred_t+k) - normalize(z_goal) ||_2^2
```

Optionally add action smoothness:

```text
cost_total = cost + lambda_action * sum_t || a_t ||_2^2
```

Default:

```yaml
lambda_action: 0.001
```

### 6.4 Target performance thresholds

The coding agent should implement the code to measure these thresholds, not hard-code claims that they will be achieved.

Recommended experimental goals:

#### Phase 1: FetchReachDense-v4

Baseline SAC should reach:

```text
mean_success_rate >= 0.90
```

within a modest compute budget.

JEPA-MPC should reach:

```text
mean_success_rate >= 0.70
```

after training on a collected dataset.

#### Phase 2: FetchPushDense-v4

Baseline SAC should target:

```text
mean_success_rate >= 0.70
```

JEPA-pretrained policy should target:

```text
equal or better sample efficiency than SAC from scratch
```

Primary comparison:

```text
environment steps required to reach mean_success_rate >= 0.50
```

#### Phase 3: FetchPushDense-v4 primary contact-control task

Dense FetchPush is the default Phase 3 benchmark because it preserves object contact and pushing
dynamics while providing enough shaped signal for reliable single-machine comparisons.

SAC should target:

```text
mean_success_rate >= 0.70
```

Primary comparison:

```text
area under success-rate curve
steps to reach 0.25, 0.50, 0.75 success
final success after fixed budget
```

#### Sparse stress test: FetchPush-v4

SAC+HER should be the traditional baseline.

Primary comparison:

```text
area under success-rate curve
steps to reach 0.25, 0.50, 0.75 success
final success after fixed budget
```

---

## 7. Models

## 7.1 Traditional RL baseline

Use Stable-Baselines3.

### Dense reward baseline

Use SAC:

```python
SAC(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
)
```

Use configs rather than hard-coding.

### Sparse reward baseline

Use SAC + HER replay buffer.

Required behavior:

- Use `MultiInputPolicy`.
- Use `HerReplayBuffer`.
- Use the native goal-conditioned observation dict.
- Do not flatten observations before HER.

Config example:

```yaml
algorithm: SAC
policy: MultiInputPolicy
replay_buffer_class: HerReplayBuffer
replay_buffer_kwargs:
  n_sampled_goal: 4
  goal_selection_strategy: future
  copy_info_dict: true
gamma: 0.98
batch_size: 256
learning_rate: 0.0003
```

### RL training outputs

For each run, save:

```text
outputs/{experiment_id}/rl/{method}/seed_{seed}/
  model.zip
  monitor.csv
  eval_metrics.csv
  config_resolved.yaml
  training_stdout.log
```

Evaluation should run every fixed number of environment steps.

Default:

```yaml
eval_freq: 10000
n_eval_episodes: 20
```

---

## 7.2 State JEPA model

### Purpose

Learn action-conditioned latent dynamics without reconstructing observations.

### Inputs

At each time step:

```text
x_t = concat(observation_t, achieved_goal_t)
a_t = action_t
g = desired_goal
```

For world-model training, desired goal is optional. The default JEPA model should predict future achieved state representation, not the command goal.

### Encoder

```python
StateEncoder(
    input_dim: int,
    latent_dim: int = 128,
    hidden_dims: list[int] = [256, 256],
    activation: "relu" | "silu" = "silu",
    layer_norm: bool = True,
)
```

Output:

```text
z_t in R^latent_dim
```

Normalize using either:

- layer norm inside encoder,
- final `F.normalize(z, dim=-1)`,
- or both, controlled by config.

### Target encoder

Maintain a target encoder as an EMA copy of the online encoder.

EMA update:

```text
theta_target <- m * theta_target + (1 - m) * theta_online
```

Default:

```yaml
ema_momentum: 0.996
```

Target encoder is stop-gradient only.

### Predictor

Action-conditioned latent predictor:

```python
ActionConditionedPredictor(
    latent_dim: int = 128,
    action_dim: int = 4,
    hidden_dims: list[int] = [256, 256],
    horizon: int = 1,
)
```

Two supported modes:

#### One-step predictor

```text
z_hat_{t+1} = p(z_t, a_t)
```

#### Multi-step recurrent predictor

```text
z_hat_{t+1} = p(z_t, a_t)
z_hat_{t+2} = p(z_hat_{t+1}, a_{t+1})
...
z_hat_{t+k} = p(z_hat_{t+k-1}, a_{t+k-1})
```

Default horizons:

```yaml
prediction_horizons: [1, 2, 4, 8]
```

### JEPA loss

For a batch of trajectory windows:

```text
z_t = online_encoder(x_t)
z_target_{t+h} = stopgrad(target_encoder(x_{t+h}))
z_pred_{t+h} = predictor.rollout(z_t, actions[t:t+h])
```

Loss:

```text
L_jepa = mean_h cosine_distance(normalize(z_pred_{t+h}), normalize(z_target_{t+h}))
```

Where:

```text
cosine_distance(a, b) = 2 - 2 * cosine_similarity(a, b)
```

Optional variance regularization to prevent collapse:

```text
L_var = mean(max(0, gamma - std(z))^2)
```

Optional covariance regularization:

```text
L_cov = off_diag_covariance_penalty(z)
```

Total:

```text
L = L_jepa + lambda_var * L_var + lambda_cov * L_cov
```

Default:

```yaml
lambda_var: 1.0
lambda_cov: 0.04
variance_gamma: 1.0
```

If collapse is not observed, allow these to be disabled.

### JEPA training outputs

```text
outputs/{experiment_id}/jepa/{method}/seed_{seed}/
  encoder.pt
  target_encoder.pt
  predictor.pt
  normalizer.pkl
  train_metrics.csv
  val_metrics.csv
  config_resolved.yaml
```

Metrics per epoch:

```text
train_loss
val_loss
cosine_similarity_h1
cosine_similarity_h2
cosine_similarity_h4
cosine_similarity_h8
latent_std_mean
latent_std_min
latent_norm_mean
learning_rate
```

---

## 7.3 Autoencoder dynamics baseline

Purpose: compare JEPA against a reconstruction-based latent model.

Architecture:

```text
encoder(x_t) -> z_t
decoder(z_t) -> x_hat_t
predictor(z_t, a_t) -> z_hat_t+1
decoder(z_hat_t+1) -> x_hat_t+1
```

Loss:

```text
L = reconstruction_loss(x_t, x_hat_t)
  + beta * reconstruction_loss(x_t+1, x_hat_t+1)
  + eta * latent_prediction_loss(z_hat_t+1, stopgrad(z_t+1))
```

Default:

```yaml
beta: 1.0
eta: 1.0
```

Use same encoder and predictor sizes as JEPA for fairness.

---

## 7.4 JEPA-pretrained RL policy

Purpose: test whether JEPA representation improves policy sample efficiency.

Two modes:

### Frozen encoder

```text
z_t = frozen_encoder(x_t)
policy_input = concat(z_t, desired_goal)
```

Train SAC or TD3 on top.

### Fine-tuned encoder

Use the JEPA encoder as initialization, then allow policy gradients to update it.

Implementation note:

Stable-Baselines3 custom feature extractors should be implemented for this.

Required class:

```python
class JepaFeatureExtractor(BaseFeaturesExtractor):
    ...
```

It must support:

```yaml
encoder_checkpoint: path
freeze_encoder: true | false
include_desired_goal: true
```

---

## 7.5 JEPA latent MPC

Purpose: evaluate JEPA as a planner, not just as a representation.

### Planner interface

```python
class LatentMPC:
    def act(self, obs: dict[str, np.ndarray], goal: np.ndarray) -> np.ndarray:
        ...
```

### Candidate action sampling

Support:

1. random shooting,
2. CEM, optional but recommended.

Default random shooting config:

```yaml
horizon: 8
num_candidates: 1024
action_low: [-1.0, -1.0, -1.0, -1.0]
action_high: [1.0, 1.0, 1.0, 1.0]
lambda_action: 0.001
```

CEM config:

```yaml
horizon: 8
num_candidates: 512
num_elites: 64
iterations: 4
init_std: 0.7
min_std: 0.05
```

### Scoring

State goal encoding:

```text
z_goal = target_encoder(x_goal_proxy)
```

For Fetch tasks, construct `x_goal_proxy` by taking the current observation and replacing achieved goal / object goal components with desired goal where appropriate.

Important: implement this carefully and document assumptions per environment.

Score:

```text
score = - || normalize(z_pred_final) - normalize(z_goal) ||_2^2
```

Execute first action of best sequence.

---

## 8. Dataset Collection

### 8.1 Dataset sources

Support collection from:

1. random policy,
2. trained RL policy,
3. mixture policy.

Config:

```yaml
dataset:
  source: random | rl_policy | mixture
  num_episodes: 1000
  max_episode_steps: 50
  save_format: npz
```

Mixture example:

```yaml
mixture:
  random_fraction: 0.30
  rl_policy_fraction: 0.70
```

### 8.2 Stored fields

Each trajectory dataset must contain:

```text
observations
achieved_goals
desired_goals
actions
rewards
terminated
truncated
infos
episode_ids
timestep_ids
```

For visual mode also store:

```text
images
```

### 8.3 Dataset file layout

```text
outputs/{experiment_id}/datasets/{env_id}/{source}/seed_{seed}/
  trajectories.npz
  metadata.json
```

Metadata must include:

```json
{
  "env_id": "...",
  "seed": 0,
  "num_episodes": 1000,
  "num_transitions": 50000,
  "source": "random",
  "obs_mode": "state",
  "created_at": "...",
  "git_commit": "..."
}
```

---

## 9. Training and Evaluation Pipeline

## 9.1 Minimum pipeline

The coding agent must implement the following runnable pipeline:

```bash
# 1. Train traditional RL baseline
uv run python -m jepa_robotics.cli.train_rl \
  --config configs/experiments/phase1_fetch_reach.yaml \
  --method sac \
  --seed 0

# 2. Collect dataset
uv run python -m jepa_robotics.cli.collect_dataset \
  --config configs/experiments/phase1_fetch_reach.yaml \
  --policy outputs/.../model.zip \
  --seed 0

# 3. Train JEPA
uv run python -m jepa_robotics.cli.train_jepa \
  --config configs/jepa/state_jepa.yaml \
  --dataset outputs/.../trajectories.npz \
  --seed 0

# 4. Evaluate JEPA-MPC
uv run python -m jepa_robotics.cli.evaluate \
  --config configs/jepa/jepa_mpc.yaml \
  --checkpoint outputs/.../encoder.pt \
  --seed 0

# 5. Make plots
uv run python -m jepa_robotics.cli.plot \
  --experiment outputs/phase1_fetch_reach
```

## 9.2 Experiment phases

### Phase 0: Reacher debug, optional

Purpose: debug model training and plotting quickly.

### Phase 1: FetchReachDense-v4

Purpose: prove the stack works.

Methods:

```text
SAC
JEPA-MPC
JEPA-pretrained SAC
Autoencoder-MPC
```

### Phase 2: FetchPushDense-v4

Purpose: main dense-reward result.

Methods:

```text
SAC
JEPA-pretrained SAC
Autoencoder-pretrained SAC
JEPA-MPC
```

### Phase 3: FetchPushDense-v4 primary contact control

Purpose: main object-contact result with dense reward.

Methods:

```text
SAC
JEPA-pretrained SAC
Autoencoder-pretrained SAC
JEPA-MPC
```

### Sparse stress test: FetchPush-v4

Purpose: compare against HER.

Methods:

```text
SAC+HER
JEPA-pretrained SAC+HER
JEPA-MPC
```

### Phase 4: Generalization

Purpose: test representation robustness.

Train on default environment. Evaluate on modified wrappers:

1. observation noise,
2. goal distribution shift,
3. action noise,
4. object initial-position shift,
5. optional visual distractors if visual mode exists.

---

## 10. Metrics

## 10.1 RL metrics

For every evaluation checkpoint:

```text
global_step
seed
method
env_id
mean_episode_reward
std_episode_reward
mean_success_rate
std_success_rate
mean_episode_length
```

## 10.2 Sample-efficiency metrics

For each method and seed:

```text
steps_to_success_25
steps_to_success_50
steps_to_success_75
auc_success
auc_reward
final_success_rate
final_mean_reward
```

Define steps-to-threshold as the first training step where the rolling mean success rate reaches the threshold.

If threshold is never reached, record:

```text
NaN
```

and report the fraction of seeds that reached the threshold.

## 10.3 JEPA model metrics

```text
train_loss
val_loss
cosine_similarity_h1
cosine_similarity_h2
cosine_similarity_h4
cosine_similarity_h8
latent_std_mean
latent_std_min
latent_norm_mean
prediction_error_by_horizon
```

## 10.4 Representation quality metrics

Train linear probes on frozen latent states to predict:

```text
achieved_goal
desired_goal
object_position
gripper_position
```

Probe metrics:

```text
r2
mse
mae
```

Use scikit-learn ridge regression or a tiny PyTorch linear model.

## 10.5 MPC metrics

```text
planning_horizon
num_candidates
planning_time_ms
chosen_action_norm
predicted_goal_latent_distance
actual_goal_distance_after_step
correlation_predicted_vs_actual_progress
```

---

## 11. Required Plots

All plots must be generated from saved CSV logs, not from live training objects.

Save each plot as both `.png` and `.pdf`.

Output layout:

```text
reports/{experiment_id}/plots/
  sample_efficiency_success.png
  sample_efficiency_success.pdf
  sample_efficiency_reward.png
  sample_efficiency_reward.pdf
  steps_to_threshold_bar.png
  auc_success_bar.png
  final_success_boxplot.png
  reward_boxplot.png
  jepa_loss_curves.png
  jepa_horizon_cosine_similarity.png
  latent_std_over_time.png
  prediction_error_by_horizon.png
  probe_r2_bar.png
  probe_mse_bar.png
  mpc_predicted_vs_actual_progress.png
  mpc_planning_time_histogram.png
  generalization_success_heatmap.png
  generalization_reward_heatmap.png
```

### 11.1 Sample-efficiency plot

X-axis:

```text
environment steps
```

Y-axis:

```text
mean success rate
```

Lines:

```text
SAC
SAC+HER
JEPA-pretrained SAC
Autoencoder-pretrained SAC
JEPA-MPC
```

Show:

- mean across seeds,
- 95% confidence interval or standard error ribbon.

### 11.2 Reward learning curve

X-axis:

```text
environment steps
```

Y-axis:

```text
mean episode reward
```

Same methods and seeds.

### 11.3 Steps-to-threshold plot

Bar plot for:

```text
success >= 0.25
success >= 0.50
success >= 0.75
```

Report unreached thresholds explicitly in the table.

### 11.4 AUC plot

Bar plot:

```text
AUC of success rate over environment steps
```

### 11.5 Final performance plot

Boxplot or violin plot:

```text
final success rate by method across seeds
```

### 11.6 JEPA diagnostics

Required:

1. train/validation JEPA loss over epochs,
2. cosine similarity by horizon,
3. latent standard deviation over time,
4. prediction error by horizon.

### 11.7 Representation probe plots

Required:

1. linear-probe R^2 by predicted variable,
2. linear-probe MSE by predicted variable,
3. comparison against raw observations and autoencoder latents.

### 11.8 MPC diagnostic plots

Required:

1. predicted latent progress vs actual environment progress,
2. planning time distribution,
3. success rate by planning horizon,
4. success rate by candidate count.

### 11.9 Generalization plots

For each generalization condition:

```text
method x perturbation_level -> success_rate
method x perturbation_level -> mean_reward
```

Use heatmaps and line plots.

---

## 12. Statistical Analysis

Implement an analysis script:

```bash
uv run python -m jepa_robotics.cli.analyze \
  --experiment outputs/phase2_fetch_push_dense
```

It should produce:

```text
reports/{experiment_id}/tables/
  aggregate_metrics.csv
  threshold_metrics.csv
  statistical_tests.csv
```

Minimum analysis:

1. mean and standard error by method,
2. per-seed metrics,
3. paired comparison when seeds are matched,
4. bootstrap confidence intervals for AUC and final success.

Recommended bootstrap:

```yaml
bootstrap_samples: 10000
confidence_level: 0.95
```

Statistical tests:

- Paired t-test if distributions look reasonable.
- Wilcoxon signed-rank as non-parametric backup.

Do not overclaim significance. Report effect sizes.

---

## 13. Config System

Use YAML configs validated by Pydantic.

Example:

```yaml
experiment:
  name: phase2_fetch_push_dense
  output_dir: outputs/phase2_fetch_push_dense
  seeds: [0, 1, 2, 3, 4]

env:
  id: FetchPushDense-v4
  obs_mode: state
  max_episode_steps: 50

rl:
  algorithm: SAC
  total_timesteps: 500000
  eval_freq: 10000
  n_eval_episodes: 20
  learning_rate: 0.0003
  batch_size: 256
  gamma: 0.98
  buffer_size: 1000000

dataset:
  source: mixture
  num_episodes: 1000

jepa:
  latent_dim: 128
  hidden_dims: [256, 256]
  prediction_horizons: [1, 2, 4, 8]
  batch_size: 512
  epochs: 100
  learning_rate: 0.0003
  ema_momentum: 0.996
  lambda_var: 1.0
  lambda_cov: 0.04

mpc:
  planner: cem
  horizon: 8
  num_candidates: 512
  num_elites: 64
  iterations: 4
  lambda_action: 0.001

device:
  preferred: auto
```

---

## 14. Logging

Every CLI run must save:

```text
config_resolved.yaml
metrics.csv
stdout.log
stderr.log
```

Use a consistent metrics schema.

Recommended columns:

```text
timestamp
experiment
method
seed
env_id
phase
step
epoch
metric
value
```

For easier plotting, also allow wide-format metric CSVs.

---

## 15. Acceptance Criteria

The implementation is acceptable when all of the following are true.

### 15.1 Code quality

```bash
uv run ruff check .
uv run mypy src
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
```

all pass.

### 15.2 Minimal experiment works

This command sequence works end-to-end on CPU in under a small smoke-test budget:

```bash
uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.collect_dataset --config configs/experiments/phase1_fetch_reach.yaml --smoke-test
uv run python -m jepa_robotics.cli.train_jepa --config configs/jepa/state_jepa.yaml --smoke-test
uv run python -m jepa_robotics.cli.evaluate --config configs/jepa/jepa_mpc.yaml --smoke-test
uv run python -m jepa_robotics.cli.plot --experiment outputs/smoke
```

### 15.3 Real experiment runs

The following full experiments should run without code changes:

```bash
scripts/train_baselines.sh
scripts/train_jepa.sh
scripts/make_all_plots.sh
```

### 15.4 Required outputs exist

For each real experiment, the code must produce:

```text
outputs/.../metrics.csv
reports/.../plots/*.png
reports/.../plots/*.pdf
reports/.../tables/aggregate_metrics.csv
reports/.../tables/threshold_metrics.csv
```

### 15.5 Comparison result is answerable

At the end of the project, the repo must allow answering:

1. Which method reaches 25%, 50%, and 75% success fastest?
2. Which method has the largest AUC success?
3. Which method has the best final success?
4. Does JEPA pretraining improve sample efficiency over SAC from scratch?
5. Does JEPA latent MPC work at all in FetchReach?
6. Does JEPA latent MPC transfer to FetchPush?
7. Does JEPA representation encode object/goal information better than autoencoder latents?
8. Under which perturbations does JEPA generalize better or worse?

---

## 16. Implementation Order for Coding Agent

Implement in this order:

1. Create project skeleton with `uv`, `pyproject.toml`, and `src/` layout.
2. Implement config schemas and loading.
3. Implement device and seed utilities.
4. Implement environment factory for FetchReach and FetchPush.
5. Implement dataset collection from random policy.
6. Implement trajectory dataset loading and window sampling.
7. Implement StateEncoder, Predictor, JEPA module, EMA updater, and loss.
8. Implement JEPA trainer with synthetic test and then real dataset test.
9. Implement RL baseline training with SB3 SAC.
10. Implement SAC+HER for sparse FetchPush.
11. Implement evaluation rollouts and metrics.
12. Implement plots from CSV logs.
13. Implement latent MPC with random shooting.
14. Upgrade latent MPC to CEM.
15. Implement JEPA-pretrained SB3 feature extractor.
16. Implement autoencoder baseline.
17. Implement representation probes.
18. Implement generalization wrappers.
19. Implement statistical analysis.
20. Add smoke tests and push coverage to 90%+.
21. Add README instructions and example commands.

---

## 17. Important Design Constraints

1. Keep environment interaction separate from model training.
2. Keep PyTorch training independent from Stable-Baselines3 training.
3. Never require CUDA; CPU and MPS should be valid.
4. Never make plotting depend on re-running experiments.
5. Use configs for every hyperparameter.
6. Keep all outputs deterministic with seed control where possible.
7. Write tests for small pieces, not long training runs.
8. Use small smoke-test configs for CI.
9. Avoid global mutable state except for explicitly seeded RNGs.
10. Do not silently fall back from requested CUDA/MPS to CPU. Fail clearly.

---

## 18. Suggested Default Hyperparameters

### JEPA

```yaml
latent_dim: 128
encoder_hidden_dims: [256, 256]
predictor_hidden_dims: [256, 256]
activation: silu
layer_norm: true
batch_size: 512
epochs: 100
learning_rate: 0.0003
weight_decay: 0.000001
ema_momentum: 0.996
prediction_horizons: [1, 2, 4, 8]
lambda_var: 1.0
lambda_cov: 0.04
gradient_clip_norm: 10.0
```

### SAC

```yaml
learning_rate: 0.0003
buffer_size: 1000000
batch_size: 256
tau: 0.05
gamma: 0.98
train_freq: 1
gradient_steps: 1
learning_starts: 10000
```

### SAC+HER

```yaml
n_sampled_goal: 4
goal_selection_strategy: future
online_sampling: true
max_episode_length: 50
```

### MPC

```yaml
planner: cem
horizon: 8
num_candidates: 512
num_elites: 64
iterations: 4
lambda_action: 0.001
```

---

## 19. Deliverables

The coding agent should deliver:

1. Working repo with the structure above.
2. `README.md` with installation and experiment commands.
3. `spec.md` preserved in repo.
4. Config files for all phases.
5. Tests with 90%+ line coverage.
6. Baseline RL implementation.
7. JEPA training implementation.
8. JEPA-MPC implementation.
9. JEPA-pretrained SAC implementation.
10. Autoencoder baseline implementation.
11. Plotting and statistical analysis scripts.
12. Example report generated from at least one smoke experiment.

---

## 20. Report Template

Generate a final markdown report at:

```text
reports/{experiment_id}/report.md
```

With sections:

```text
# Experiment Report

## Setup
- Environment
- Methods
- Seeds
- Training budgets

## Main Results
- Success-rate curves
- Reward curves
- Steps to threshold
- AUC success

## JEPA Diagnostics
- Loss curves
- Horizon prediction quality
- Latent collapse diagnostics

## Representation Analysis
- Probe results
- Comparison to autoencoder latents

## MPC Analysis
- Predicted vs actual progress
- Planning time
- Horizon/candidate-count sensitivity

## Generalization
- Perturbation results
- Failure modes

## Conclusions
- What worked
- What failed
- Recommended next experiment
```

---

## 21. Known Risks and Mitigations

### Risk: JEPA collapse

Mitigation:

- Add variance/covariance regularization.
- Use EMA target encoder.
- Normalize latents.
- Monitor latent standard deviation.

### Risk: JEPA-MPC performs poorly

Mitigation:

- Start with FetchReachDense-v4.
- Use dense rewards and state observations first.
- Test one-step prediction before multi-step rollout.
- Add CEM after random shooting works.
- Compare predicted progress with actual progress.

### Risk: SAC+HER setup is fragile

Mitigation:

- Keep native dict observations.
- Use `MultiInputPolicy`.
- Use `HerReplayBuffer`.
- Avoid flattening the observation before SB3 gets it.

### Risk: MPS incompatibilities

Mitigation:

- Add CPU fallback only in `auto` mode.
- Keep operations simple.
- Avoid exotic PyTorch ops.
- Test basic train step on MPS if available.

### Risk: long training times

Mitigation:

- Maintain smoke-test configs.
- Cache datasets.
- Use smaller networks for debug.
- Parallelize seeds at the script level, not inside the trainer.

---

## 22. Definition of Done

The project is done when a user can run:

```bash
uv sync
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
bash scripts/train_baselines.sh
bash scripts/train_jepa.sh
bash scripts/make_all_plots.sh
```

and obtain:

1. trained RL baselines,
2. trained JEPA model,
3. JEPA-MPC evaluation,
4. sample-efficiency plots,
5. representation probe plots,
6. MPC diagnostic plots,
7. aggregate result tables,
8. final report markdown.

The final report must explicitly state whether JEPA improved sample efficiency, final performance, or generalization, and where it failed.
