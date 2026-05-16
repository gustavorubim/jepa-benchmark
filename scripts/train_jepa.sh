#!/usr/bin/env bash
set -euo pipefail
uv run python -m jepa_robotics.cli.collect_dataset --config configs/experiments/phase1_fetch_reach.yaml
uv run python -m jepa_robotics.cli.train_jepa --config configs/jepa/state_jepa.yaml
uv run python -m jepa_robotics.cli.evaluate \
  --config configs/jepa/jepa_mpc.yaml \
  --checkpoint outputs/state_jepa/jepa/state_jepa/seed_0/encoder.pt \
  --dataset outputs/state_jepa/datasets/FetchReachDense-v3/random/seed_0/trajectories.npz
