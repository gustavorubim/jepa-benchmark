#!/usr/bin/env bash
set -euo pipefail
uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase1_fetch_reach.yaml
uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase2_fetch_push_dense.yaml
uv run python -m jepa_robotics.cli.train_rl --config configs/experiments/phase3_fetch_push_sparse.yaml --method sac_her
