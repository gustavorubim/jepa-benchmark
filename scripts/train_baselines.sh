#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-iteration}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
SEEDS="${SEEDS:-0}"

uv run python -m jepa_robotics.cli.run_suite \
  --mode "${MODE}" \
  --phases phase1_fetch_reach phase3_fetch_push_sparse \
  --seeds ${SEEDS} \
  --max-parallel "${MAX_PARALLEL}"
