#!/usr/bin/env bash
set -euo pipefail
uv run python -m jepa_robotics.cli.plot --experiment outputs/phase1_fetch_reach
uv run python -m jepa_robotics.cli.analyze --experiment outputs/phase1_fetch_reach
