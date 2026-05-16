#!/usr/bin/env bash
set -euo pipefail
uv run pytest --cov=src/jepa_robotics --cov-report=term-missing --cov-fail-under=90
