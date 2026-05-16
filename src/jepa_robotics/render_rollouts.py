"""Compatibility entry point for `python -m jepa_robotics.render_rollouts`."""

from __future__ import annotations

from jepa_robotics.cli.render_rollouts import main, run

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
