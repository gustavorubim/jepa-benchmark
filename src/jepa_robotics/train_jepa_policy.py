"""Compatibility entry point for `python -m jepa_robotics.train_jepa_policy`."""

from __future__ import annotations

from jepa_robotics.cli.train_jepa_policy import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
