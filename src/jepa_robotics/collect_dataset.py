"""Compatibility entry point for `python -m jepa_robotics.collect_dataset`."""

from __future__ import annotations

from jepa_robotics.cli.collect_dataset import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
