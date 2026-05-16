"""Compatibility entry point for `python -m jepa_robotics.plot`."""

from __future__ import annotations

from jepa_robotics.cli.plot import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
