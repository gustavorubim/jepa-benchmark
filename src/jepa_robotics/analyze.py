"""Compatibility entry point for `python -m jepa_robotics.analyze`."""

from __future__ import annotations

from jepa_robotics.cli.analyze import main, run

__all__ = ["main", "run"]

if __name__ == "__main__":
    main()
