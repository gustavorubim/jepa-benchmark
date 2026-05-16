"""Small provenance helpers for reproducible run metadata."""

from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path


def git_commit(cwd: str | Path | None = None) -> str:
    """Return the current git commit, or ``unknown`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def platform_metadata(cwd: str | Path | None = None) -> dict[str, str]:
    """Collect lightweight metadata that is useful when comparing runs."""
    return {
        "git_commit": git_commit(cwd),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
