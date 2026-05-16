"""Rich logging setup."""

from __future__ import annotations

from rich.console import Console

console = Console()


def log_status(message: str) -> None:
    console.print(message)
