"""Timing helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def elapsed_timer() -> Iterator[Callable[[], float]]:
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    yield elapsed
