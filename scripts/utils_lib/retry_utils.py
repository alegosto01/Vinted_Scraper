from __future__ import annotations

import time


def sleep_with_backoff(attempt: int, base_delay: float = 3.0) -> None:
    time.sleep(max(0.0, base_delay) * max(1, attempt))


def sleep_if_positive(seconds: float | int | None) -> None:
    if not seconds:
        return
    if seconds > 0:
        time.sleep(seconds)
