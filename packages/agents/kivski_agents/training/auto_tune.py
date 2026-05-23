"""Auto-tuning helpers for the vectorised training loop.

The training throughput scales roughly linearly with the number of parallel
envs as long as we have CPU cores to run them on. Picking a good default
without making the user think is the goal here.

Heuristics:

* ``detect_optimal_num_envs`` returns the requested value if provided;
  otherwise it scales with ``os.cpu_count()`` but caps at 64 because the
  benefit plateaus after that (model forward + optimiser updates start to
  dominate the wallclock).
* ``detect_optimal_workers`` decides how many subprocess workers should
  shard the env pool. We never request more workers than envs.

Both helpers are deliberately tiny and pure so the CLI can dump the chosen
numbers for the user.
"""

from __future__ import annotations

import os

__all__ = [
    "detect_optimal_num_envs",
    "detect_optimal_workers",
    "envs_per_worker_split",
]


def detect_optimal_num_envs(requested: int | None = None) -> int:
    """Return a sensible ``num_envs`` for the host machine.

    Args:
        requested: If a positive integer is passed, it is honoured verbatim
            and returned unchanged. Pass ``None`` (or ``0``) to ask for
            auto-detection.

    Returns:
        The chosen ``num_envs``.

    Heuristic:
        - Reserve ~2 cores for the host (OS, model forward, optimiser, IO).
        - Floor at 8 so even a tiny machine still trains in parallel.
        - Cap at 64 to dodge diminishing returns from the central forward.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    cpu = int(os.cpu_count() or 4)
    return max(8, min(64, cpu - 2))


def detect_optimal_workers(num_envs: int) -> int:
    """Return how many subprocess workers should split ``num_envs``.

    The chosen worker count divides the env pool evenly when possible:
    each worker hosts ``num_envs // num_workers`` envs (the remainder is
    handed to the first ``num_envs % num_workers`` workers).

    Args:
        num_envs: Total parallel env count chosen by the caller.

    Returns:
        Worker count in ``[1, num_envs]``. Cores minus one is the upper
        bound so the main process keeps a thread for forward/IO work.
    """
    if int(num_envs) <= 0:
        return 1
    cpu = int(os.cpu_count() or 4)
    return max(1, min(int(num_envs), cpu - 1))


def envs_per_worker_split(num_envs: int, num_workers: int) -> list[int]:
    """Return a list of per-worker env counts summing to ``num_envs``.

    Example: ``envs_per_worker_split(10, 3) == [4, 3, 3]``.

    Args:
        num_envs: Total envs to distribute.
        num_workers: Number of workers to split across.
    """
    n = int(num_envs)
    w = max(1, int(num_workers))
    if n <= 0 or w <= 0:
        return []
    base = n // w
    rem = n - base * w
    return [base + (1 if i < rem else 0) for i in range(w)]
