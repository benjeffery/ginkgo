"""The resources a running task was given, readable from its own body.

A task's ``threads`` and ``memory`` are scheduling declarations, kept out of
its cache key so that resizing a task reruns nothing. A body that sizes a
thread pool or a native memory budget still needs the values it was actually
given — after site overrides and any retry escalation — so the worker that
runs the body publishes them here for :func:`task_resources` to read.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class TaskResources:
    """What the running task attempt was allocated.

    Parameters
    ----------
    threads : int
        CPU threads.
    memory_mib : int | None
        Memory in MiB, or ``None`` when the task declared none.
    time_limit_seconds : int | None
        Wall-clock limit, or ``None`` when the task declared none.
    """

    threads: int
    memory_mib: int | None = None
    time_limit_seconds: int | None = None


_CURRENT: ContextVar[TaskResources | None] = ContextVar("ginkgo_task_resources", default=None)


def task_resources() -> TaskResources:
    """Return the resources of the task attempt whose body is running.

    Raises
    ------
    RuntimeError
        Outside a task body, where no attempt has been allocated anything.
    """
    current = _CURRENT.get()
    if current is None:
        raise RuntimeError("task_resources() is only available inside a running task body")
    return current


@contextmanager
def task_resources_scope(payload: dict[str, Any]) -> Iterator[TaskResources]:
    """Publish a worker payload's resources for the duration of the body."""
    resources = TaskResources(
        threads=int(payload.get("threads") or 1),
        memory_mib=payload.get("memory_mib"),
        time_limit_seconds=payload.get("time_limit_seconds"),
    )
    token = _CURRENT.set(resources)
    try:
        yield resources
    finally:
        _CURRENT.reset(token)
