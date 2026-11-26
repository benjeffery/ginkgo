"""Executors whose workers share the driver's filesystem."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from ginkgo import Out, file, task, task_resources
from ginkgo.runtime.evaluator import ConcurrentEvaluator
from ginkgo.runtime.executor_registry import ExecutorRegistry
from ginkgo.runtime.remote_executor import RemoteJobResult, RemoteJobState
from ginkgo.runtime.rundir import RunDir
from ginkgo.runtime.worker import run_task


class SharedFilesystemExecutor:
    """Runs each attempt's payload with :func:`run_task` in a thread, as a cluster node would."""

    shared_filesystem = True

    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []

    def submit(self, *, attempt: dict[str, Any]) -> _Handle:
        self.attempts.append(attempt)
        return _Handle(attempt)


class _Handle:
    def __init__(self, attempt: dict[str, Any]) -> None:
        self.job_id = uuid.uuid4().hex[:8]
        self._response: dict[str, Any] | None = None
        self._thread = threading.Thread(target=self._run, args=(attempt,))
        self._thread.start()

    def _run(self, attempt: dict[str, Any]) -> None:
        # The payload is the local worker's, paths and all.
        assert Path(attempt["transport_dir"]).is_dir()
        self._response = run_task(attempt)

    def state(self) -> RemoteJobState:
        if self._thread.is_alive():
            return RemoteJobState.RUNNING
        return RemoteJobState.SUCCEEDED

    def result(self) -> RemoteJobResult:
        self._thread.join()
        assert self._response is not None
        return RemoteJobResult(state=RemoteJobState.SUCCEEDED, payload=self._response)

    def cancel(self) -> None:
        pass

    def logs_tail(self, *, lines: int = 100) -> str:
        return "not the task's log"


@task(threads=3, memory="1536Mi", time_limit="90m")
def write_allocation(label: str, out: Out[file]):
    allocated = task_resources()
    Path(out).write_text(
        f"{label} {allocated.threads} {allocated.memory_mib} {allocated.time_limit_seconds}\n"
    )
    print("written by the task")


def evaluate_on(executor: SharedFilesystemExecutor, expr: Any) -> Any:
    registry = ExecutorRegistry.for_executor(executor, name="cluster")
    run_dir = RunDir.create(run_id=uuid.uuid4().hex, root=Path.cwd() / "runs")
    evaluator = ConcurrentEvaluator(jobs=2, executor_registry=registry, run_dir=run_dir)
    return evaluator.evaluate(expr)


def test_shared_filesystem_task_needs_no_artifact_store_and_writes_driver_paths():
    executor = SharedFilesystemExecutor()

    result = evaluate_on(executor, write_allocation(label="a", out="results/a.txt"))

    assert Path(result).resolve() == Path("results/a.txt").resolve()
    assert Path("results/a.txt").read_text() == "a 3 1536 5400\n"
    [attempt] = executor.attempts
    assert Path(attempt["transport_dir"]).parent == Path.cwd() / ".ginkgo" / "transport"
    assert not Path(attempt["transport_dir"]).exists()  # Removed once the task finished.
    assert attempt["resources"]["memory_mib"] == 1536
    assert attempt["resources"]["time_limit_seconds"] == 5400
    # The worker wrote the task's own log; the executor's logs did not replace it.
    [log] = Path("runs").rglob("*.stdout.log")
    assert "written by the task" in log.read_text()
