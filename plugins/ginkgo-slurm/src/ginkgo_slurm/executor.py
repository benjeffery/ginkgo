"""Submit task attempts with ``sbatch`` and follow them with one shared ``squeue`` poll.

Each attempt becomes one batch job that runs :mod:`ginkgo_slurm.worker` from the
project root. The payload and the worker's response are files beside the job's
log under ``jobs_dir``, on the filesystem the driver and every node share.

Job state comes from the response file first: a job that wrote one has
finished, whatever Slurm still reports. Otherwise ``squeue --states=all``,
polled once per interval for every job the executor follows, says whether the
job is pending, running or gone. Slurm accounting (``sacct``) is not needed.
A job that left the queue without a response failed, once ``result_grace`` has
passed for the response to become visible over the shared filesystem.
"""

from __future__ import annotations

import getpass
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ginkgo.runtime.remote_executor import RemoteJobResult, RemoteJobState

Runner = Callable[..., subprocess.CompletedProcess[str]]

PENDING_STATES = frozenset(
    {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD"}
)
RUNNING_STATES = frozenset(
    {"RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT", "SIGNALING", "RESIZING", "STOPPED"}
)
CANCELLED_STATES = frozenset({"CANCELLED", "PREEMPTED", "DEADLINE"})


@dataclass(frozen=True, kw_only=True)
class SlurmSettings:
    """One ``[remote.executors.<name>]`` table of ``type = "slurm"``.

    Parameters
    ----------
    partition, account : str | None
        ``--partition`` and ``--account`` for every job.
    time_limit : str
        ``--time`` for a task that declares no ``time_limit`` of its own.
    sbatch_args : Sequence[str]
        Further ``sbatch`` options, such as ``["--qos=long"]``.
    setup : Sequence[str]
        Shell lines the job runs before the worker starts, for example
        ``mkdir -p "$TMPDIR"``. Jobs inherit the driver's environment.
    python : str
        The interpreter that runs the worker: the driver's own by default, so
        it must be on the shared filesystem.
    jobs_dir : str
        Where job scripts, payloads, responses and Slurm logs go, relative to
        the project root unless absolute.
    job_name_prefix : str
        Prefix of every job name, so ``squeue`` shows which run a job is for.
    poll_interval : float
        Seconds between ``squeue`` polls.
    result_grace : float
        Seconds a job may be gone from the queue before a missing response
        counts as a failure.
    submit_retries : int
        Further ``sbatch`` attempts after a failed submission.
    """

    partition: str | None = None
    account: str | None = None
    time_limit: str = "60m"
    sbatch_args: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    python: str = field(default_factory=lambda: sys.executable)
    jobs_dir: str = ".ginkgo/slurm"
    job_name_prefix: str = "ginkgo"
    poll_interval: float = 30.0
    result_grace: float = 60.0
    submit_retries: int = 3

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> SlurmSettings:
        known = {item.name for item in fields(cls)}
        unknown = set(settings) - known
        if unknown:
            raise ValueError(
                f"Unknown slurm executor settings {sorted(unknown)}; "
                f"supported settings are {sorted(known)}"
            )
        values = dict(settings)
        for name in ("sbatch_args", "setup"):
            if name in values:
                if isinstance(values[name], str) or not isinstance(values[name], Sequence):
                    raise ValueError(f"slurm executor setting {name!r} must be a list of strings")
                values[name] = tuple(str(item) for item in values[name])
        return cls(**values)


def build_slurm_executor(settings: Mapping[str, Any]) -> SlurmExecutor:
    """Build the backend from its executor table."""
    return SlurmExecutor(SlurmSettings.from_mapping(settings))


class SlurmExecutor:
    """Run each task attempt as one Slurm batch job on a shared filesystem."""

    shared_filesystem = True

    def __init__(
        self,
        settings: SlurmSettings,
        *,
        runner: Runner = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        background_poll: bool = True,
    ) -> None:
        self.settings = settings
        self._runner = runner
        self._clock = clock
        self._queue = _QueuePoller(
            interval=settings.poll_interval,
            runner=runner,
            clock=clock,
            background=background_poll,
        )

    def submit(self, *, attempt: dict[str, Any]) -> SlurmJobHandle:
        """Write the attempt's payload and job script, then ``sbatch`` it."""
        jobs_dir = Path(self.settings.jobs_dir)
        if not jobs_dir.is_absolute():
            jobs_dir = Path.cwd() / jobs_dir
        jobs_dir.mkdir(parents=True, exist_ok=True)
        name = _job_name(self.settings.job_name_prefix, attempt)
        files = _JobFiles(jobs_dir=jobs_dir, name=name)

        # The payload holds resolved secrets: readable by this user only.
        descriptor = os.open(files.payload, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(attempt, handle)
        files.result.unlink(missing_ok=True)
        files.script.write_text(self._script(files), encoding="utf-8")

        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--output={files.log}",
            f"--chdir={Path.cwd()}",
            *self._resource_options(attempt.get("resources") or {}),
            *self.settings.sbatch_args,
            str(files.script),
        ]
        job_id = self._sbatch(argv)
        self._queue.track(job_id)
        return SlurmJobHandle(
            job_id=job_id,
            files=files,
            queue=self._queue,
            runner=self._runner,
            clock=self._clock,
            result_grace=self.settings.result_grace,
        )

    def _script(self, files: _JobFiles) -> str:
        worker = shlex.join(
            [
                self.settings.python,
                "-m",
                "ginkgo_slurm.worker",
                str(files.payload),
                str(files.result),
            ]
        )
        return "\n".join(
            ["#!/bin/bash", "set -euo pipefail", *self.settings.setup, f"exec {worker}", ""]
        )

    def _resource_options(self, resources: Mapping[str, Any]) -> list[str]:
        options = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={resources.get('threads') or 1}"]
        memory_mib = resources.get("memory_mib")
        if memory_mib:
            options.append(f"--mem={memory_mib}M")
        seconds = resources.get("time_limit_seconds")
        if seconds is None:
            from ginkgo.core.resources import parse_time_limit

            seconds = parse_time_limit(self.settings.time_limit)
            assert seconds is not None  # The setting is a string, never None.
        options.append(f"--time={math.ceil(seconds / 60)}")
        if self.settings.partition:
            options.append(f"--partition={self.settings.partition}")
        if self.settings.account:
            options.append(f"--account={self.settings.account}")
        gpus = resources.get("gpu") or 0
        if gpus:
            gpu_type = resources.get("gpu_type")
            options.append(f"--gres=gpu:{gpu_type}:{gpus}" if gpu_type else f"--gres=gpu:{gpus}")
        return options

    def _sbatch(self, argv: list[str]) -> str:
        delay = 1.0
        for attempt in range(self.settings.submit_retries + 1):
            done = self._runner(argv, capture_output=True, text=True, check=False)
            if done.returncode == 0 and done.stdout.strip():
                return done.stdout.strip().split(";")[0]
            if attempt == self.settings.submit_retries:
                raise RuntimeError(
                    f"sbatch failed with exit code {done.returncode}: "
                    f"{(done.stderr or done.stdout).strip()}"
                )
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class _JobFiles:
    jobs_dir: Path
    name: str

    @property
    def payload(self) -> Path:
        return self.jobs_dir / f"{self.name}.json"

    @property
    def result(self) -> Path:
        return self.jobs_dir / f"{self.name}.result.json"

    @property
    def script(self) -> Path:
        return self.jobs_dir / f"{self.name}.sh"

    @property
    def log(self) -> Path:
        return self.jobs_dir / f"{self.name}.log"


class SlurmJobHandle:
    """One submitted attempt: its state, its response and its Slurm log."""

    def __init__(
        self,
        *,
        job_id: str,
        files: _JobFiles,
        queue: _QueuePoller,
        runner: Runner,
        clock: Callable[[], float],
        result_grace: float,
    ) -> None:
        self._job_id = job_id
        self._files = files
        self._queue = queue
        self._runner = runner
        self._clock = clock
        self._result_grace = result_grace
        self._submitted = clock()
        self._gone_since: float | None = None
        self._final: RemoteJobResult | None = None
        self._lock = threading.Lock()

    @property
    def job_id(self) -> str:
        return self._job_id

    def state(self) -> RemoteJobState:
        with self._lock:
            return self._poll()

    def result(self) -> RemoteJobResult:
        while True:
            with self._lock:
                if self._poll().is_terminal:
                    assert self._final is not None
                    return self._final
            time.sleep(min(self._queue.interval, 5.0))

    def cancel(self) -> None:
        self._runner(["scancel", self._job_id], capture_output=True, text=True, check=False)

    def logs_tail(self, *, lines: int = 100) -> str:
        try:
            text = self._files.log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _poll(self) -> RemoteJobState:
        if self._final is not None:
            return self._final.state
        response = self._read_response()
        if response is not None:
            ok = response.get("ok", False)
            return self._finish(
                RemoteJobState.SUCCEEDED if ok else RemoteJobState.FAILED, payload=response
            )

        slurm_state = self._queue.state(self._job_id, since=self._submitted)
        if slurm_state is _UNKNOWN or slurm_state in PENDING_STATES:
            return RemoteJobState.PENDING
        if slurm_state in RUNNING_STATES:
            return RemoteJobState.RUNNING

        # Finished or gone without a response yet: give the shared filesystem
        # time to show the response before calling it a failure.
        now = self._clock()
        if self._gone_since is None:
            self._gone_since = now
        if now - self._gone_since < self._result_grace:
            return RemoteJobState.RUNNING
        response = self._read_response()
        if response is not None:
            ok = response.get("ok", False)
            return self._finish(
                RemoteJobState.SUCCEEDED if ok else RemoteJobState.FAILED, payload=response
            )
        state = (
            RemoteJobState.CANCELLED if slurm_state in CANCELLED_STATES else RemoteJobState.FAILED
        )
        logs = self.logs_tail()
        reason = f"Slurm job {self._job_id} ended ({slurm_state or 'no longer queued'})"
        return self._finish(state, payload={}, logs=f"{reason} without a result\n{logs}")

    def _finish(
        self, state: RemoteJobState, *, payload: dict[str, Any], logs: str | None = None
    ) -> RemoteJobState:
        self._queue.forget(self._job_id)
        # The job script and its Slurm log stay beside the run for inspection.
        self._files.payload.unlink(missing_ok=True)
        self._files.result.unlink(missing_ok=True)
        self._final = RemoteJobResult(state=state, payload=payload, logs=logs)
        return state

    def _read_response(self) -> dict[str, Any] | None:
        # Listing the directory refreshes an NFS client's view of new files.
        try:
            os.listdir(self._files.jobs_dir)
            text = self._files.result.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


class _Unknown:
    """No ``squeue`` poll has run since the job was submitted."""


_UNKNOWN = _Unknown()


class _QueuePoller:
    """Poll ``squeue`` once per interval for every job any handle follows."""

    def __init__(
        self,
        *,
        interval: float,
        runner: Runner,
        clock: Callable[[], float],
        background: bool = True,
    ) -> None:
        self.interval = interval
        self._background = background
        self._runner = runner
        self._clock = clock
        self._tracked: set[str] = set()
        self._states: dict[str, str] = {}
        self._polled_at: float | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def track(self, job_id: str) -> None:
        with self._lock:
            self._tracked.add(job_id)
            if self._background and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(
                    target=self._run, name="ginkgo-slurm-squeue", daemon=True
                )
                self._thread.start()

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._tracked.discard(job_id)
            self._states.pop(job_id, None)

    def state(self, job_id: str, *, since: float) -> str | None | _Unknown:
        """The job's last polled state, ``None`` if it was not queued, or unknown."""
        with self._lock:
            if self._polled_at is None or self._polled_at < since:
                return _UNKNOWN
            return self._states.get(job_id)

    def poll(self) -> None:
        """Run one ``squeue`` and record the state of every followed job."""
        started = self._clock()
        done = self._runner(
            [
                "squeue",
                "--noheader",
                "--states=all",
                f"--user={getpass.getuser()}",
                "--format=%i %T",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            return  # Keep the last known states; the next poll may succeed.
        states = {}
        for line in done.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = re.sub(r"\W.*", "", parts[1])
        with self._lock:
            self._states = {job: states[job] for job in self._tracked if job in states}
            self._polled_at = started

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._tracked:
                    self._thread = None
                    return
            self.poll()
            self._wake.wait(self.interval)
            self._wake.clear()


def _job_name(prefix: str, attempt: Mapping[str, Any]) -> str:
    task = str(attempt.get("task_name") or "task").rsplit(".", 1)[-1]
    raw = f"{prefix}-{task}-{attempt.get('node_id', 0)}-{attempt.get('attempt', 1)}"
    run = str(attempt.get("run_id") or "")[-8:]
    if run:
        raw = f"{raw}-{run}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
