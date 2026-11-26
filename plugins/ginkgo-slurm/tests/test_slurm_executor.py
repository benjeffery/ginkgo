"""The Slurm executor's submissions, job states and worker, with Slurm's commands faked."""

from __future__ import annotations

import json
import os
import stat
import subprocess

import pytest
from ginkgo_slurm.executor import SlurmExecutor, SlurmSettings, build_slurm_executor

from ginkgo.plugins.api import ExecutorType
from ginkgo.plugins.registry import plugin_registry
from ginkgo.runtime.remote_executor import RemoteJobState


class FakeSlurm:
    """Answers sbatch, squeue and scancel as a cluster would, recording each call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.queue: dict[str, str] = {}
        self.next_id = 100
        self.sbatch_failures = 0

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if argv[0] == "sbatch":
            if self.sbatch_failures:
                self.sbatch_failures -= 1
                return subprocess.CompletedProcess(argv, 1, "", "Socket timed out")
            self.next_id += 1
            self.queue[str(self.next_id)] = "PENDING"
            return subprocess.CompletedProcess(argv, 0, f"{self.next_id};cluster\n", "")
        if argv[0] == "squeue":
            lines = "".join(f"{job} {state}\n" for job, state in self.queue.items())
            return subprocess.CompletedProcess(argv, 0, lines, "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def count(self, command: str) -> int:
        return sum(call[0] == command for call in self.calls)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake, clock = FakeSlurm(), Clock()
    settings = SlurmSettings(
        partition="standard",
        account="lab",
        sbatch_args=("--qos=long",),
        setup=('mkdir -p "$TMPDIR"',),
        python="/shared/venv/bin/python",
        poll_interval=3600,
        result_grace=60,
        job_name_prefix="proj",
    )
    executor = SlurmExecutor(settings, runner=fake, clock=clock, background_poll=False)
    return executor, fake, clock


def attempt(**resources) -> dict:
    return {
        "task_name": "workflow.modules.train.train_model",
        "node_id": 7,
        "attempt": 2,
        "run_id": "20261126_abcdef12",
        "transport_dir": "/shared/project/.ginkgo/transport/t",
        "secret_values": ["hunter2"],
        "resources": {"threads": 4, "memory_mib": 16384, "time_limit_seconds": 5400, **resources},
    }


def finish(handle, response: dict) -> None:
    handle._files.result.write_text(json.dumps(response))


def test_submission_maps_resources_and_writes_a_private_payload(slurm, tmp_path):
    executor, fake, _ = slurm
    handle = executor.submit(attempt=attempt(gpu=1, gpu_type="a100"))

    [sbatch] = [call for call in fake.calls if call[0] == "sbatch"]
    name = "proj-train_model-7-2-abcdef12"
    jobs = tmp_path / ".ginkgo/slurm"
    assert sbatch == [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--output={jobs / name}.log",
        f"--chdir={tmp_path}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16384M",
        "--time=90",
        "--partition=standard",
        "--account=lab",
        "--gres=gpu:a100:1",
        "--qos=long",
        str(jobs / f"{name}.sh"),
    ]
    assert handle.job_id == "101"
    payload = jobs / f"{name}.json"
    assert stat.S_IMODE(payload.stat().st_mode) == 0o600
    assert json.loads(payload.read_text())["secret_values"] == ["hunter2"]
    script = (jobs / f"{name}.sh").read_text().splitlines()
    assert script == [
        "#!/bin/bash",
        "set -euo pipefail",
        'mkdir -p "$TMPDIR"',
        f"exec /shared/venv/bin/python -m ginkgo_slurm.worker {payload} {jobs / name}.result.json",
    ]


def test_executor_default_time_applies_when_a_task_declares_none(slurm):
    executor, fake, _ = slurm
    executor.submit(attempt=attempt(time_limit_seconds=None, memory_mib=None))
    [sbatch] = [call for call in fake.calls if call[0] == "sbatch"]
    assert "--time=60" in sbatch
    assert not any(option.startswith("--mem") for option in sbatch)


def test_states_follow_the_queue_until_the_response_arrives(slurm):
    executor, fake, clock = slurm
    handle = executor.submit(attempt=attempt())
    clock.now += 1
    executor._queue.poll()
    assert handle.state() == RemoteJobState.PENDING
    fake.queue[handle.job_id] = "RUNNING"
    executor._queue.poll()
    assert handle.state() == RemoteJobState.RUNNING

    finish(handle, {"ok": True, "result": 3})
    assert handle.state() == RemoteJobState.SUCCEEDED
    result = handle.result()
    assert result.payload == {"ok": True, "result": 3}
    assert not handle._files.payload.exists()  # It held secrets.


def test_a_task_failure_arrives_as_its_response(slurm):
    executor, _, _ = slurm
    handle = executor.submit(attempt=attempt())
    finish(handle, {"ok": False, "error": {"type": "ValueError"}})
    result = handle.result()
    assert result.state == RemoteJobState.FAILED
    assert result.payload["error"] == {"type": "ValueError"}


def test_state_is_pending_until_a_poll_has_seen_the_job(slurm):
    executor, fake, clock = slurm
    executor._queue.poll()
    clock.now += 5
    handle = executor.submit(attempt=attempt())
    del fake.queue[handle.job_id]  # Not listed yet, but no poll since submission either.
    assert handle.state() == RemoteJobState.PENDING


def test_a_job_gone_without_a_response_fails_after_the_grace_period(slurm):
    executor, fake, clock = slurm
    handle = executor.submit(attempt=attempt())
    handle._files.log.write_text("slurmstepd: error: Detected 1 oom_kill event\n")
    fake.queue[handle.job_id] = "OUT_OF_MEMORY"
    clock.now += 1
    executor._queue.poll()
    assert handle.state() == RemoteJobState.RUNNING  # Waiting for a late response.
    clock.now += 61
    assert handle.state() == RemoteJobState.FAILED
    result = handle.result()
    assert result.payload == {}
    assert "OUT_OF_MEMORY" in result.logs and "oom_kill" in result.logs


def test_a_cancelled_job_is_reported_cancelled(slurm):
    executor, fake, clock = slurm
    handle = executor.submit(attempt=attempt())
    fake.queue[handle.job_id] = "CANCELLED+"
    clock.now += 1
    executor._queue.poll()
    assert handle.state() == RemoteJobState.RUNNING  # Waiting for a late response.
    clock.now += 61
    assert handle.state() == RemoteJobState.CANCELLED
    handle.cancel()
    assert ["scancel", handle.job_id] in fake.calls


def test_one_poll_serves_every_job(slurm):
    executor, fake, clock = slurm
    handles = [executor.submit(attempt=attempt()) for _ in range(5)]
    before = fake.count("squeue")
    clock.now += 1
    executor._queue.poll()
    for handle in handles:
        fake.queue[handle.job_id] = "RUNNING"
    executor._queue.poll()
    assert all(handle.state() == RemoteJobState.RUNNING for handle in handles)
    assert fake.count("squeue") == before + 2


def test_sbatch_is_retried_then_reported(slurm, monkeypatch):
    executor, fake, _ = slurm
    monkeypatch.setattr("ginkgo_slurm.executor.time.sleep", lambda seconds: None)
    fake.sbatch_failures = 2
    assert executor.submit(attempt=attempt()).job_id == "101"
    fake.sbatch_failures = 10
    with pytest.raises(RuntimeError, match="Socket timed out"):
        executor.submit(attempt=attempt())


def test_settings_are_validated_and_the_type_shares_the_filesystem():
    executor_type = plugin_registry().get(ExecutorType, "slurm")
    assert executor_type.shared_filesystem
    executor = build_slurm_executor({"partition": "p", "setup": ["module load x"]})
    assert executor.settings.setup == ("module load x",)
    with pytest.raises(ValueError, match="Unknown slurm executor settings"):
        build_slurm_executor({"partitions": "p"})
    with pytest.raises(ValueError, match="list of strings"):
        build_slurm_executor({"setup": "mkdir x"})


def test_worker_runs_the_payload_and_removes_it(tmp_path, monkeypatch):
    from ginkgo_slurm import worker

    payload = tmp_path / "job.json"
    payload.write_text(json.dumps({"transport_dir": str(tmp_path)}))
    result = tmp_path / "job.result.json"
    monkeypatch.setattr(
        "ginkgo.runtime.worker.run_task",
        lambda payload: {"ok": True, "run_dir_root": payload["run_dir_root"]},
    )
    assert worker.main(str(payload), str(result)) == 0
    assert not payload.exists()
    assert json.loads(result.read_text()) == {
        "ok": True,
        "run_dir_root": str(tmp_path / "runs"),
    }


def test_worker_reports_an_unreadable_payload(tmp_path):
    from ginkgo_slurm import worker

    payload = tmp_path / "job.json"
    payload.write_text("not json")
    result = tmp_path / "job.result.json"
    assert worker.main(str(payload), str(result)) == 1
    response = json.loads(result.read_text())
    assert response["ok"] is False
    assert not payload.exists()
    assert not any(path.name.endswith(".partial") for path in tmp_path.iterdir())
    assert os.listdir(tmp_path) == [result.name]
