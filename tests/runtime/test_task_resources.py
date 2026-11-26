"""Task time limits, and the resources a running task's body sees."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from ginkgo import Out, file, task, task_resources
from ginkgo.core.resources import Resources, memory_mib, parse_time_limit
from ginkgo.runtime.evaluator import ConcurrentEvaluator
from ginkgo.runtime.rundir import RunDir


@task(threads=3, memory="1536Mi", time_limit="90m")
def write_allocation(label: str, out: Out[file]):
    allocated = task_resources()
    Path(out).write_text(
        f"{label} {allocated.threads} {allocated.memory_mib} {allocated.time_limit_seconds}\n"
    )


def test_local_task_sees_its_allocation_without_it_entering_the_cache_key():
    run_dir = RunDir.create(run_id=uuid.uuid4().hex, root=Path.cwd() / "runs")
    ConcurrentEvaluator(jobs=1, cores=4, run_dir=run_dir).evaluate(
        write_allocation(label="local", out="local.txt")
    )
    assert Path("local.txt").read_text() == "local 3 1536 5400\n"


def test_task_resources_is_only_available_inside_a_task():
    with pytest.raises(RuntimeError, match="inside a running task"):
        task_resources()


@pytest.mark.parametrize(
    "value,seconds",
    [("30s", 30), ("90m", 5400), ("5h", 18000), ("2d", 172800), ("01:30:00", 5400), (None, None)],
)
def test_time_limits_parse_to_seconds(value, seconds):
    assert parse_time_limit(value) == seconds
    assert Resources(time_limit=value).time_limit_seconds == seconds


@pytest.mark.parametrize("value", ["90", "1:30", "0m", 5, "soon"])
def test_invalid_time_limits_are_refused(value):
    with pytest.raises(ValueError, match="time_limit"):
        Resources(time_limit=value)


def test_memory_mib_rounds_up_and_keeps_declared_precision():
    assert memory_mib("8Gi") == 8192
    assert memory_mib("512Mi") == 512
    assert memory_mib("1G") == 954
    assert memory_mib(None) is None


def test_time_limit_is_a_resource_only_setting(tmp_path, monkeypatch):
    from ginkgo.core.source_hash import _without_resource_kwargs

    one = '@task(time_limit="5h")\ndef f():\n    pass\n'
    two = '@task(time_limit="1h")\ndef f():\n    pass\n'
    assert _without_resource_kwargs(one) == _without_resource_kwargs(two)
