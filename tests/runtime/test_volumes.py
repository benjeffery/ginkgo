"""Volumes: each task attempt mounts them itself, and publishes what it wrote when it succeeds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from tests.conftest import STUB_OBJECT_STORES
from tests.runtime.test_shared_filesystem_executor import SharedFilesystemExecutor

import ginkgo
from ginkgo import Out, file, folder, shell, task
from ginkgo.plugins.api import Plugin, VolumeContext, VolumeType
from ginkgo.plugins.builtin import BUILTIN
from ginkgo.plugins.registry import PluginRegistry, use_plugin_registry
from ginkgo.runtime.evaluator import ConcurrentEvaluator
from ginkgo.runtime.executor_registry import ExecutorRegistry
from ginkgo.runtime.rundir import RunDir
from ginkgo.runtime.volumes import (
    VolumeConfigError,
    _Remapper,
    default_mountpoint,
    open_driver_volumes,
    volume_specs,
)


def _digests(root: Path) -> dict[str, str]:
    """Each file's contents and time: a rewrite is a change, as it is to an object store."""
    return {
        path.relative_to(root).as_posix(): (
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}:{path.stat().st_mtime_ns}"
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _within(relative: str, scope: str) -> bool:
    return not scope or relative == scope or relative.startswith(scope + "/")


class CopyMount:
    """A volume with explicit publication, for tests: a task's mount is a private copy.

    The driver's mount is the source directory itself, so it always sees what
    is published. Every call is logged to ``settings["log"]``.
    """

    def __init__(self, context: VolumeContext) -> None:
        self._source = Path(context.settings["source"])
        self._log = Path(context.settings["log"])
        self._role = context.role
        self._private: Path | None = None
        if context.mountpoint is not None:
            assert context.mountpoint == self._source
            self._path = context.mountpoint
        else:
            self._private = Path(tempfile.mkdtemp(prefix="copy-volume-"))
            self._path = self._private / "mnt"
            shutil.copytree(self._source, self._path, symlinks=True)
            self._seen = _digests(self._path)
        self._record("mount", path=str(self._path))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def source(self) -> str:
        return f"copy://{self._source}/"

    def _record(self, operation: str, **fields: Any) -> None:
        with self._log.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"role": self._role, "op": operation, **fields}) + "\n")

    def publish(self, paths: Any = None) -> None:
        self._record("publish", paths=None if paths is None else list(paths))
        if self._private is None:
            return  # The driver's writes went straight to the source.
        scopes = [""] if paths is None else list(paths)
        current = _digests(self._path)
        for relative in sorted(set(current) | set(self._seen)):
            if not any(_within(relative, scope) for scope in scopes):
                continue
            if current.get(relative) == self._seen.get(relative):
                continue
            target = self._source / relative
            if relative in current:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._path / relative, target)
                self._seen[relative] = current[relative]
            else:
                target.unlink(missing_ok=True)
                self._seen.pop(relative, None)

    def refresh(self, paths: Any) -> None:
        self._record("refresh", paths=list(paths))

    def digest(self, path: str) -> str | None:
        # Like an ETag: the published file's identity, new with every write.
        target = self._source / path
        if not target.is_file():
            return None
        stat = target.stat()
        return f"{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}"

    def close(self) -> None:
        self._record("close")
        if self._private is not None:
            shutil.rmtree(self._private)


COPY_VOLUMES = Plugin(
    name="copy-volumes",
    contributions=(VolumeType(name="copy", label="Copy", mount=CopyMount),),
)


@pytest.fixture(autouse=True)
def copy_volumes():
    with use_plugin_registry(
        PluginRegistry(plugins=(BUILTIN, STUB_OBJECT_STORES, COPY_VOLUMES))
    ) as registry:
        yield registry


@pytest.fixture
def bucket(tmp_path):
    source = tmp_path / "bucket"
    source.mkdir()
    return source


def calls(log: Path, **match: Any) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    return [entry for entry in entries if all(entry.get(k) == v for k, v in match.items())]


def volume_config(bucket: Path) -> dict[str, Any]:
    """Volume ``data`` over *bucket*, which the driver sees in place."""
    table = {
        "type": "copy",
        "source": str(bucket),
        "log": str(bucket.parent / "log"),
        "mountpoint": str(bucket),
    }
    return {"volumes": {"data": table}}


def run(
    expr_fn: Any,
    *,
    bucket: Path,
    executor: SharedFilesystemExecutor | None = None,
    local: bool = False,
    **settings: Any,
) -> Any:
    """Evaluate the expression *expr_fn* builds on the cluster, or locally.

    Outputs are not copied into the artifact store unless *settings* say so.
    """
    executor = executor or SharedFilesystemExecutor()
    with open_driver_volumes(volume_config(bucket), project_root=Path.cwd()) as volumes:
        run_dir = RunDir.create(run_id=uuid.uuid4().hex, root=Path.cwd() / "runs")
        options: dict[str, Any] = {
            "jobs": 2,
            "run_dir": run_dir,
            "volumes": volumes,
            "output_copies": False,
            **settings,
        }
        if not local:
            options["executor_registry"] = ExecutorRegistry.for_executor(executor, name="cluster")
        evaluator = ConcurrentEvaluator(**options)
        return evaluator.evaluate(expr_fn(ginkgo.volume("data").path))


@task()
def produce(text: str, out: Out[file]):
    # The task writes through its own mount, not the driver's.
    assert str(out).startswith(str(ginkgo.volume("data").path) + "/")
    Path(out).write_text(text)


@task()
def shout(src: file, out: Out[file]):
    Path(out).write_text(Path(src).read_text().upper())


@task()
def fail_after_writing(out: Out[file], early: str):
    Path(early).write_text("kept")
    ginkgo.publish(early)
    Path(out).write_text("discarded")
    raise RuntimeError("the task failed")


@task()
def fill_folder(names: list[str], out: Out[folder]):
    Path(out).mkdir(exist_ok=True)
    for name in names:
        (Path(out) / name).write_text(name)


@task()
def plan_shouts(root: str) -> list[file]:
    return [shout(src=f"{root}/in/{name}", out=f"{root}/out/{name}") for name in ("a", "b")]


def test_task_writes_through_its_own_mount_and_publishes_on_success(bucket):
    executor = SharedFilesystemExecutor()

    def flow(root):
        first = produce(text="hello", out=f"{root}/results/hello.txt")
        return shout(src=first, out=f"{root}/results/loud.txt")

    result = run(flow, bucket=bucket, executor=executor)

    # The result names the driver's paths, and the bucket holds both outputs.
    assert str(result) == str(bucket / "results/loud.txt")
    assert "copy-volume-" not in str(result)
    assert (bucket / "results/hello.txt").read_text() == "hello"
    assert (bucket / "results/loud.txt").read_text() == "HELLO"
    # Each attempt mounted the volume and published; the driver refreshed each output.
    log = bucket.parent / "log"
    assert len(calls(log, role="task", op="mount")) == 2
    assert len(calls(log, role="task", op="publish", paths=None)) == 2
    refreshed = [entry["paths"] for entry in calls(log, role="driver", op="refresh")]
    assert ["results/hello.txt"] in refreshed and ["results/loud.txt"] in refreshed
    # The payload carried the driver's paths; the worker rewrote them.
    driver_root = str(bucket)
    assert all(attempt["volumes"]["data"]["root"] == driver_root for attempt in executor.attempts)


def test_unchanged_rerun_hits_the_cache_without_mounting(bucket):
    def flow(root):
        return produce(text="hello", out=f"{root}/results/hello.txt")

    run(flow, bucket=bucket)
    executor = SharedFilesystemExecutor()
    run(flow, bucket=bucket, executor=executor)

    assert executor.attempts == []


def test_failed_task_publishes_only_what_it_published_itself(bucket):
    def flow(root):
        return fail_after_writing(out=f"{root}/out.txt", early=f"{root}/early.txt")

    with pytest.raises(Exception, match="the task failed"):
        run(flow, bucket=bucket)

    assert (bucket / "early.txt").read_text() == "kept"
    assert not (bucket / "out.txt").exists()


def test_declared_outputs_are_cleared_in_the_task_mount(bucket):
    (bucket / "results/listing").mkdir(parents=True)
    (bucket / "results/listing/stale").write_text("from an earlier run")

    def flow(root):
        return fill_folder(names=["a", "b"], out=f"{root}/results/listing")

    run(flow, bucket=bucket)

    assert sorted(p.name for p in (bucket / "results/listing").iterdir()) == ["a", "b"]


def test_tasks_a_dynamic_task_returns_get_the_drivers_paths(bucket):
    (bucket / "in").mkdir()
    (bucket / "in/a").write_text("a")
    (bucket / "in/b").write_text("b")

    results = run(lambda root: plan_shouts(root=str(root)), bucket=bucket)

    assert [Path(r).name for r in results] == ["a", "b"]
    assert all("copy-volume-" not in str(r) for r in results)
    assert (bucket / "out/a").read_text() == "A"
    assert (bucket / "out/b").read_text() == "B"


@task(kind="shell")
def copy_on_driver(src: file, out: Out[file]):
    return shell(cmd=f"cp {src} {out}")


def test_command_the_driver_runs_is_published_by_the_drivers_mount(bucket):
    (bucket / "in.txt").write_text("x")

    run(
        lambda root: copy_on_driver(src=f"{root}/in.txt", out=f"{root}/out.txt"),
        bucket=bucket,
        local=True,
    )

    assert (bucket / "out.txt").read_text() == "x"
    log = bucket.parent / "log"
    assert calls(log, role="driver", op="publish", paths=["out.txt"])
    assert not calls(log, role="task")


@pytest.fixture
def content_reads(monkeypatch):
    """Every path whose bytes ginkgo hashes."""
    from ginkgo.core import hashing
    from ginkgo.runtime.caching import hash_memo

    read: list[Path] = []

    def recording(original):
        def hash_reading(path):
            read.append(Path(path).resolve())
            return original(path)

        return hash_reading

    monkeypatch.setattr(hash_memo, "hash_file", recording(hashing.hash_file))
    monkeypatch.setattr(hash_memo, "hash_directory", recording(hashing.hash_directory))
    return read


def test_driver_names_files_in_the_volume_by_their_objects_without_reading(bucket, content_reads):
    def flow(root):
        first = produce(text="hello", out=f"{root}/results/hello.txt")
        listing = fill_folder(names=["a", "b"], out=f"{root}/results/listing")
        return [shout(src=first, out=f"{root}/results/loud.txt"), listing]

    run(flow, bucket=bucket)
    executor = SharedFilesystemExecutor()
    run(flow, bucket=bucket, executor=executor)

    assert executor.attempts == []  # The rerun found every output unchanged.
    assert not [path for path in content_reads if path.is_relative_to(bucket.resolve())]


def test_a_rewritten_output_counts_as_changed_even_with_the_same_bytes(bucket):
    def flow(root):
        first = produce(text="hello", out=f"{root}/results/hello.txt")
        return shout(src=first, out=f"{root}/results/loud.txt")

    run(flow, bucket=bucket)
    executor = SharedFilesystemExecutor()
    run(flow, bucket=bucket, executor=executor, rerun=frozenset({"produce"}))

    # An object's identity is new with every write, so the consumer runs again.
    assert [attempt["task_name"].rsplit(".", 1)[-1] for attempt in executor.attempts] == [
        "produce",
        "shout",
    ]


def test_copied_outputs_keep_content_digests(bucket, content_reads):
    run(
        lambda root: produce(text="hello", out=f"{root}/results/hello.txt"),
        bucket=bucket,
        output_copies=True,
    )

    assert (bucket / "results/hello.txt").resolve() in content_reads


def test_volume_is_only_available_during_a_run():
    with pytest.raises(RuntimeError, match="while ginkgo runs"):
        ginkgo.volume("data")


def test_publish_outside_volumes_does_nothing(tmp_path):
    ginkgo.publish(tmp_path / "anything")


def test_unknown_volume_names_the_configured_ones(bucket):
    with (
        open_driver_volumes(volume_config(bucket), project_root=Path.cwd()),
        pytest.raises(VolumeConfigError, match="Available: data"),
    ):
        ginkgo.volume("other")


def test_driver_mountpoint_is_fixed_per_project_and_volume(tmp_path):
    one = default_mountpoint(project_root=tmp_path, name="data")
    assert one == default_mountpoint(project_root=tmp_path, name="data")
    assert one != default_mountpoint(project_root=tmp_path / "other", name="data")
    assert one.name == "data"
    assert one.parent.parent.name == f"ginkgo-{os.getuid()}"


def test_volume_tables_are_checked(tmp_path):
    specs = volume_specs(
        {"volumes": {"data": {"type": "copy", "source": "x", "mountpoint": "/mnt/data"}}},
        project_root=tmp_path,
    )
    assert specs["data"].settings == {"source": "x"}
    assert specs["data"].mountpoint == Path("/mnt/data")
    for tables, message in [
        ({"data": {"source": "x"}}, "needs a type"),
        ({"da ta": {"type": "copy"}}, "letters, digits"),
        ({"data": {"type": "copy", "mountpoint": "relative"}}, "absolute"),
        ({"data": "copy"}, "must be a table"),
    ]:
        with pytest.raises(VolumeConfigError, match=message):
            volume_specs({"volumes": tables}, project_root=tmp_path)


def test_remapper_rewrites_paths_below_roots_only():
    remap = _Remapper([("/drv/data", "/task/mnt"), ("/drv/data/deeper", "/other")])
    value = {
        "file": file("/drv/data/x.txt"),
        "folder": folder("/drv/data/dir"),
        "root": "/drv/data",
        "deeper": "/drv/data/deeper/y",
        "sibling": "/drv/database/z",
        "path": PurePosixPath("/drv/data/p"),
        "nested": [("/drv/data/a", 1)],
        "untouched": ["/elsewhere"],
    }

    mapped = remap(value)

    assert mapped["file"] == "/task/mnt/x.txt" and type(mapped["file"]) is file
    assert mapped["folder"] == "/task/mnt/dir" and type(mapped["folder"]) is folder
    assert mapped["root"] == "/task/mnt"
    assert mapped["deeper"] == "/other/y"
    assert mapped["sibling"] == "/drv/database/z"
    assert mapped["path"] == PurePosixPath("/task/mnt/p")
    assert mapped["nested"] == [("/task/mnt/a", 1)]
    assert mapped["untouched"] is value["untouched"]
