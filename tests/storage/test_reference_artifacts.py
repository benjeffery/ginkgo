"""``[cache] output_copies = false``: outputs are recorded by digest, never copied."""

from __future__ import annotations

import uuid
from pathlib import Path

from ginkgo import Out, file, folder, task
from ginkgo.runtime.artifacts.artifact_store import REFERENCE_BACKEND, LocalArtifactStore
from ginkgo.runtime.caching.index import CacheIndex
from ginkgo.runtime.evaluator import ConcurrentEvaluator
from ginkgo.runtime.rundir import RunDir
from ginkgo.workspace_layout import WorkspaceLayout


@task()
def write(text: str, out: Out[file]):
    # Bodies run in worker processes: record each execution in the working directory.
    with open("calls.log", "a") as calls:
        calls.write(text)
    Path(out).write_text(text)


def calls() -> str:
    return Path("calls.log").read_text() if Path("calls.log").exists() else ""


@task()
def write_folder(text: str, out: Out[folder]):
    Path(out).mkdir()
    (Path(out) / "part.txt").write_text(text)


def run(expr, *, copies: bool):
    run_dir = RunDir.create(run_id=uuid.uuid4().hex, root=Path.cwd() / "runs")
    return ConcurrentEvaluator(jobs=1, run_dir=run_dir, output_copies=copies).evaluate(expr)


def stored_blobs() -> list[Path]:
    blobs = WorkspaceLayout.for_cwd().artifacts / "blobs"
    return [path for path in blobs.iterdir()] if blobs.exists() else []


def test_outputs_are_recorded_without_copies_and_reruns_hit():
    run([write(text="a", out="a.txt"), write_folder(text="b", out="b")], copies=False)
    assert not stored_blobs()
    run([write(text="a", out="a.txt"), write_folder(text="b", out="b")], copies=False)
    assert calls() == "a"


def test_a_missing_or_edited_output_reruns_its_task_instead_of_being_restored():
    run(write(text="a", out="a.txt"), copies=False)
    Path("a.txt").unlink()
    run(write(text="a", out="a.txt"), copies=False)
    assert calls() == "aa"
    Path("a.txt").write_text("edited by hand")
    run(write(text="a", out="a.txt"), copies=False)
    assert calls() == "aaa"
    assert Path("a.txt").read_text() == "a"


def test_default_copies_still_restore_outputs():
    run(write(text="a", out="a.txt"), copies=True)
    assert stored_blobs()
    Path("a.txt").unlink()
    run(write(text="a", out="a.txt"), copies=True)
    assert calls() == "a"
    assert Path("a.txt").read_text() == "a"


def test_reference_records_pass_integrity_checks_and_upgrade_when_copies_resume(tmp_path):
    index = CacheIndex.in_memory()
    source = tmp_path / "output.txt"
    source.write_text("payload")
    reference = LocalArtifactStore(index=index, root=tmp_path / "store", keep_copies=False)
    record = reference.store(src_path=source)
    assert record.storage_backend == REFERENCE_BACKEND
    assert reference.exists(artifact_id=record.artifact_id)
    assert reference.is_reference(artifact_id=record.artifact_id)
    assert reference.matches(artifact_id=record.artifact_id, path=source)
    assert not reference.restore(artifact_id=record.artifact_id, dest_path=tmp_path / "copy")
    assert reference.integrity_problems() == []

    copying = LocalArtifactStore(index=index, root=tmp_path / "store")
    upgraded = copying.store(src_path=source)
    assert upgraded.storage_backend == "local"
    assert copying.artifact_path(artifact_id=upgraded.artifact_id).exists()
