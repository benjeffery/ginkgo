"""Volumes: named storage that every process running tasks mounts for itself.

A volume is a ``[volumes.<name>]`` table in ``ginkgo.toml`` or a ``--config``
file, whose ``type`` names a :class:`~ginkgo.plugins.api.VolumeType` — an
object-store prefix mounted through FUSE, say. Unlike a shared filesystem, a
mount keeps what it writes private until it publishes it, and every process
mounts the volume at its own path. Ginkgo bridges both differences, so task
bodies keep reading and writing ordinary paths:

- The driver mounts each volume once per run, at a fixed mountpoint
  (:func:`default_mountpoint`). The workflow builds its paths below
  :func:`volume`'s ``path``, and the driver hashes and checks outputs there.
- Each task attempt that refers to a volume mounts it again, wherever it runs:
  in a local worker process or a remote job. Paths below the driver's
  mountpoint in its arguments are rewritten to its own mount, and paths below
  its mount in its result are rewritten back. Its declared outputs are cleared
  there, rather than on the driver, whose deletions nobody else would see.
- When the attempt succeeds, everything it wrote is published, then its mount
  closes. A failed attempt publishes nothing: its writes are discarded with
  its mount. :func:`publish` publishes paths earlier, for a long task whose
  finished pieces should survive a later failure.
- Before the driver checks a task's outputs, it refreshes them in its own
  mount, so it sees what the task published.
- When outputs are not copied into the artifact store, the driver identifies
  a file in a volume by the object it is stored as (:meth:`DriverVolumes.digest`)
  rather than by reading it back.

The driver's mountpoint is part of every path the workflow passes, and so of
cache keys built from path strings. It is therefore fixed per project, user
and volume, and the same on every host.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

from ginkgo.core.expr import Expr, ExprList
from ginkgo.core.hashing import hash_str
from ginkgo.core.types import file, folder
from ginkgo.errors import GinkgoError
from ginkgo.plugins.api import MountedVolume, VolumeContext, VolumeType
from ginkgo.plugins.registry import plugin_registry

__all__ = [
    "DriverVolumes",
    "TaskVolumes",
    "VolumeConfigError",
    "VolumeInfo",
    "VolumeSpec",
    "default_mountpoint",
    "open_driver_volumes",
    "publish",
    "task_volume_scope",
    "volume",
    "volume_specs",
]

logger = logging.getLogger(__name__)

VOLUMES_CONFIG_KEY = "volumes"
"""Runtime config table holding one table per volume."""

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class VolumeConfigError(GinkgoError, ValueError):
    """A ``[volumes]`` table ginkgo cannot use."""


@dataclass(frozen=True, kw_only=True)
class VolumeSpec:
    """One configured volume.

    Parameters
    ----------
    name : str
        The table's name, ``[volumes.<name>]``.
    type : str
        The :class:`~ginkgo.plugins.api.VolumeType` that mounts it.
    settings : dict[str, Any]
        The rest of the table, for the volume type.
    mountpoint : Path
        Where the driver mounts it.
    """

    name: str
    type: str
    settings: dict[str, Any]
    mountpoint: Path


@dataclass(frozen=True, kw_only=True)
class VolumeInfo:
    """A volume as the calling process sees it.

    Parameters
    ----------
    name : str
        The volume's name.
    path : Path
        Where this process sees the volume's root: the driver's mountpoint in
        a flow, the task's own mount in a task body.
    source : str
        What is mounted, as a URI.
    """

    name: str
    path: Path
    source: str


def default_mountpoint(*, project_root: Path, name: str) -> Path:
    """Return where the driver mounts volume *name* unless its table sets ``mountpoint``.

    Below node-local ``/tmp``, since a FUSE mount on a network home directory
    is often refused, and fixed for the project, the user and the volume, so a
    driver on any host passes tasks the same paths and finds its cache entries.
    """
    root = project_root.resolve()
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    base = Path(os.path.realpath("/tmp")) / f"ginkgo-{os.getuid()}"
    return base / f"{root.name}-{digest}" / name


def volume_specs(
    runtime_config: Mapping[str, Any], *, project_root: Path
) -> dict[str, VolumeSpec]:
    """Read the ``[volumes]`` tables of *runtime_config*.

    Raises
    ------
    VolumeConfigError
        When a table is malformed. Whether its ``type`` is installed is
        checked when it is mounted.
    """
    tables = runtime_config.get(VOLUMES_CONFIG_KEY) or {}
    if not isinstance(tables, Mapping):
        raise VolumeConfigError("[volumes] must hold one table per volume, such as [volumes.data]")
    specs: dict[str, VolumeSpec] = {}
    for name, table in tables.items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise VolumeConfigError(
                f"Volume name {name!r} may use only letters, digits, '-' and '_'"
            )
        if not isinstance(table, Mapping):
            raise VolumeConfigError(f"[volumes.{name}] must be a table")
        settings = dict(table)
        kind = settings.pop("type", None)
        if not isinstance(kind, str) or not kind:
            raise VolumeConfigError(
                f"[volumes.{name}] needs a type: the name of a volume type a plugin provides"
            )
        mountpoint = settings.pop("mountpoint", None)
        if mountpoint is None:
            path = default_mountpoint(project_root=project_root, name=name)
        elif isinstance(mountpoint, str) and Path(mountpoint).expanduser().is_absolute():
            path = Path(mountpoint).expanduser()
        else:
            raise VolumeConfigError(f"[volumes.{name}] mountpoint must be an absolute path")
        specs[name] = VolumeSpec(name=name, type=kind, settings=settings, mountpoint=path)
    return specs


# ----- Paths ------------------------------------------------------------------


def _below(text: str, root: str) -> str | None:
    """Return *text*'s path relative to *root*, ``""`` for the root, or ``None``."""
    if text == root:
        return ""
    if text.startswith(root + "/"):
        return text[len(root) + 1 :]
    return None


def _strings(value: Any) -> Iterator[str]:
    """Yield every string a task argument or result holds, as :class:`_Remapper` sees them."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, PurePath):
        yield str(value)
    elif type(value) in (list, tuple, set, frozenset):
        for item in value:
            yield from _strings(item)
    elif type(value) is dict:
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, Expr):
        for item in value.args.values():
            yield from _strings(item)
    elif isinstance(value, ExprList):
        for item in value.exprs:
            yield from _strings(item)


class _Remapper:
    """Rewrite paths below one set of roots to the same paths below others.

    It rewrites strings (keeping ``file``, ``folder`` and other ``str``
    subclasses), :class:`~pathlib.PurePath` objects, the builtin containers and
    the arguments of task expressions a dynamic task returns. Anything else
    passes through unchanged, and so does any container in which nothing
    changed. A value held twice maps to one new value.
    """

    def __init__(self, pairs: Iterable[tuple[str, str]]) -> None:
        self._pairs = sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)
        self._memo: dict[int, Any] = {}

    def __call__(self, value: Any) -> Any:
        return self._map(value) if self._pairs else value

    def _string(self, text: str) -> str | None:
        for source, target in self._pairs:
            relative = _below(text, source)
            if relative is not None:
                return target if not relative else f"{target}/{relative}"
        return None

    def _map(self, value: Any) -> Any:
        if isinstance(value, str):
            mapped = self._string(value)
            if mapped is None:
                return value
            return mapped if type(value) is str else type(value)(mapped)
        if isinstance(value, PurePath):
            mapped = self._string(str(value))
            return value if mapped is None else type(value)(mapped)
        key = id(value)
        if key in self._memo:
            return self._memo[key]
        result = value
        if type(value) in (list, tuple, set, frozenset):
            items = [self._map(item) for item in value]
            if any(new is not old for new, old in zip(items, value, strict=True)):
                result = type(value)(items)
        elif type(value) is dict:
            entries = {name: self._map(item) for name, item in value.items()}
            if any(entries[name] is not item for name, item in value.items()):
                result = entries
        elif isinstance(value, Expr):
            args = {name: self._map(item) for name, item in value.args.items()}
            if any(args[name] is not item for name, item in value.args.items()):
                result = dataclasses.replace(value, args=args)
        elif isinstance(value, ExprList):
            exprs = [self._map(item) for item in value.exprs]
            if any(new is not old for new, old in zip(exprs, value.exprs, strict=True)):
                result = dataclasses.replace(value, exprs=exprs)
        self._memo[key] = result
        return result


def output_paths(*, declared: Iterable[str], value: Any) -> list[str]:
    """Return a task's declared ``Out[...]`` paths and the files and folders its result names."""
    paths = [str(path) for path in declared]
    paths.extend(text for text in _strings(value) if isinstance(text, (file, folder)))
    return paths


def _group(
    paths: Iterable[str | os.PathLike[str]], roots: Mapping[str, str]
) -> dict[str, list[str]]:
    """Sort *paths* by the volume they lie in, as paths relative to its root.

    A path outside every root, or one that climbs out of its root, is left out.
    """
    grouped: dict[str, list[str]] = {}
    for path in paths:
        text = os.fspath(path)
        for name, root in sorted(roots.items(), key=lambda item: len(item[1]), reverse=True):
            relative = _below(text, root)
            if relative is None:
                continue
            relative = os.path.normpath(relative) if relative else ""
            if relative == ".":
                relative = ""
            if relative == ".." or relative.startswith("../"):
                break
            if relative not in grouped.setdefault(name, []):
                grouped[name].append(relative)
            break
    return grouped


def _mount(*, spec_type: str, context: VolumeContext) -> MountedVolume:
    volume_type = plugin_registry().get(VolumeType, spec_type)
    return volume_type.mount(context)


def _close(name: str, mount: MountedVolume) -> None:
    """Close *mount*, logging rather than raising: closing must not mask the real outcome."""
    try:
        mount.close()
    except Exception:
        logger.warning("Closing volume %r failed", name, exc_info=True)


# ----- The driver -------------------------------------------------------------


class DriverVolumes:
    """The run's own mount of each configured volume.

    Parameters
    ----------
    specs : dict[str, VolumeSpec]
        The configured volumes.
    mounts : dict[str, MountedVolume]
        Each one's mount at its spec's mountpoint.
    """

    def __init__(self, *, specs: dict[str, VolumeSpec], mounts: dict[str, MountedVolume]) -> None:
        self._specs = specs
        self._mounts = mounts
        self.roots = {name: str(mount.path) for name, mount in mounts.items()}

    def info(self, name: str) -> VolumeInfo:
        """Return volume *name* as the driver sees it."""
        mount = self._mounts.get(name)
        if mount is None:
            raise _unknown_volume(name, self._mounts)
        return VolumeInfo(name=name, path=Path(self.roots[name]), source=mount.source)

    def payload(self) -> dict[str, dict[str, Any]]:
        """Describe the volumes for a worker, which mounts the ones its task uses."""
        return {
            name: {
                "type": spec.type,
                "settings": spec.settings,
                "root": self.roots[name],
                "source": self._mounts[name].source,
            }
            for name, spec in self._specs.items()
        }

    def owns(self, path: str | os.PathLike[str]) -> bool:
        """Whether *path* lies in a volume."""
        return bool(_group([path], self.roots))

    def digest(self, path: str | os.PathLike[str]) -> str | None:
        """Return a digest naming the object the file at *path* is stored as, if known.

        It folds in the volume's source and the file's path, since an object
        identity such as an ETag is only meaningful for one key, and is
        prefixed so that it can never equal a content digest.
        """
        grouped = _group([path], self.roots)
        if not grouped:
            return None
        [(name, [relative])] = grouped.items()
        if not relative:
            return None
        mount = self._mounts[name]
        identity = mount.digest(relative)
        if identity is None:
            return None
        return hash_str(f"ginkgo-volume-object\0{mount.source}\0{relative}\0{identity}")

    def acquire(self, paths: Iterable[str | os.PathLike[str]]) -> None:
        """See what a task published at *paths*, before checking or hashing them."""
        for name, relative in _group(paths, self.roots).items():
            self._mounts[name].refresh(relative)

    def publish(self, paths: Iterable[str | os.PathLike[str]]) -> None:
        """Publish what this process wrote at *paths*: a task it ran itself, or a restore."""
        for name, relative in _group(paths, self.roots).items():
            self._mounts[name].publish(relative)

    def close(self) -> None:
        """Unmount every volume."""
        for name, mount in reversed(self._mounts.items()):
            _close(name, mount)


_driver: DriverVolumes | None = None
"""The run's volumes, read from any thread of the driver."""


@contextlib.contextmanager
def open_driver_volumes(
    runtime_config: Mapping[str, Any], *, project_root: Path
) -> Iterator[DriverVolumes | None]:
    """Mount the configured volumes for a run, and unmount them after it.

    Yields ``None`` when no volume is configured.
    """
    global _driver
    specs = volume_specs(runtime_config, project_root=project_root)
    if not specs:
        yield None
        return
    mounts: dict[str, MountedVolume] = {}
    volumes = None
    try:
        for spec in specs.values():
            spec.mountpoint.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            context = VolumeContext(
                name=spec.name, settings=spec.settings, mountpoint=spec.mountpoint, role="driver"
            )
            mounts[spec.name] = _mount(spec_type=spec.type, context=context)
        volumes = DriverVolumes(specs=specs, mounts=mounts)
        previous, _driver = _driver, volumes
        try:
            yield volumes
        finally:
            _driver = previous
    finally:
        if volumes is not None:
            volumes.close()
        else:
            for name, mount in reversed(mounts.items()):
                _close(name, mount)


# ----- A task attempt ---------------------------------------------------------


class TaskVolumes:
    """The volumes one task attempt sees, each mounted when it is first used.

    Parameters
    ----------
    specs : Mapping[str, Mapping[str, Any]]
        :meth:`DriverVolumes.payload`.
    """

    def __init__(self, specs: Mapping[str, Mapping[str, Any]]) -> None:
        self._specs = specs
        self._mounts: dict[str, MountedVolume] = {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaskVolumes | None:
        """Return a worker payload's volumes, or ``None`` when the run has none."""
        specs = payload.get("volumes")
        return cls(specs) if specs else None

    def mount(self, name: str) -> MountedVolume:
        """Return this attempt's mount of volume *name*, mounting it on first use."""
        mount = self._mounts.get(name)
        if mount is None:
            spec = self._specs.get(name)
            if spec is None:
                raise _unknown_volume(name, self._specs)
            context = VolumeContext(
                name=name, settings=spec["settings"], mountpoint=None, role="task"
            )
            mount = self._mounts[name] = _mount(spec_type=spec["type"], context=context)
        return mount

    def info(self, name: str) -> VolumeInfo:
        """Return volume *name* as this attempt sees it."""
        mount = self.mount(name)
        return VolumeInfo(name=name, path=mount.path, source=mount.source)

    def _task_roots(self) -> dict[str, str]:
        return {name: str(mount.path) for name, mount in self._mounts.items()}

    def enter(self, *, task_def: Any, args: dict[str, Any]) -> dict[str, Any]:
        """Mount what *args* refer to, point them at those mounts, and clear declared outputs there.

        The driver leaves a declared output in a volume alone: its deletion
        would stay in its own mount, so the task's mount would still show
        the output's previous contents.
        """
        for name, spec in self._specs.items():
            if any(_below(text, spec["root"]) is not None for text in _strings(args)):
                self.mount(name)
        driver_to_task = [
            (self._specs[name]["root"], str(mount.path)) for name, mount in self._mounts.items()
        ]
        args = _Remapper(driver_to_task)(args)
        if getattr(task_def, "output_params", None):
            from ginkgo.runtime.task_runners.shell import remove_declared_output
            from ginkgo.runtime.task_validation import declared_output_paths

            roots = self._task_roots()
            for _, path, _ in declared_output_paths(task_def=task_def, resolved_args=args):
                if _group([path], roots):
                    output = Path(path)
                    remove_declared_output(output)
                    output.parent.mkdir(parents=True, exist_ok=True)
        return args

    def leave(self, result: Any) -> Any:
        """Point *result* back at the driver's paths, then publish what this attempt wrote."""
        task_to_driver = [
            (str(mount.path), self._specs[name]["root"]) for name, mount in self._mounts.items()
        ]
        result = _Remapper(task_to_driver)(result)
        for mount in self._mounts.values():
            mount.publish()
        return result

    def publish(self, paths: Iterable[str | os.PathLike[str]]) -> None:
        """Publish what this attempt wrote at *paths* now, rather than when it succeeds."""
        for name, relative in _group(paths, self._task_roots()).items():
            self._mounts[name].publish(relative)

    def close(self) -> None:
        """Unmount, discarding what was not published."""
        for name, mount in reversed(self._mounts.items()):
            _close(name, mount)
        self._mounts.clear()


_task: ContextVar[TaskVolumes | None] = ContextVar("ginkgo_task_volumes", default=None)


@contextlib.contextmanager
def task_volume_scope(volumes: TaskVolumes | None) -> Iterator[TaskVolumes | None]:
    """Make *volumes* what :func:`volume` and :func:`publish` use while a task body runs."""
    token = _task.set(volumes)
    try:
        yield volumes
    finally:
        _task.reset(token)


def _unknown_volume(name: str, known: Mapping[str, Any]) -> GinkgoError:
    available = ", ".join(sorted(known)) or "none"
    return VolumeConfigError(f"No volume named {name!r} is configured. Available: {available}.")


# ----- Public API -------------------------------------------------------------


def volume(name: str) -> VolumeInfo:
    """Return volume *name*: where this process sees it, and what it mounts.

    In a flow, ``volume(name).path`` is the driver's mountpoint, below which
    the workflow builds the paths it passes to tasks. In a task body it is
    the task's own mount, mounted on first use.

    Raises
    ------
    RuntimeError
        Outside a run, where nothing is mounted.
    VolumeConfigError
        When no volume of that name is configured.
    """
    scope = _task.get() or _driver
    if scope is None:
        raise RuntimeError(
            f"ginkgo.volume({name!r}) is only available while ginkgo runs a workflow"
            " that configures [volumes]"
        )
    return scope.info(name)


def publish(*paths: str | os.PathLike[str]) -> None:
    """Publish what this task wrote at *paths* now, rather than when it succeeds.

    For a long task whose finished pieces should survive a later failure.
    Paths outside every volume, and every path when the run has no volumes,
    need no publication, so they are ignored.
    """
    scope: TaskVolumes | DriverVolumes | None = _task.get() or _driver
    if scope is not None:
        scope.publish(paths)
