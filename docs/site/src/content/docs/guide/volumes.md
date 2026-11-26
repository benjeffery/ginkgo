---
title: "Volumes"
---

A volume is named storage that a workflow reads and writes at ordinary paths,
such as a prefix of an object-storage bucket mounted through FUSE. Ginkgo itself
does not mount anything; a volume type, which a plugin provides, does (see
[Writing A Volume Type](#writing-a-volume-type)).

A volume behaves differently from a shared filesystem:

- Each process that uses it mounts it for itself, at its own path.
- What one mount writes stays private to that mount until it is published.

Ginkgo bridges both, so task bodies need no changes.

## Configuring A Volume

Declare each volume as a table in `ginkgo.toml`, or in a `--config` file for
one site:

```toml
[volumes.bucket]
type = "bucketfs"                   # a volume type from an installed plugin
source = "oci://my-bucket@my-namespace/"
```

`type` selects the volume type. The other settings in the table belong to that
type, as its plugin documents them. `mountpoint` is the one setting ginkgo reads
itself (see below).

The flow builds its paths below the run's own mount:

```python
import ginkgo

@ginkgo.flow
def main():
    root = ginkgo.volume("bucket").path
    return summarise(table=f"{root}/inputs/table.parquet", out=f"{root}/results/summary.csv")
```

`ginkgo.volume(name)` returns the volume's `path` in the calling process and
its `source`, the URI that is mounted. It is only available while ginkgo runs a
workflow that configures that volume.

## What Happens In A Run

1. Before the workflow is imported, the run mounts each volume at its
   mountpoint. It then checks inputs, hashes files and finds cache hits through
   that mount.
2. Each task attempt whose arguments refer to a volume mounts it again, in the
   process that runs it: a local worker, or a job on a remote executor such as
   Slurm. Paths below the run's mountpoint in the task's arguments are rewritten
   to the task's own mount. Paths in the task's result are rewritten back,
   including the arguments of task calls that a dynamic task returns.
3. Declared `Out[...]` outputs in a volume are cleared in the task's mount, not
   by the driver. A deletion made through the driver's mount would stay private
   to it, so the task would still see the old contents.
4. When the attempt succeeds, everything it wrote is published, and then its
   mount closes. When it fails, nothing is published, and its writes are
   discarded.
5. Before the driver checks a task's outputs, it refreshes them in its own
   mount so that it sees what the task published.

A shell, script or notebook task that runs its command on the driver writes
through the driver's mount. The driver publishes that task's declared outputs
when it succeeds.

## How Ginkgo Identifies Files In A Volume

Ginkgo normally identifies a file by a hash of its contents. For a file in a
volume, reading the contents back would mean downloading them. When the volume
type can name the stored object, the driver uses that name instead and reads
nothing. A folder is identified by its listing and the names of the files in
it.

This happens only with `[cache] output_copies = false`. An artifact store that
keeps copies files them under their content hashes, so with copies on, the
driver reads files in a volume as it reads any other file.

What the name is depends on the volume type. A checksum of the stored bytes,
such as the MD5 that many object stores keep for each object, means that when a
task runs again and writes identical outputs, the tasks that read them stay
cached, as for any other file. A name that is new with every write, such as an
ETag, would make those tasks run again. Either way, each output is identified without a download, and a later run
recognises it from metadata alone. A content hash keyed by the file's device and
inode numbers would not survive that: a FUSE mount gets new ones every time it
mounts.

## Publishing Early

A long task can publish finished pieces itself, so that they survive a later
failure or a job that is killed:

```python
@task(threads=2, memory="8Gi")
def count_all(samples: list[str], root: untracked) -> list[str]:
    for sample in samples:
        path = f"{root}/counts/{sample}.klc"
        count(sample, path)
        ginkgo.publish(path)
```

`ginkgo.publish(*paths)` ignores paths outside every volume. It also does
nothing in a run without volumes, so the same task body runs on a plain
filesystem.

## The Driver's Mountpoint

Paths below the run's mountpoint are part of tasks' arguments, so they are part
of cache keys too. Ginkgo therefore mounts each volume at a fixed path:

```text
/tmp/ginkgo-<uid>/<project directory name>-<hash of its path>/<volume name>
```

This path is the same on every host, so a run whose driver lands on another
machine finds its cache entries again. It is on node-local `/tmp` because many
sites refuse FUSE mounts on network home directories. Set `mountpoint` in the
volume's table to use another absolute path. Changing it changes every cache key
that includes such a path, so those tasks run again.

Workers mount wherever the volume type puts them. Their paths never reach a
cache key.

## Writing A Volume Type

A plugin contributes a `VolumeType` from `ginkgo.plugins.api`:

```python
from ginkgo.plugins.api import Plugin, VolumeContext, VolumeType


class MyMount:
    def __init__(self, context: VolumeContext) -> None:
        # context.settings: the volume's table; context.role: "driver" or "task";
        # context.mountpoint: the driver's fixed path, or None for a task's mount.
        ...

    path: Path          # where this process sees the root
    source: str         # what is mounted, as a URI

    def publish(self, paths=None): ...   # paths relative to the root; None means all
    def refresh(self, paths): ...        # see what is published, dropping local changes
    def digest(self, path): ...          # the stored object's identity, or None to read it
    def close(self): ...                 # unmount, discarding what was not published


PLUGIN = Plugin(
    name="my-volumes",
    contributions=(VolumeType(name="bucketfs", label="Bucket FUSE mount", mount=MyMount),),
)
```

Workers find the type through the same `ginkgo.plugins` entry point as the
driver, so install the plugin wherever tasks run.
