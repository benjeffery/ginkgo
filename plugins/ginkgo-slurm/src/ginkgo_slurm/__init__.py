"""ginkgo-slurm: the ``slurm`` executor type, which runs tasks as Slurm batch jobs.

Slurm nodes share the driver's filesystem, so a job runs ginkgo's ordinary
worker against the driver's own paths: nothing is staged through an object
store, and outputs and task logs are written where a local task writes them.
"""

from __future__ import annotations

from ginkgo.plugins.api import ExecutorType, Plugin
from ginkgo_slurm.executor import SlurmExecutor, build_slurm_executor

__all__ = ["PLUGIN", "SlurmExecutor"]

PLUGIN = Plugin(
    name="ginkgo-slurm",
    contributions=(
        ExecutorType(
            name="slurm",
            label="Slurm",
            required_settings=(),
            build=build_slurm_executor,
            shared_filesystem=True,
        ),
    ),
)
