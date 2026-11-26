"""Run one task attempt inside a Slurm job: ``python -m ginkgo_slurm.worker PAYLOAD RESULT``.

The job starts in the project root, as the driver runs, so the payload's
paths mean what they mean on the driver. The response is written beside the
payload, atomically, for the driver to read once the job has finished.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(payload_path: str, result_path: str) -> int:
    from ginkgo.runtime.worker import error_response, run_task

    payload_file = Path(payload_path)
    try:
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
        # A command a driver-kind task runs keeps its run directory beside the
        # transport directory, which the driver reads back from the shared filesystem.
        payload.setdefault("run_dir_root", str(Path(payload["transport_dir"]) / "runs"))
        response = run_task(payload)
    except BaseException as exc:  # noqa: BLE001 - the driver must learn why, whatever happened
        response = error_response(exc)
    finally:
        # The payload holds resolved secrets; nothing needs it once the task ran.
        payload_file.unlink(missing_ok=True)

    result = Path(result_path)
    partial = result.with_name(f".{result.name}.partial")
    partial.write_text(json.dumps(response), encoding="utf-8")
    os.replace(partial, result)
    return 0 if response.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
