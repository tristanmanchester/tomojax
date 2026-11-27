"""CLI terminal records, with durable array references readable after process exit."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import h5py
import numpy as np

from tomojax.io.api import registered_members
from tomojax.wire.arrays import (
    Array,
    Axis,
    Npy,
    Storage,
    Uniform,
)
from tomojax.wire.progress import Result

if TYPE_CHECKING:
    from collections.abc import Mapping

    import tomojax as tj


class TerminalFiles:
    """Reuse the written dataset when possible. Additional arrays live beside the output."""

    def __init__(self, path: Path, cleanup: ExitStack) -> None:
        self.path = path
        self.cleanup = cleanup
        self.directory: Path | None = None
        self.paths: list[str] = []

    def reference(self, value: object) -> Storage:
        from tomojax.core.files import replaced_when_complete

        if self.directory is None:
            self.directory = Path(
                tempfile.mkdtemp(prefix=f"{self.path.name}.arrays-", dir=self.path.parent)
            )
            self.cleanup.callback(shutil.rmtree, self.directory)
        path = self.directory / f"{len(self.paths)}.npy"
        with replaced_when_complete(path, "array") as part, part.open("wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
        self.paths.append(str(path))
        return Npy(str(path))

    def diagnostic(self, pointer: str, value: object) -> Array:
        data = np.asarray(value)
        axes = tuple(
            Axis(f"axis{i}", "diagnostic", "index", Uniform(0, 1)) for i in range(data.ndim)
        )
        return Array(self.reference(data), data.shape, data.dtype.str, axes, "unspecified", pointer)


def scan_product_info(
    item: tj.Scan | tj.Alignment | tj.Reconstruction | tj.Simulation,
) -> dict[str, object]:
    """Diagnostics and ordered correction records for a product carrying a scan."""
    import tomojax as tj

    info = dict(item.info) if isinstance(item, tj.Reconstruction | tj.Alignment) else {}
    scan = item if isinstance(item, tj.Scan) else item.scan
    if scan is not None:
        info["corrections"] = [step.to_dict() for step in scan.corrections]
    return info


def saved_result(
    path: str | Path,
    item: tj.Scan | tj.Alignment | tj.Reconstruction | tj.Simulation,
    *,
    outputs: tuple[str, ...],
    method: str | None = None,
) -> Result:
    """Describe the arrays actually saved and externalize array-valued diagnostics."""
    import tomojax as tj
    from tomojax.cli.execution import summarize
    from tomojax.recon.api import solver_name

    with ExitStack() as cleanup:
        files = TerminalFiles(Path(path), cleanup)
        if isinstance(item, tj.Reconstruction):
            method = solver_name(item.method, item.geometry)
        with h5py.File(path, "r") as handle:
            arrays = registered_members(handle, files.path)
        summary = summarize(scan_product_info(item), externalize=files.diagnostic)
        result = Result(2, (*outputs, *files.paths), method, arrays, summary)
        _ = cleanup.pop_all()
        return result


def report_result(
    path: Path,
    outputs: tuple[str, ...],
    report: Mapping[str, object],
    *,
    arrays: dict[str, Array] | None = None,
) -> Result:
    """Terminal record for a report or an interchange export."""
    from tomojax.cli.execution import summarize

    with ExitStack() as cleanup:
        files = TerminalFiles(path, cleanup)
        info = summarize(report, externalize=files.diagnostic)
        record = Result(2, (*outputs, *files.paths), None, arrays or {}, info)
        _ = cleanup.pop_all()
        return record
