"""CLI terminal records, with durable array references readable after process exit."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from tomojax._loading.registration import registered_member, unit_of
from tomojax.core.registration import (
    pose_columns,
    recorded_length_unit,
    volume_values,
)
from tomojax.io.api import registered_members
from tomojax.wire.arrays import (
    Array,
    Axis,
    Coordinates,
    Hdf5,
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

    def reference(self, value: object, dataset: str | None = None) -> Storage:
        if dataset is not None and self.path.suffix.lower() != ".npz":
            return Hdf5(str(self.path), dataset)
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


def _decode(value: object, default: str) -> str:
    if value is None:
        return default
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def stored_grid(handle: h5py.File, path: Path) -> tuple[tj.Grid | None, dict, dict, str]:
    """Read volume placement and disk axis order without touching its pixels."""
    from tomojax.io.api import saved_volume_grid

    grid_meta = json.loads(_decode(handle["/entry"].attrs.get("grid_meta_json"), "{}"))
    geom_group = handle.get("/entry/geometry")
    geom_meta = (
        json.loads(_decode(geom_group.attrs.get("geometry_meta_json"), "{}"))
        if isinstance(geom_group, h5py.Group)
        else {}
    )
    grid = saved_volume_grid(grid_meta, geom_meta)
    return grid, grid_meta, geom_meta, stored_axes(handle, path)


def stored_axes(handle: h5py.File, path: Path) -> str:
    from tomojax.io.api import saved_volume_axes

    attribute = handle["/entry/processing/tomojax"].attrs.get("volume_axes_order")
    return saved_volume_axes(None if attribute is None else _decode(attribute, ""), path)


def disk_volume(
    path: Path,
    grid: tj.Grid | None,
    axes: str,
    shape: tuple[int, ...],
    dtype: str,
    meta: Mapping[str, Any],
) -> Array:
    """Describe the export source in its recorded storage order, including legacy volumes."""
    from tomojax.cli.execution import volume_metadata

    storage = Hdf5(str(path), "/entry/processing/tomojax/volume")
    if grid is None:
        disk_axes = tuple(Axis(a, "stored_volume", "index", Uniform(0, 1)) for a in axes)
        unit = "unspecified"
    else:
        volume = volume_metadata(
            grid,
            storage=storage,
            dtype=dtype,
            length_unit=recorded_length_unit(meta) or "scan_length",
            geometry_meta=meta,
        )
        disk_axes = tuple(volume.axes["zyx".index(a)] for a in axes)
        unit = volume.unit
    _, domain = volume_values(np.dtype(dtype), recorded_length_unit(meta) or "scan_length", meta)
    return Array(storage, shape, dtype, disk_axes, unit, domain)


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
    include_scan: bool = True,
) -> Result:
    """Describe the arrays actually saved and externalize array-valued diagnostics."""
    import tomojax as tj
    from tomojax.cli.execution import summarize
    from tomojax.recon.api import solver_name

    with ExitStack() as cleanup:
        files = TerminalFiles(Path(path), cleanup)
        scan = item if isinstance(item, tj.Scan) else item.scan
        unit = unit_of(item if isinstance(item, tj.Reconstruction) else scan)
        arrays: dict[str, Array] = {}
        if isinstance(item, tj.Reconstruction):
            method = solver_name(item.method, item.geometry)
        if files.path.suffix.lower() != ".npz":
            with h5py.File(path, "r") as handle:
                arrays = registered_members(handle, files.path)
        elif (
            not isinstance(item, tj.Scan) and (volume := registered_member(item, unit)) is not None
        ):
            arrays[volume.name] = volume.at(files.reference(volume.pixels))
        if include_scan and scan is not None and "projections" not in arrays:
            view = Axis("view", "acquisition", "degree", Coordinates(files.reference(scan.angles)))
            detector = registered_member(scan, unit, view=view)
            assert detector is not None
            arrays[detector.name] = detector.at(files.reference(detector.pixels))
        if isinstance(item, tj.Alignment) and "poses" not in arrays:
            poses = item.poses
            arrays["poses"] = Array(
                files.reference(poses),
                poses.shape,
                poses.dtype.str,
                (arrays["projections"].axes[0], Axis("column", "pose", "index", Uniform(0, 1))),
                "mixed",
                "poses",
                columns=pose_columns(unit),
            )
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
