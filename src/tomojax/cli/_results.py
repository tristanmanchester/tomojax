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

from tomojax.cli.products import VOLUME_DATASET
from tomojax.core.registration import pose_columns, recorded_length_unit
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

_PROJECTIONS = "/entry/instrument/detector/data"
_ANGLES = "/entry/sample/transformations/rotation_angle"
_POSES = "/entry/processing/tomojax/align/thetas"


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


def unit_of(item: tj.Scan | tj.Frames | tj.Reconstruction | None) -> str:
    """Return the recorded spatial unit or the wire name for uncalibrated lengths."""
    import tomojax as tj

    if isinstance(item, tj.Reconstruction):
        unit = item.length_unit
    else:
        unit = recorded_length_unit(item.source.geometry_metadata) if item and item.source else None
    return unit or "scan_length"


def _decode(value: object, default: str) -> str:
    if value is None:
        return default
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def stored_grid(handle: h5py.File, path: Path) -> tuple[tj.Grid | None, dict, dict, str]:
    """Read volume placement and disk axis order without touching its pixels."""
    from tomojax._files import saved_volume_grid

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

    storage = Hdf5(str(path), VOLUME_DATASET)
    if grid is None:
        disk_axes = tuple(Axis(a, "stored_volume", "index", Uniform(0, 1)) for a in axes)
        unit = "unspecified"
    else:
        volume = volume_metadata(
            grid,
            storage=storage,
            dtype=dtype,
            length_unit=recorded_length_unit(meta) or "scan_length",
        )
        disk_axes = tuple(volume.axes["zyx".index(a)] for a in axes)
        unit = volume.unit
    return Array(storage, shape, dtype, disk_axes, unit, "attenuation")


def scan_axes(det: tj.Detector, view: Axis, unit: str) -> tuple[Axis, Axis, Axis]:
    """Pixel centres in the detector's local frame, in stored (view, v, u) order."""
    return (
        view,
        Axis("v", "detector", unit, Uniform(det.center[1] - (det.nv - 1) * det.dv / 2, det.dv)),
        Axis("u", "detector", unit, Uniform(det.center[0] - (det.nu - 1) * det.du / 2, det.du)),
    )


def saved_result(
    path: str | Path,
    item: tj.Scan | tj.Alignment | tj.Reconstruction | tj.Simulation,
    *,
    outputs: tuple[str, ...],
    method: str | None = None,
    include_scan: bool = True,
) -> Result:
    """Describe the arrays actually saved and externalise array-valued diagnostics."""
    import tomojax as tj
    from tomojax.cli.execution import summarise, volume_metadata
    from tomojax.recon.api import solver_name

    with ExitStack() as cleanup:
        files = TerminalFiles(Path(path), cleanup)
        scan = item if isinstance(item, tj.Scan) else item.scan
        unit = unit_of(item if isinstance(item, tj.Reconstruction) else scan)
        arrays: dict[str, Array] = {}
        info: Mapping[str, object] = {}
        if isinstance(item, tj.Reconstruction):
            method, info = solver_name(item.method, item.geometry), item.info
            grid, volume, key = item.grid, item.volume, "volume"
        elif isinstance(item, tj.Alignment):
            info = item.info
            grid, volume, key = item.grid, item.image, "image"
        elif isinstance(item, tj.Simulation):
            grid, volume, key = item.scan.grid, item.phantom, "phantom"
        else:
            grid, volume, key = None, None, ""
        if grid is not None and volume is not None:
            arrays[key] = volume_metadata(
                grid,
                storage=files.reference(volume, VOLUME_DATASET),
                dtype=np.dtype(volume.dtype).str,
                length_unit=unit,
            )
        if include_scan and scan is not None:
            view = Axis(
                "view", "acquisition", "degree", Coordinates(files.reference(scan.angles, _ANGLES))
            )
            data = scan.projections
            arrays["projections"] = Array(
                files.reference(data, _PROJECTIONS),
                data.shape,
                np.dtype(data.dtype).str,
                scan_axes(scan.detector, view, unit),
                "1",
                "line_integrals",
            )
            if isinstance(item, tj.Alignment):
                poses = item.poses
                storage = files.reference(poses, _POSES if scan.poses is not None else None)
                # NXtomo persists poses in FP32. Synthesised zero poses retain their NPY dtype.
                dtype = np.dtype("float32").str if isinstance(storage, Hdf5) else poses.dtype.str
                columns = pose_columns(unit)
                arrays["poses"] = Array(
                    storage,
                    poses.shape,
                    dtype,
                    (view, Axis("column", "pose", "index", Uniform(0, 1))),
                    "mixed",
                    "poses",
                    columns=columns,
                )
        summary = summarise(info, externalise=files.diagnostic)
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
    from tomojax.cli.execution import summarise

    with ExitStack() as cleanup:
        files = TerminalFiles(path, cleanup)
        info = summarise(report, externalise=files.diagnostic)
        record = Result(2, (*outputs, *files.paths), None, arrays or {}, info)
        _ = cleanup.pop_all()
        return record
