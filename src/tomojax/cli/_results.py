"""CLI terminal records, with durable array references readable after process exit."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, get_args

import numpy as np

from tomojax.wire.arrays import (
    Array,
    Axis,
    Column,
    Coordinates,
    Hdf5,
    Npy,
    PoseColumn,
    Storage,
    Uniform,
)
from tomojax.wire.progress import Result

if TYPE_CHECKING:
    from collections.abc import Mapping

    import tomojax as tj

VOLUME_DATASET = "/entry/processing/tomojax/volume"
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


def length_unit(meta: Mapping[str, Any]) -> str:
    """The recorded acquisition length unit, or an explicitly uncalibrated scan unit."""
    return str(
        meta.get("length_unit")
        or (meta.get("acquisition") or {}).get("length_unit")
        or "scan_length"
    )


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
        volume = volume_metadata(grid, storage=storage, dtype=dtype, length_unit=length_unit(meta))
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
    from tomojax.geometry.api import is_cone_beam

    with ExitStack() as cleanup:
        files = TerminalFiles(Path(path), cleanup)
        scan = item if isinstance(item, tj.Scan) else item.scan
        unit = length_unit(
            scan.source.geometry_metadata if scan is not None and scan.source is not None else {}
        )
        arrays: dict[str, Array] = {}
        info: Mapping[str, object] = {}
        if isinstance(item, tj.Reconstruction):
            method, info = item.method, item.info
            if method == "fbp" and is_cone_beam(item.geometry):
                method = "fdk"
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
                columns = tuple(
                    Column(name, "degree" if i < 3 else unit)
                    for i, name in enumerate(get_args(PoseColumn))
                )
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
