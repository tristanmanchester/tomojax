"""What a file holds and a volume for other software: :func:`inspect` and :func:`export`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from os import PathLike
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tomojax import progress
from tomojax._loading.stack_files import slice_name, tiff_stack_entry
from tomojax._scan.model import Alignment, Reconstruction
from tomojax._scan.simulation import Simulation
from tomojax._typed_arrays import object_mapping
from tomojax.io.api import Inspection

if TYPE_CHECKING:
    from tomojax.geometry import Grid


def inspect(path: str | PathLike[str]) -> Inspection:
    """Describe a dataset or acquisition and run structural and calibration checks.

    Reads metadata and a bounded sample of the frames, never the whole stack: the
    projections' shape, type and statistics, the angles, geometry, detector,
    flats and darks, the corrections recorded, saved poses, memory estimates and a
    saved volume, and what loading the file as a scan reads (``loading``: its
    length unit, where its rotation center comes from, frames a stopped scan
    leaves out, or the argument it needs). ``valid`` is True when the checks
    performed for that format find no issues in ``issues`` (frame layout, angles,
    sample frames, and raw calibration for recognized acquisitions). A file that
    records no rotation axis (``loading["geometry"] == "not_recorded"``) is valid
    but needs ``geometry`` declared before reconstruction or alignment.

    Parameters
    ----------
    path : str or path-like
        HDF5/NXtomo dataset, saved reconstruction, or recognized raw acquisition.

    Returns
    -------
    Inspection
        The report, a mapping; ``tomojax inspect --json`` prints the same.

    Raises
    ------
    ValueError
        If the projections or acquisition metadata have invalid dimensions.
    OSError
        If the input cannot be opened.

    Examples
    --------
    >>> import tomojax as tj
    >>> report = tj.inspect("scan.nxs")  # doctest: +SKIP
    >>> report["valid"], report["angles"]["coverage"]  # doctest: +SKIP
    """
    from tomojax.io import validate_dataset
    from tomojax.io.api import inspect_dataset

    with progress.task("inspect", Path(path).name, None, ""):
        issues = validate_dataset(Path(path))["issues"]
        report = inspect_dataset(Path(path))
        loading = _loading(Path(path)) if report["projection"]["found"] else None
        return cast(
            "Inspection", {**report, "loading": loading, "valid": not issues, "issues": issues}
        )


def _loading(path: Path) -> dict[str, object]:
    """What :func:`tomojax.load` reads from ``path`` alone, or why it needs more.

    Calibration (flats and darks) is the dataset check's to report, in ``issues``.
    """
    from tomojax._loading.loader import discover_frames, stored_frames
    from tomojax.io.projection_layout import saved_projection_layout

    try:
        frames, located = discover_frames(path, apply_poses=True, layout_only=True, angles=None)
        if frames is None:
            try:  # a saved dataset records what its loading found
                record = saved_projection_layout(path)[0]
            except KeyError:
                stored, _ = stored_frames(
                    path, located, require_raw=True, apply_poses=True, angles=None
                )
                record = stored.source
        else:
            record = frames.source
    except (ValueError, KeyError) as err:  # the reader names the argument to supply
        message = err.args[0] if isinstance(err, KeyError) and err.args else str(err)
        return {
            "message": str(message),
            "length_unit": None,
            "center": None,
            "geometry": None,
            "frame_records": None,
        }
    metadata = object_mapping(record.geometry_metadata if record is not None else None)
    acquisition = object_mapping(metadata.get("acquisition"))
    return {
        "message": None,
        "length_unit": metadata.get("length_unit"),
        "center": metadata.get("detector_center_status"),
        "geometry": acquisition.get("geometry_status"),
        "frame_records": acquisition.get("frame_records"),
    }


def saved_volume_grid(
    grid_meta: Mapping[str, Any] | None,
    geometry_meta: Mapping[str, Any] | None,
) -> Grid | None:
    """Resolve the grid belonging to a saved dataset's stored volume.

    Alignments store the scan grid in ``grid_meta`` and the working image's grid
    under ``geometry_meta["alignment_info"]["working_grid"]``. Reconstructions
    and simulations store the volume's grid in ``grid_meta``.
    """
    from tomojax.geometry import Grid

    meta = geometry_meta or {}
    if "reconstruction_method" not in meta:
        alignment_info = meta.get("alignment_info")
        if isinstance(alignment_info, Mapping):
            working = alignment_info.get("working_grid")
            if isinstance(working, Mapping):
                return Grid.from_dict(working)
    if grid_meta:
        return Grid.from_dict(grid_meta)
    return None


def export(
    path: str | PathLike[str],
    reconstruction: Reconstruction | Alignment | Simulation,
    *,
    dtype: Literal["float32", "uint16"] = "float32",
    range: tuple[float, float] | None = None,  # noqa: A002 the command's --range
    prefix: str = "slice",
) -> None:
    """Write a reconstruction, alignment image or simulation phantom for other software.

    Interchange export for external tools (use :func:`tomojax.save` and
    :func:`tomojax.load_reconstruction` to round-trip TomoJAX datasets).
    ``path`` ending in ``.raw`` writes one little-endian, z-major file; otherwise
    ``path`` is a directory of z-slice TIFFs (y rows, x columns), numbered from
    the bottom of the volume. A JSON sidecar gives the shape, voxel size,
    physical grid placement (``vol_origin_xyz`` and ``grid``) and value scaling.
    TIFF directory exports stage the complete slice set before replacing an
    existing stack so stale slices are never mixed in.

    Parameters
    ----------
    path : str or path-like
        Directory for TIFF slices, or a ``.raw`` file.
    reconstruction : Reconstruction, Alignment or Simulation
        The volume (reconstruction, alignment working image, or simulation
        phantom) and its grid.
    dtype : {"float32", "uint16"}, optional
        Sample type; ``uint16`` maps ``range`` to 0..65535. Default is float32.
    range : tuple of float or None, optional
        The values ``uint16`` maps to 0 and 65535 (finite, ``low < high``);
        None takes the 0.1 and 99.9 percentiles of up to 16 slices. Only valid
        with ``dtype="uint16"``.
    prefix : str, optional
        TIFF file name prefix. Default is "slice".

    Raises
    ------
    ValueError
        If ``range`` is passed with ``float32`` or has nonfinite or non-increasing bounds,
        or an alignment holds no working image.

    Examples
    --------
    >>> import tomojax as tj
    >>> tj.export("slices", tj.load_reconstruction("recon.nxs"))  # doctest: +SKIP
    """
    scan = reconstruction.scan
    source = scan.source.source_path if scan is not None and scan.source else None
    export_volume(
        path, reconstruction, dtype=dtype, value_range=range, prefix=prefix, source=source
    )


def export_volume(
    path: str | PathLike[str],
    reconstruction: Reconstruction | Alignment | Simulation,
    *,
    dtype: Literal["float32", "uint16"],
    value_range: tuple[float, float] | None,
    prefix: str,
    source: str | dict[str, object] | None,
) -> None:
    """Export registered pixels with the durable source known to the caller."""
    from tomojax._scan.model import Alignment
    from tomojax._scan.simulation import Simulation

    if isinstance(reconstruction, Simulation):
        volume = reconstruction.phantom
        grid = reconstruction.scan.grid
    elif isinstance(reconstruction, Alignment):
        if reconstruction.image is None or reconstruction.grid is None:
            raise ValueError("alignment holds no working image to export")
        volume = reconstruction.image
        grid = reconstruction.grid
    else:
        volume = reconstruction.volume
        grid = reconstruction.grid
    write_volume(
        Path(path),
        lambda z: np.asarray(volume[z], np.float32),
        (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])),
        (grid.vx, grid.vy, grid.vz),
        dtype=dtype,
        value_range=value_range,
        prefix=prefix,
        grid=grid,
        source=source,
    )


def validate_export_options(
    path: Path, dtype: str, value_range: Sequence[float] | None, prefix: str
) -> tuple[float, float] | None:
    """Check file naming and scaling before opening any output."""
    from tomojax.errors import InvalidParameter

    if path.suffix.lower() != ".raw" and (
        not prefix or Path(prefix).name != prefix or prefix in (".", "..")
    ):
        raise InvalidParameter("prefix", "prefix must be a file name within the TIFF directory")
    try:
        return validate_export_range(dtype, value_range)
    except ValueError as exc:
        raise InvalidParameter("range", str(exc)) from exc


def validate_export_range(
    dtype: str, value_range: Sequence[float] | None
) -> tuple[float, float] | None:
    """Validate optional ``(low, high)`` scaling bounds for volume export."""
    if value_range is None:
        return None
    if dtype != "uint16":
        raise ValueError("range applies only to dtype='uint16'")
    if len(value_range) != 2:
        raise ValueError("range must contain two values: (low, high)")
    low, high = float(value_range[0]), float(value_range[1])
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        raise ValueError(f"range must be finite with low < high, got ({low}, {high})")
    return low, high


def _write_tiff_directory(
    out: Path,
    plane: Callable[[int], np.ndarray],
    convert: Callable[[np.ndarray], np.ndarray],
    nz: int,
    prefix: str,
    sidecar_text: str,
) -> Path:
    """Write TIFF z-slices and sidecar in a sibling staging directory before replacing ``out``."""
    import imageio.v3 as iio

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.partial-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for z in range(nz):
            progress.check_canceled()
            iio.imwrite(staging / slice_name(prefix, z), convert(plane(z)), extension=".tif")
            progress.step("slices")
        _ = (staging / f"{prefix}.json").write_text(sidecar_text)
        if not out.exists():
            staging.replace(out)
        else:
            for stale in out.iterdir():
                if tiff_stack_entry(stale, staging):
                    stale.unlink()
            for item in staging.iterdir():
                item.replace(out / item.name)
            staging.rmdir()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out / f"{prefix}.json"


def write_volume(
    out: Path,
    plane: Callable[[int], np.ndarray],
    shape: tuple[int, int, int],
    voxel: tuple[float, float, float],
    *,
    dtype: str,
    value_range: Sequence[float] | None,
    prefix: str,
    source: str | dict[str, object] | None = None,
    grid: Grid | None = None,
) -> Path:
    """Write ``shape`` (z, y, x) z-planes, each ``plane(z)``, as TIFFs or one raw file.

    Returns the sidecar's path.
    """
    from tomojax.core.files import replaced_when_complete
    from tomojax.geometry import grid_volume_origin

    with progress.task("export", out.name, shape[0], "slices"):
        validated_range = validate_export_options(out, dtype, value_range, prefix)
        nz = shape[0]
        raw = out.suffix.lower() == ".raw"
        low, high = (0.0, 1.0)
        if dtype == "uint16":
            low, high = _value_range(plane, nz) if validated_range is None else validated_range

        def convert(values: np.ndarray) -> np.ndarray:
            if dtype == "float32":
                return values
            scaled = (values - low) / (high - low) * 65535.0
            return np.clip(np.round(scaled), 0, 65535).astype(np.uint16)

        info: dict[str, object] = {
            "source": source,
            "format": "raw" if raw else "tiff",
            "dtype": dtype,
            "shape_zyx": list(shape),
            "voxel_size_xyz": list(voxel),
            "byte_order": "little" if raw else None,
            "value_range": [low, high] if dtype == "uint16" else None,
            "slice_axis": "z, from the bottom of the volume",
        }
        if grid is not None:
            info["vol_origin_xyz"] = list(grid_volume_origin(grid))
            info["grid"] = grid.to_dict()
        sidecar_text = json.dumps(info, indent=2) + "\n"

        if not raw:
            return _write_tiff_directory(out, plane, convert, nz, prefix, sidecar_text)
        parts = f"volume {np.prod(shape) * np.dtype(dtype).itemsize / 2**30:.1f} GiB"
        little = np.dtype(dtype).newbyteorder("<")
        with replaced_when_complete(out, parts) as temporary, temporary.open("wb") as stream:
            for z in range(nz):
                progress.check_canceled()
                _ = stream.write(convert(plane(z)).astype(little).tobytes())
                progress.step("slices")
        sidecar = out.with_suffix(".json")
        with replaced_when_complete(sidecar, "sidecar") as temporary:
            _ = temporary.write_text(sidecar_text)
        return sidecar


def _value_range(plane: Callable[[int], np.ndarray], nz: int) -> tuple[float, float]:
    """The 0.1 and 99.9 percentiles of up to 16 evenly spaced slices."""
    count = min(nz, 16)
    picks = sorted({round(i * (nz - 1) / max(1, count - 1)) for i in range(count)})
    values = np.concatenate([plane(z).ravel() for z in picks])
    low, high = (float(v) for v in np.percentile(values[np.isfinite(values)], [0.1, 99.9]))
    return low, high if high > low else low + 1.0


__all__ = ["export", "inspect", "saved_volume_grid", "validate_export_range", "write_volume"]
