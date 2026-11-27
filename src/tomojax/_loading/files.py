"""What a file holds and a volume for other software: :func:`inspect` and :func:`export`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np

from tomojax import progress
from tomojax._scan.model import Alignment, Reconstruction, geometry_record
from tomojax._scan.simulation import Simulation
from tomojax._typed_arrays import object_mapping
from tomojax.core.registration import recorded_length_unit, volume_axes
from tomojax.io.api import (
    ExportCrop,
    ExportDtype,
    ExportSource,
    Inspection,
    inspect_export,
    metadata_record,
    normalize_json,
    plan_export,
    volume_entry,
)
from tomojax.wire.arrays import Array, Hdf5

if TYPE_CHECKING:
    from tomojax.geometry import Grid


def inspect(path: str | PathLike[str]) -> Inspection:
    """Describe a dataset, acquisition, or checkpoint directory and run checks.

    For a dataset or acquisition, reads metadata and a bounded sample of the
    frames, never the whole stack: the projections' shape, type and statistics,
    the angles, geometry, detector, flats and darks, the corrections recorded,
    saved poses, memory estimates and a saved volume, and what loading the file
    as a scan reads (``loading``: its length unit, where its rotation center
    comes from, frames a stopped scan leaves out, or the argument it needs).
    ``valid`` is True when the checks performed for that format find no issues
    in ``issues`` (frame layout, angles, sample frames, and raw calibration for
    recognized acquisitions). A file that records no rotation axis
    (``loading["geometry"] == "not_recorded"``) is valid but needs ``geometry``
    declared before reconstruction or alignment.

    For a checkpoint directory (``checkpoint=`` of a stopped run),
    ``checkpoint`` reports the saved run, how it was requested, save
    timestamps, entries, size, and whether a live process holds it.

    Parameters
    ----------
    path : str or path-like
        HDF5/NXtomo dataset, saved reconstruction, recognized raw acquisition,
        or checkpoint directory.

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
    from tomojax.core.checkpoint import describe
    from tomojax.io import validate_dataset
    from tomojax.io.api import inspect_dataset

    with progress.task("inspect", Path(path).name, None, ""):
        report: Mapping[str, object]
        if (record := inspect_export(Path(path))) is not None:
            report = {
                **_no_dataset(path),
                "record": record,
                "contents": record["arrays"],
                "checkpoint": None,
            }
        elif Path(path).is_dir() and (saved := describe(path)) is not None:
            report = {**_no_dataset(path), "checkpoint": saved}
        else:
            issues = validate_dataset(Path(path))["issues"]
            dataset = inspect_dataset(Path(path))
            loading = _loading(Path(path)) if dataset["projection"]["found"] else None
            checked: dict[str, object] = {
                "loading": loading,
                "checkpoint": None,
                "valid": not issues,
                "issues": issues,
            }
            report = {**dataset, **checked}
        return cast("Inspection", {"record": None, "contents": {}, **report, "schema_version": 2})


def _no_dataset(path: str | PathLike[str]) -> dict[str, object]:
    """The dataset sections of a report on a path that holds none (a checkpoint)."""
    absent = {"found": False}
    return {
        "schema_version": 2,
        "input_path": str(path),
        "projection": {**absent, "path": None, "shape": None, "dtype": None, "n_views": None},
        "angles": {**absent, "path": None, "count": None, "units": None, "coverage": None},
        "geometry": {**absent, "type": None, "meta_found": False, "meta_keys": []},
        "detector_metadata": {**absent, "nu": None, "nv": None, "du": None, "dv": None},
        "flats_darks": {"image_key_found": False, "flat_count": 0, "dark_count": 0},
        "corrections": [],
        "alignment": {**absent, "params_found": False},
        "memory_estimates": {"feasible": False, "modes": {}, "notes": "not a dataset"},
        "volume": {**absent, "shape": None, "axes": None, "method": None},
        "loading": None,
        "valid": True,
        "issues": [],
    }


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


def volume_source(item: Reconstruction | Simulation | Alignment) -> ExportSource:
    """Pair one result's pixels with its geometry and acquisition record."""
    scan = item.scan
    geometry = item.geometry if isinstance(item, Reconstruction) else item.scan.geometry
    dataset = geometry_record(
        geometry, name=scan.name if scan else "sample", source=scan.source if scan else None
    )
    meta = dataset.geometry_metadata
    if scan is not None:
        meta["corrections"] = [c.to_dict() for c in scan.corrections]
        if scan.view_positions is not None:
            meta["view_positions"] = scan.view_positions.tolist()
    if isinstance(item, Reconstruction):
        name, pixels, grid = "volume", item.volume, item.grid
        meta["reconstruction_method"] = item.method
        meta["reconstruction_info"] = normalize_json(item.info)
        if item.length_unit:
            meta["length_unit"] = item.length_unit
    elif isinstance(item, Simulation):
        name, pixels, grid = "phantom", item.phantom, item.scan.grid
    else:
        if item.image is None or item.grid is None:
            raise ValueError("alignment holds no working image to export")
        name, pixels, grid = "image", item.image, item.grid
        meta["alignment_info"] = {**item.info, "working_grid": grid.to_dict()}
    record = metadata_record(dataset.to_nxtomo_metadata())
    unit = recorded_length_unit(meta) or "scan_length"
    domain = (
        "mask" if np.dtype(pixels.dtype).kind in "biu" else str(meta.get("quantity", "attenuation"))
    )
    from tomojax.core.host_volume import mapped_storage, release_planes

    mapped = mapped_storage(pixels)

    def read(region: tuple[slice, ...]) -> np.ndarray:
        data = np.asarray(pixels[region])
        if mapped is not None and isinstance(pixels, np.ndarray):
            # Read the live mapping, including private edits made after a rename.
            data = data.copy()
            release_planes(pixels, region[0].start, region[0].stop)
        return data

    return ExportSource(
        name,
        pixels.shape,
        np.dtype(pixels.dtype),
        volume_axes(grid, unit),
        "1" if domain == "mask" else f"1/{unit}",
        domain,
        read,
        record,
    )


@contextmanager
def saved_export_source(path: Path, member: str | None) -> Iterator[tuple[ExportSource, Array]]:
    """Read a legacy volume by windows from the same open file as its registration."""
    from tomojax.geometry import Grid

    with h5py.File(path, "r") as file:
        meta, data, order = volume_entry(file)
        geom = meta.geometry_meta or {}
        name = (
            "phantom"
            if geom.get("volume_role") == "phantom"
            else "image"
            if "alignment_info" in geom and "reconstruction_method" not in geom
            else "volume"
        )
        if member is not None and member != name:
            raise ValueError(f"{path} has no {member} member to export; available: {name}")
        shape = tuple(data.shape[order.index(a)] for a in "zyx")
        unit = recorded_length_unit(geom) or "scan_length"
        grid_dict = meta.grid.to_dict() if isinstance(meta.grid, Grid) else meta.grid
        grid = saved_volume_grid(grid_dict, geom) or Grid(
            shape[2], shape[1], shape[0], 1.0, 1.0, 1.0
        )
        axes = volume_axes(grid, unit)
        record = metadata_record(meta)
        domain = str(geom.get("quantity", "attenuation"))

        def read(region: tuple[slice, ...]) -> np.ndarray:
            selection = tuple(region["zyx".index(a)] for a in order)
            return np.asarray(data[selection]).transpose(*(order.index(a) for a in "zyx"))

        source = ExportSource(name, shape, data.dtype, axes, f"1/{unit}", domain, read, record)
        disk = Array(
            Hdf5(str(path), str(data.name)),
            data.shape,
            data.dtype.str,
            tuple(axes["zyx".index(a)] for a in order),
            source.unit,
            source.domain,
        )
        yield source, disk


def export(
    path: str | PathLike[str],
    item: Reconstruction | Alignment | Simulation,
    *,
    dtype: ExportDtype | None = None,
    range: tuple[float, float] | None = None,  # noqa: A002 the command's --range
    crop: ExportCrop | None = None,
    binning: int = 1,
    axis: str | None = None,
    compress: bool | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Write a registered volume for other software.

    Parameters
    ----------
    path : str or path-like
        Destination. The longest matching suffix selects the format. A directory
        with no recognized suffix holds numbered TIFF slices.
    item : Reconstruction, Alignment or Simulation
        Reconstruction volume, alignment working image or simulation phantom.
        An integer or boolean volume supplies mask labels.
    dtype : {"float32", "uint16", "uint8"} or None, optional
        Output type. Defaults to float32, or uint8 for a mask.
    range : pair of float or None, optional
        Physical values mapped to zero and the integer maximum. The default uses
        finite 0.1 and 99.9 percentiles of up to 16 evenly spaced planes.
    crop : sequence of intervals or None, optional
        Half-open (start, stop) pairs, slices, or None in stored (z, y, x) order.
    binning : int, optional
        Mean over complete spatial bins after cropping. Default is one.
    axis : str or None, optional
        Stack planes along this axis. The default preserves stored order.
    compress : bool or None, optional
        Use the format's lossless codec. TIFF uses LZW and defaults to compressed for masks.
    dry_run : bool, optional
        Return the resolved plan before writing any files.

    Returns
    -------
    dict
        Shape, registration, value mapping, destination files and resource estimates.
        The receipt also reports bytes written and clipped voxel counts.

    Raises
    ------
    ValueError
        If the selection, value mapping or destination is incompatible with the member.

    Examples
    --------
    >>> import tomojax as tj
    >>> recon = tj.load_reconstruction("scan.nxs")  # doctest: +SKIP
    >>> receipt = tj.export("detail.tif", recon, crop=(slice(10, 30), None, None))  # doctest: +SKIP
    """
    source = volume_source(item)
    plan = plan_export(
        Path(path),
        source,
        dtype=dtype,
        value_range=range,
        crop=crop,
        binning=binning,
        axis=axis,
        compress=compress,
    )
    return plan.details if dry_run else plan.publish(force=bool(plan.details["destination_exists"]))
