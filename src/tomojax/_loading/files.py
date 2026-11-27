"""What a file holds and a volume for other software: :func:`inspect` and :func:`export`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import partial
import hashlib
from os import PathLike
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from tomojax import progress
from tomojax._data import FrameStack, Hdf5Frames, PinnedFrames
from tomojax._loading.registration import registered_member, unit_of
from tomojax._scan.model import Alignment, Reconstruction, Scan, projection_record
from tomojax._scan.simulation import Simulation
from tomojax._typed_arrays import object_mapping
from tomojax.core.input_files import input_hdf5, input_path, open_input_hdf5, reading
from tomojax.core.registration import recorded_length_unit, volume_axes, volume_values
from tomojax.geometry import Grid
from tomojax.io.api import (
    ExportCrop,
    ExportDtype,
    ExportSource,
    Inspection,
    inspect_export,
    metadata_record,
    plan_export,
    registered_members,
    saved_volume_grid,
    volume_entry,
)
from tomojax.io.projection_layout import MatlabFrames, ProjectionPixels, ProjectionStorage
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Member


def inspect(path: str | PathLike[str]) -> Inspection:
    """Describe a dataset, acquisition, or checkpoint directory and run checks.

    For a dataset or acquisition, reads metadata and a bounded sample of the
    frames, never the whole stack: the projections' shape, type and statistics,
    the angles, geometry, detector, flats and darks, the corrections recorded,
    saved poses, memory estimates and a saved volume, and what loading the file
    as a scan reads (``loading``: its length unit, where its rotation center
    comes from, full-turn parallel orbit coverage such as ``full_turn``,
    ``axis_to_edge_px``, ``field_of_view_px``, ``overlap_px``, and
    ``central_gap_px``, frames a stopped scan leaves out, or the argument it
    needs).
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
    from tomojax.io.inspection_memory import memory_estimates

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
            loading, grid = _loading(Path(path)) if dataset["projection"]["found"] else (None, None)
            if grid is not None:
                dataset["memory_estimates"] = memory_estimates(
                    [grid.nx, grid.ny, grid.nz], dataset["projection"]
                )
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


def _loading(path: Path) -> tuple[dict[str, object], Grid | None]:
    """What :func:`tomojax.load` reads from ``path`` alone, or why it needs more.

    Calibration (flats and darks) is the dataset check's to report, in ``issues``.
    """
    from tomojax._loading.loader import loading_geometry
    from tomojax.geometry.coverage import full_parallel_turn, horizontal_detector, panel_field

    try:
        record, geometry = loading_geometry(path)
    except (ValueError, KeyError) as err:  # the reader names the argument to supply
        message = err.args[0] if isinstance(err, KeyError) and err.args else str(err)
        return {
            "message": str(message),
            "length_unit": None,
            "center": None,
            "geometry": None,
            "frame_records": None,
        }, None
    metadata = object_mapping(record.geometry_metadata if record is not None else None)
    acquisition = object_mapping(metadata.get("acquisition"))
    coverage = {}
    if full_parallel_turn(geometry):
        coverage["full_turn"] = True
        roll = float(getattr(geometry, "detector_roll", 0.0))
        if roll % 90 == 0 and metadata.get("detector_center_status") != "not_recorded":
            panel = horizontal_detector(geometry.detector, roll)
            near, far = panel_field(panel.nu, panel.center[0] / panel.du)
            coverage.update(
                axis_to_edge_px=near, field_of_view_px=2 * far, overlap_px=max(0.0, 2 * near)
            )
            if near < 0:
                coverage["central_gap_px"] = -2 * near
    return {
        **coverage,
        "message": None,
        "length_unit": metadata.get("length_unit"),
        "center": metadata.get("detector_center_status"),
        "geometry": acquisition.get("geometry_status"),
        "frame_records": acquisition.get("frame_records"),
    }, geometry.grid


def export_source(
    item: Reconstruction | Simulation | Alignment | Scan, *, samples: ProjectionPixels | None = None
) -> ExportSource:
    """Pair a member's window reader with its geometry and acquisition record."""
    from tomojax._scan.records import result_dataset
    from tomojax.io.api import replace_dataset

    unit = unit_of(item if isinstance(item, Reconstruction | Scan) else item.scan)
    view = Axis("view", "acquisition", "degree", Coordinates(Member("source", "angles")))
    member = registered_member(item, unit, view=view)
    if member is None:
        raise ValueError("alignment holds no working image to export")
    pixels = member.pixels if samples is None else samples
    if isinstance(item, Scan):
        record = partial(projection_record, item)
    else:
        dataset = result_dataset(item)
        record = metadata_record(replace_dataset(dataset, volume=None).to_nxtomo_metadata())
    from tomojax.core.host_volume import mapped_storage, release_planes
    from tomojax.io._export_resume import hdf5_identity, mapped_identity

    mapped = mapped_storage(pixels)
    if mapped is not None and mapped.filename is not None:
        _ = input_path(Path(mapped.filename))
    elif isinstance(pixels, PinnedFrames):
        input_hdf5(pixels.signal.file)
    elif isinstance(pixels, Hdf5Frames | MatlabFrames):
        storage = pixels.stored if isinstance(pixels, MatlabFrames) else pixels
        _ = input_path(Path(storage.path), container=True)

    def read(region: tuple[slice, ...]) -> np.ndarray:
        if isinstance(pixels, FrameStack | MatlabFrames):
            views, rows, cols = region
            return pixels.frames_at(views).window(rows, cols)[:]
        data = np.asarray(pixels[region])
        if mapped is not None and isinstance(pixels, np.ndarray):
            # Read the live mapping, including private edits made after a rename.
            data = data.copy()
            release_planes(pixels, region[0].start, region[0].stop)
        return data

    def identity() -> tuple[int, ...] | None:
        if isinstance(pixels, PinnedFrames | Hdf5Frames | MatlabFrames):
            stack = pixels.stored if isinstance(pixels, MatlabFrames) else pixels
            if isinstance(stack, PinnedFrames):
                generation = hdf5_identity(stack.signal)
            else:
                with open_input_hdf5(Path(stack.path)) as file:
                    field = file[stack.dataset]
                    generation = hdf5_identity(field) if isinstance(field, h5py.Dataset) else None
            if generation is None:
                return None
            views = hashlib.blake2b(stack.frames.tobytes(), digest_size=8).digest()
            return (*generation, stack.first_row, stack.first_col, int.from_bytes(views))
        return mapped_identity(pixels)

    return ExportSource(
        member.name,
        pixels.shape,
        np.dtype(pixels.dtype),
        member.axes,
        member.unit,
        member.domain,
        read,
        record,
        identity,
    )


@contextmanager
def saved_export_source(path: Path, member: str | None) -> Iterator[tuple[ExportSource, Array]]:
    """Read registered volume windows from the same open generation as the metadata."""
    from tomojax._data import read_format_2
    from tomojax._loading.loader import stored_scan

    with reading(path), open_input_hdf5(path) as file:
        members = registered_members(file, path) if "tomojax_format" in file.attrs else {}
        if member == "projections" or (member is None and members.keys() == {"projections"}):
            if "projections" not in members:
                raise ValueError(
                    f"{path} has no format-2 projections member. For an rc6 scan, "
                    "save a new file with tj.save(new_path, tj.load(old_path))."
                )
            payload = read_format_2(file, include_volume=False, lazy_projections=True)
            selected = ProjectionStorage.from_payload(path, payload)
            if selected.domain == "counts":
                raise ValueError("export a corrected scan to write line integrals")
            kept = selected.key == 0
            pixels = selected.stack if kept.all() else selected.frames_at(kept)
            scan = stored_scan(selected, pixels, angles=None, apply_poses=True)
            yield export_source(scan, samples=pixels), members["projections"]
            return
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
        if members:
            disk = members[name]
            axes = disk.axes
        else:
            unit = recorded_length_unit(geom) or "scan_length"
            grid = saved_volume_grid(meta.grid, geom) or Grid(
                shape[2], shape[1], shape[0], 1.0, 1.0, 1.0
            )
            axes = volume_axes(grid, unit)
            value_unit, domain = volume_values(data.dtype, unit, geom)
            disk = Array(
                Hdf5(str(path), "/entry/processing/tomojax/volume"),
                data.shape,
                data.dtype.str,
                tuple(axes["zyx".index(a)] for a in order),
                value_unit,
                domain,
            )

        from tomojax.core.host_volume import mapped_storage, release_planes

        pixels = meta.volume
        direct = isinstance(pixels, np.ndarray) and mapped_storage(pixels) is not None

        def read(region: tuple[slice, ...]) -> np.ndarray:
            if direct and isinstance(pixels, np.ndarray):
                values = pixels[region].copy()
                release_planes(pixels, region[0].start, region[0].stop)
                return values
            selection = tuple(region["zyx".index(a)] for a in order)
            return np.asarray(data[selection]).transpose(*(order.index(a) for a in "zyx"))

        from tomojax.io._export_resume import hdf5_identity

        source = ExportSource(
            name,
            shape,
            data.dtype,
            axes,
            disk.unit,
            disk.domain,
            read,
            metadata_record(meta),
            lambda: hdf5_identity(data),
        )
        yield source, disk


def export(
    path: str | PathLike[str],
    item: Reconstruction | Alignment | Simulation | Scan,
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
        with no recognized suffix holds numbered TIFF slices. For HDF5, append
        ``::/entry/group/signal`` to choose the dataset.
        Files and directories are checked against live inputs, including physical aliases.
    item : Reconstruction, Alignment, Simulation or Scan
        Reconstruction volume, alignment working image, simulation phantom or scan projections.
        An integer or boolean volume supplies mask labels.
    dtype : {"float32", "uint16", "uint8"} or None, optional
        Output type. Defaults to float32, or uint8 for a mask.
    range : pair of float or None, optional
        Physical values mapped to zero and the integer maximum. The default uses
        finite 0.1 and 99.9 percentiles of up to 16 evenly spaced planes.
    crop : sequence of intervals or None, optional
        Half-open intervals in stored (z, y, x) or (view, v, u) order.
    binning : int, optional
        Mean over complete spatial bins after cropping. Default is one.
    axis : str or None, optional
        Stack planes along this axis. The default preserves stored order.
    compress : bool or None, optional
        Use the format's lossless codec. OME-Zarr volumes and masks default to compressed.
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
    if not isinstance(path, str | PathLike) or not isinstance(
        item, Reconstruction | Alignment | Simulation | Scan
    ):
        raise TypeError(
            "tj.export(path, item) expects a destination path and a Scan, Reconstruction, "
            "Alignment or Simulation with registered geometry"
        )
    with reading():
        source = export_source(item)
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
        return (
            plan.details
            if dry_run
            else plan.publish(force=bool(plan.details["destination_exists"]))
        )
