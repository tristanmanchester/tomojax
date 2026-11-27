from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import h5py
import msgspec
import numpy as np

from tomojax import progress
from tomojax.core.files import replaced_when_complete
from tomojax.core.host_volume import mapped_storage
from tomojax.core.input_files import open_input_hdf5, reading
from tomojax.wire.arrays import Array, Coordinates

from ._format2 import (
    contiguous,
    copy_pixels,
    finish_record,
    local_map,
    nxdata,
    open_record,
    read_format_2,
    signal_at,
    volume_member,
    write_record,
)
from ._io_acquisition import ROTATION_ANGLE_PATHS, angle_scale_degrees
from ._io_frames import NXTOMO_FRAME_PATHS, Hdf5Frames, LocatedFrames, locate_frames
from ._io_helpers import (
    attr_to_str,
    axes_log_warning,
    child_dataset,
    child_group,
    ensure_group,
    load_detector_metadata,
    load_sample_metadata,
    load_source_metadata,
    native_file,
    sizes_of,
)
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata, ValidationReport
from ._record import array_record, feature_record
from .geometry_meta import normalize_geometry_type

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from os import PathLike


def _reading_pixels(*arrays: object) -> AbstractContextManager[None]:
    return reading(
        *(
            Path(mapping.filename)
            for pixels in arrays
            if (mapping := mapped_storage(pixels)) is not None and mapping.filename is not None
        )
    )


def save_nxtomo(
    path: str,
    projections: np.ndarray | None,
    *,
    metadata: NXTomoMetadata | None = None,
    overwrite: bool = True,
) -> None:
    """Write a format-2 NeXus dataset with unfiltered contiguous pixels.

    `metadata` owns the persistence contract: geometry, detector/grid definitions,
    optional reconstructed volume, and TomoJAX-specific extras. Volume inputs in
    `metadata.volume` use `(z, y, x)` order, on disk and in memory.

    Parameters
    ----------
    path : str
        NXtomo output file; parent directories are created.
    projections : numpy.ndarray or None
        Projection stack in (view, v, u) order; None for a reconstruction saved without
        its scan, which then needs ``metadata.volume`` and keeps the geometry only.
    metadata : NXTomoMetadata or None, optional
        Acquisition and reconstruction metadata; absent angles stay absent.
        None writes projections without detector or angular calibration.
    overwrite : bool, optional
        Replace an existing file; False requires a new path.

    Returns
    -------
    None
        Write projections and metadata to the NXtomo container.

    Raises
    ------
    ValueError
        If projection shape or volume-axis metadata are invalid.
    OSError
        If output cannot be written or already exists with overwrite=False.
    """
    meta = metadata or NXTomoMetadata()

    proj = projections
    if proj is None and meta.volume is None:
        raise ValueError("a dataset without projections must hold a volume")
    if proj is not None and proj.ndim != 3:
        raise ValueError("projections must be (n_views, nv, nu)")
    if meta.volume is not None and meta.volume.ndim != 3:
        raise ValueError("volume must be 3D (nz, ny, nx)")
    with (
        _reading_pixels(proj, meta.volume),
        replaced_when_complete(path, sizes_of(proj, meta.volume), overwrite=overwrite) as temporary,
        h5py.File(temporary, "w") as f,
    ):
        volume = None
        if meta.volume is not None:
            entry = ensure_group(f, "entry", "NXentry")
            group = ensure_group(entry, str(volume_member(meta.geometry_meta or {})), "NXdata")
            volume = contiguous(
                group, "data", tuple(meta.volume.shape), np.dtype(meta.volume.dtype)
            )
            copy_pixels(volume, meta.volume)
        write_record(f, Path(path), meta, proj, volume)


def save_mask(
    path: Path,
    registration: Array,
    *,
    name: str,
    recipe: dict,
    box: tuple[slice, ...],
    inside: np.ndarray,
    coordinates: dict[str, np.ndarray],
    history: list,
) -> None:
    """Write a boxed mask and its registered coordinates to a native container.

    Parameters
    ----------
    path : Path
        Native output path.
    registration : Array
        Full mask shape and coordinate axes.
    name : str
        Feature name, preserved verbatim in the record.
    recipe : dict
        Resolved operation that produced the mask.
    box : tuple of slice
        Window containing every selected voxel.
    inside : numpy.ndarray
        Mask values within the window.
    coordinates : dict
        Coordinate vectors keyed by nonuniform axis name.
    history : list
        Provenance records from earlier saves.
    """
    encoded = quote(name, safe="")
    if encoded in (".", ".."):
        encoded = encoded.replace(".", "%2E")
    record = feature_record(registration, name, recipe, history)
    with (
        _reading_pixels(inside, *coordinates.values()),
        replaced_when_complete(path, "feature mask") as temporary,
        h5py.File(temporary, "w") as file,
    ):
        entry = ensure_group(file, "entry", "NXentry")
        masks = ensure_group(entry, "masks", "NXcollection")
        group = ensure_group(masks, encoded, "NXdata")
        mask = group.create_dataset(
            "data",
            shape=registration.shape,
            dtype="u1",
            chunks=True,
            compression="gzip",
            fillvalue=0,
        )
        if inside.size:
            mask[box] = inside
        desc = array_record(registration.shape, "uint8", registration.axes, "1", "mask")
        vectors = {}
        for axis, description in zip(registration.axes, desc["axes"], strict=True):
            if isinstance(axis.coordinates, Coordinates):
                vector = group.create_dataset(axis.name, data=coordinates[axis.name])
                vector.attrs["units"] = axis.unit
                vectors[axis.name] = vector
                table = f"mask_{axis.name}"
                record["tables"][table] = {"path": vector.name, "unit": axis.unit}
                description["coordinates"] = {"table": table}
        nxdata(group, mask, registration.axes, unit="1", domain="mask", coordinates=vectors)
        record.update(
            format=2,
            kind="feature",
            arrays={
                "mask": {
                    **desc,
                    "path": mask.name,
                    "fixed": msgspec.to_builtins(registration.fixed),
                }
            },
            feature={**record["feature"], "box": [[b.start, b.stop] for b in box]},
        )
        masks.attrs["default"] = encoded
        finish_record(file, entry, path, record, default="masks")


@contextmanager
def slab_volume(
    path: Path, meta: NXTomoMetadata, shape: tuple[int, int, int]
) -> Iterator[h5py.Dataset]:
    """Open a NeXus reconstruction for single-pass slab writes.

    Parameters
    ----------
    path : Path
        Destination for the completed volume.
    meta : NXTomoMetadata
        Acquisition geometry and volume grid, including physical origin and units.
    shape : tuple of int
        Full volume dimensions in (z, y, x) order.

    Yields
    ------
    h5py.Dataset
        Registered float32 attenuation dataset, writable until context exit.
    """
    with (
        replaced_when_complete(path, f"volume {math.prod(shape) * 4 / 2**30:.1f} GiB") as temporary,
        h5py.File(temporary, "w") as f,
    ):
        entry = ensure_group(f, "entry", "NXentry")
        group = ensure_group(entry, "volume", "NXdata")
        volume = contiguous(group, "data", shape, np.dtype("<f4"))
        yield volume
        write_record(f, path, meta, None, volume)


_TILE_BYTES = 64 << 20


def read_projections(stack: Hdf5Frames) -> np.ndarray:
    views, rows, cols = stack.shape
    with progress.task("load", Path(stack.path).name, views, "views"):
        active = progress.for_unit("views")
        if active is None:
            return np.asarray(stack)
        step = max(1, _TILE_BYTES // max(1, rows * cols * stack.dtype.itemsize))
        if step >= views:
            result = np.asarray(stack)
            active.advance(views, {})
            progress.check_canceled()
            return result
        result = np.empty(stack.shape, stack.dtype)
        for start in range(0, views, step):
            stop = min(views, start + step)
            result[start:stop] = stack[start:stop]
            active.advance(stop, {})
            progress.check_canceled()
        return result


def load_nxtomo(path: str, *, lazy: bool = False) -> LoadedNXTomo:
    """Load an NXtomo dataset and TomoJAX extras.

    Returns a ``LoadedNXTomo`` payload with typed accessors for projections and
    persisted metadata. ``grid`` is present only when the file stores grid
    metadata; a saved volume does not establish physical voxels. Volumes are returned in
    ``(z, y, x)`` order. With ``lazy`` the projections are an
    :class:`~tomojax._data._io_frames.Hdf5Frames`, read only as indexed. When
    ``image_key`` or angles have a different length from the frame stack (for
    example when a scan stopped early), the common prefix from frame 0 is read
    with a warning and the original counts are recorded in
    ``metadata.geometry_meta["acquisition"]["frame_records"]``.

    Parameters
    ----------
    path : str
        NXtomo/HDF5 input file.
    lazy : bool, optional
        Keep projections lazy rather than materializing them; defaults to False.

    Returns
    -------
    LoadedNXTomo
        Projection payload and metadata, with any saved volume in (z, y, x) order.

    Raises
    ------
    KeyError
        If required NXtomo groups or projection data are absent.
    ValueError
        If per-frame metadata or volume-axis conventions are invalid.
    OSError
        If the input cannot be opened.
    """
    return read_nxtomo(path, lazy=lazy, layout_only=False)


def read_nxtomo(
    path: str, *, lazy: bool, layout_only: bool, located: LocatedFrames | None = None
) -> LoadedNXTomo:
    with open_input_hdf5(Path(path)) as f:
        if native_file(f):
            return read_format_2(f, layout_only=layout_only, lazy_projections=lazy)
        entry = child_group(f, "/entry")
        if entry is None:
            raise KeyError("Missing /entry")
        if located is None:
            where = next((p for p in NXTOMO_FRAME_PATHS if p in f), None)
            if where is None:
                raise KeyError("Could not find projections dataset under /entry")
            located = locate_frames(path, data_path=where)
        stack = located.stack
        key = located.image_key
        if key is None:
            axes_log_warning("load_nxtomo: missing image_key for %s; defaulting to zeros", path)
            key = np.zeros(len(stack), np.int32)
        out: LoadedDataset = {
            "projections": stack if lazy else read_projections(stack),
            "image_key": key,
            "angles": None if located.angles is None else located.angles.astype(np.float32),
        }
        geometry = child_group(entry, "geometry")
        if geometry is not None and "type" in geometry.attrs:
            out["geometry_type"] = normalize_geometry_type(attr_to_str(geometry.attrs["type"]))
        load_detector_metadata(out, entry)
        load_source_metadata(out, entry)
        load_sample_metadata(out, entry)
        payload = LoadedNXTomo.from_dataset(out)
        if located.acquisition:
            meta = dict(payload.metadata.geometry_meta or {})
            meta["acquisition"] = located.acquisition
            payload.metadata.geometry_meta = meta
        return payload


def native_payload(
    path: Path,
    *,
    include_volume: bool = True,
    layout_only: bool = False,
    lazy_projections: bool = False,
    **paths: str | None,
) -> LoadedNXTomo | None:
    """Read a marked TomoJAX generation before acquisition discovery."""
    if path.suffix.lower() not in {".nxs", ".h5", ".hdf5", ".hdf", ".nx"} or not path.is_file():
        return None
    with open_input_hdf5(path) as file:
        if native_file(file):
            return read_format_2(
                file,
                include_volume=include_volume,
                layout_only=layout_only,
                lazy_projections=lazy_projections,
                **paths,
            )
    return None


def volume_entry(file: h5py.File) -> tuple[NXTomoMetadata, h5py.Dataset]:
    """Locate a saved volume and read its registration from the open file.

    Parameters
    ----------
    file : h5py.File
        Open TomoJAX HDF5 container.

    Returns
    -------
    tuple
        Metadata and the live volume dataset in (z, y, x) order.
    """
    payload = read_format_2(file, layout_only=True)
    record = payload.file_record
    assert record is not None
    member = volume_member(record["geometry"]["meta"])
    if member not in record["arrays"]:
        raise ValueError(f"{file.filename} holds no reconstructed volume")
    data = signal_at(file, record["arrays"][member]["path"])
    # The caller reads windows of this dataset for nonlocal or filtered storage.
    payload.metadata.volume = local_map(file, data)
    return payload.metadata, data


def saved_without_frames(path: str | PathLike[str]) -> bool:
    """Whether ``path`` is a reconstruction saved without its scan: a volume and no frames.

    Parameters
    ----------
    path : str or path-like
        HDF5 file to look in.

    Returns
    -------
    bool
        True when the file holds a saved volume and no stack of frames.
    """
    with h5py.File(os.fspath(path), "r") as file:
        if not native_file(file):
            return False
        arrays = open_record(file)["arrays"]
        return "projections" not in arrays and bool(arrays.keys() & {"volume", "image", "phantom"})


def validate_nxtomo(path: str) -> ValidationReport:
    """Check the lightweight NXtomo schema and report any issues.

    Parameters
    ----------
    path : str
        NXtomo file to check.

    Returns
    -------
    dict
        Schema report; an empty issues list means no problems were found.
    """
    report: ValidationReport = {"issues": []}
    try:
        with open_input_hdf5(Path(path)) as f:
            e = child_group(f, "/entry")
            if e is None:
                report["issues"].append("Missing /entry")
                return report
            issues, angle_path, n_views = _required_entry_issues(f, e)
            report["issues"].extend(issues)
            key = child_dataset(e, "instrument/detector/image_key")
            if key is not None:
                report["issues"].extend(_image_key_issues(key, frames=n_views))
            angles = None if angle_path is None else child_dataset(f, angle_path)
            if angles is not None:
                report["issues"].extend(_rotation_angle_issues(angles, frames=n_views, path=path))
    except OSError as exc:
        report["issues"].append(f"Unable to read HDF5 file: {exc}")
    except ValueError as exc:
        report["issues"].append(str(exc))
    return report


def _required_entry_issues(f: h5py.File, e: h5py.Group) -> tuple[list[str], str | None, int | None]:
    """Missing or misshapen required NXtomo members, the angle path and the view count."""
    issues = []
    field = e.get("definition")
    stored = np.asarray(field).tolist() if isinstance(field, h5py.Dataset) else None
    stored = stored.decode() if isinstance(stored, bytes) else stored
    if "NXtomo" not in (stored, attr_to_str(e.attrs.get("definition"))):
        issues.append("entry/definition must be 'NXtomo'")
    data = e.get("instrument/detector/data")
    if data is None:
        issues.append("Missing instrument/detector/data")
    angle_path = next((p for p in ROTATION_ANGLE_PATHS if p in f), None)
    if angle_path is None:
        issues.append("Missing sample rotation_angle")
    if "instrument/detector/image_key" not in e:
        issues.append("Missing instrument/detector/image_key")
    if "sample" not in e or "name" not in e["sample"]:
        issues.append("Missing sample/name")
    n_views = None
    if data is not None:
        if not isinstance(data, h5py.Dataset) or data.ndim not in (2, 3):
            issues.append("instrument/detector/data must be (n_views, nu) or (n_views, nv, nu)")
        else:
            n_views = data.shape[0]
    return issues, angle_path, n_views


def _image_key_issues(image_key: h5py.Dataset, *, frames: int | None) -> list[str]:
    """Shape, dtype and value issues in an image_key dataset."""
    issues = []
    if image_key.ndim != 1:
        issues.append("instrument/detector/image_key must be 1D (n_views,)")
    elif frames is not None and image_key.shape[0] != frames:
        issues.append("image_key length must match #views in detector/data")
    if image_key.dtype.kind not in {"i", "u"}:
        issues.append("image_key must use integer dtype")
    elif image_key.ndim == 1:
        values = np.asarray(image_key[...])
        allowed = np.asarray([0, 1, 2, 3], dtype=values.dtype)
        invalid_values, invalid_counts = np.unique(
            values[~np.isin(values, allowed)], return_counts=True
        )
        if invalid_values.size > 0:
            formatted = ", ".join(
                f"{int(value)} ({int(count)} frame{'s' if int(count) != 1 else ''})"
                for value, count in zip(invalid_values, invalid_counts, strict=True)
            )
            issues.append("image_key values must be in {0, 1, 2, 3}; found " + formatted)
    return issues


def _rotation_angle_issues(angles: h5py.Dataset, *, frames: int | None, path: str) -> list[str]:
    """Shape and unit issues in a rotation-angle dataset."""
    issues = []
    if angles.ndim != 1:
        issues.append("rotation_angle must be 1D (n_views,)")
    elif frames is not None and angles.shape[0] != frames:
        issues.append("rotation_angle length must match #views in detector/data")
    units = attr_to_str(angles.attrs.get("units"))
    try:
        angle_scale_degrees(units or "", path=path)
    except ValueError as exc:
        issues.append(str(exc))
    return issues
