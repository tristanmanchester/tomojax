"""A stack of frames in an HDF5 file, read only when, and as far as, it is indexed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np

from tomojax.core.files import validate_virtual_pixels

from ._frame_stack import FrameStack
from ._hdf5_stored import read_stored
from ._io_acquisition import ROTATION_ANGLE_PATHS, angles_in_degrees, checked_image_key

if TYPE_CHECKING:
    from ._io_types import JsonObject


@dataclass(frozen=True)
class Hdf5Frames(FrameStack):
    """Frames ``frames`` of ``dataset`` in ``path``, read on indexing.

    It holds no open file: each read opens the file, reads the frames asked for
    and closes it, so it outlives the call that made it and is safe to read
    from several threads. Unfiltered data are read at their stored byte
    offsets, many reads at once, rather than through HDF5. Indexing by a
    slice or index array selects frames (``stack[a:b]`` reads frames a to b of
    this stack) and returns NumPy arrays in the file's dtype; :meth:`frames_at`
    and :meth:`window` narrow the stack without reading.
    A stored sinogram ``(view, u)`` has one detector row: reads expose the
    same ``(view, v, u)`` shape as a three-dimensional stack.
    """

    path: str
    dataset: str
    _efficient_row_windows: bool = False  # a window of rows reads only those rows
    _bulk_row_reads: ClassVar[bool] = True  # many views in one HDF5 request

    @classmethod
    def open(cls, path: str, dataset: str) -> Hdf5Frames:
        """Every frame of ``dataset`` in ``path``."""
        path = str(Path(path).absolute())  # later reads must not depend on the working directory
        with h5py.File(path, "r") as file:
            data = file[dataset]
            if not isinstance(data, h5py.Dataset) or data.ndim not in (2, 3):
                raise ValueError(f"{path}:{dataset} is not a 2-D sinogram or 3-D stack of frames")
            n, cols = int(data.shape[0]), int(data.shape[-1])
            rows = 1 if data.ndim == 2 else int(data.shape[1])
            validate_virtual_pixels(data)
            # Unfiltered data supports direct reads of partial chunks. __getitem__
            # bypasses the chunk cache so narrow windows do not fetch whole frames.
            windows = not data.is_virtual and data.id.get_create_plist().get_nfilters() == 0
            return cls(
                path,
                dataset,
                frames=np.arange(n),
                rows=rows,
                cols=cols,
                dtype=np.dtype(data.dtype),
                _efficient_row_windows=windows,
            )

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        if indices.size and np.any(np.diff(indices) <= 0):
            raise ValueError("Hdf5Frames reads frames in increasing order only")
        # Chunk caching can amplify a narrow read to the entire chunk even with
        # no compression. Filtered/virtual datasets retain their normal cache.
        cache = 0 if self._efficient_row_windows else None
        with h5py.File(self.path, "r", rdcc_nbytes=cache) as file:
            data = file[self.dataset]
            if not isinstance(data, h5py.Dataset):
                raise ValueError(f"{self.path}:{self.dataset} no longer holds a detector stack")
            if indices.size == 0:
                return np.empty((0, self.rows, self.cols), self.dtype)
            stored = read_stored(data, indices, rows, cols)
            if stored is not None:
                return stored
            detector = (cols,) if data.ndim == 2 else (rows, cols)
            first, last = int(indices[0]), int(indices[-1]) + 1
            if last - first == indices.size:  # contiguous: one hyperslab
                return np.asarray(data[(slice(first, last), *detector)]).reshape(
                    indices.size, self.rows, self.cols
                )
            # h5py fancy indexing is far slower than reading each run of frames.
            out = np.empty((indices.size, self.rows, self.cols), self.dtype)
            # The singleton row is only a view of the destination allocation;
            # read_direct still addresses the stored dataset's physical rank.
            destination_array = out[:, 0, :] if data.ndim == 2 else out
            starts = np.concatenate(([0], np.flatnonzero(np.diff(indices) > 1) + 1))
            stops = np.concatenate((starts[1:], [indices.size]))
            for start, stop in zip(starts, stops, strict=True):
                source = (slice(int(indices[start]), int(indices[stop - 1]) + 1), *detector)
                destination = (slice(int(start), int(stop)), *(slice(None),) * len(detector))
                data.read_direct(destination_array, source_sel=source, dest_sel=destination)
            return out


_FRAME_PATHS = (
    "/entry/instrument/detector/data",
    "/entry/data/projections",
    "/entry/projections",
    "/exchange/data",  # APS Data Exchange
)
_IMAGE_KEY_PATH = "/entry/instrument/detector/image_key"


def _gda_paths(file: h5py.File) -> tuple[str, str, str] | None:
    """A coherent GDA imaging acquisition, including stripped NeXus exports.

    The rotation command, not the encoder's raw counts, supplies the angles.
    Matching the stack, key and command together avoids unrelated NXdata
    signals and duplicate instrument links. Entry names are not prescribed.
    """
    found: list[tuple[str, str, str]] = []
    for name, entry in file.items():
        if not isinstance(entry, h5py.Group):
            continue
        prefix = f"/{name}"
        data = f"{prefix}/imaging/data"
        stack = _dataset(file, data)
        if stack is None or stack.ndim not in (2, 3):
            continue
        key = next(
            (
                f"{prefix}/{p}"
                for p in (
                    "tomo_entry/instrument/detector/image_key",
                    "instrument/EtherCAT/image_key",
                )
                if _dataset(file, f"{prefix}/{p}") is not None
            ),
            None,
        )
        angle = next(
            (
                f"{prefix}/{p}"
                for p in (
                    "imaging/smaract_zrot_value_set",
                    "imaging_sum/smaract_zrot_value_set",
                    "imaging_sum/smaract_zrot",
                )
                if _dataset(file, f"{prefix}/{p}") is not None
            ),
            None,
        )
        if key is not None and angle is not None:
            found.append((data, key, angle))
    if len(found) > 1:
        raise KeyError("Several GDA imaging acquisitions found: name their data/key/angle paths")
    return found[0] if found else None


def _gda_metadata(
    file: h5py.File,
    gda: tuple[str, str, str] | None,
    where: str,
    key_at: str | None,
    angles_at: str | None,
) -> tuple[JsonObject | None, str | None]:
    """Record discovery and the known GDA angular-command unit correction."""
    acquisition: JsonObject | None = None
    units_override = None
    if gda is not None and key_at is not None and angles_at is not None:
        acquisition = {
            "layout": "diamond_gda",
            "data_path": where,
            "image_key_path": key_at,
            "angles_path": angles_at,
        }
        units = file[angles_at].attrs.get("units", "")
        units = units.decode() if isinstance(units, bytes) else str(units)
        # GDA's SmarAct angular command can inherit the positioner's linear
        # unit label. Only this identified rotation command has that quirk.
        if angles_at == gda[2] and angles_at.endswith("/smaract_zrot_value_set") and units == "mm":
            units_override = "degree"
            acquisition.update(angle_units_recorded=units, angle_units="degree")
            logging.getLogger(__name__).warning(
                "%s:%s: GDA angular command is labeled mm; interpreting it as degrees",
                file.filename,
                angles_at,
            )
        # Older GDA exports (DIAD) write the paired rotation command with no unit
        # label; the layout's rotation stage moves in degrees.
        elif angles_at == gda[2] and not units.strip():
            units_override = "degree"
            acquisition.update(angle_units_recorded="", angle_units="degree")
    return acquisition, units_override


@dataclass(frozen=True)
class LocatedFrames:
    """Located HDF5 frames with optional frame labels and rotation angles.

    ``flats`` and ``darks`` are stacks of their own beside the frames, as in
    APS Data Exchange (``data_white``, ``data_dark``), when the file keeps
    them apart rather than marking them in ``image_key``.

    Parameters
    ----------
    stack : lazy HDF5 stack
        Located detector-frame stack, read only as indexed.
    image_key : numpy.ndarray or None
        Frame labels when present: sample, flat, dark or other.
    angles : numpy.ndarray or None
        One angle per frame, in degrees, when present.
    flats : lazy HDF5 stack or None, optional
        Separate open-beam frames, when not interleaved with the sample.
    darks : lazy HDF5 stack or None, optional
        Separate beam-off frames.
    acquisition : dict or None, optional
        Discovered acquisition paths and unit corrections.
    image_key_path : str or None, optional
        HDF5 path the frame labels were read from, when present.
    angles_path : str or None, optional
        HDF5 path the rotation angles were read from, when present.
    return_images : int, optional
        Frames the file marks as return (alignment) images, excluded from the views.
    """

    stack: Hdf5Frames
    image_key: np.ndarray | None
    angles: np.ndarray | None
    flats: Hdf5Frames | None = None
    darks: Hdf5Frames | None = None
    acquisition: JsonObject | None = None
    image_key_path: str | None = None
    angles_path: str | None = None
    return_images: int = 0

    @property
    def samples(self) -> Hdf5Frames:
        """The detector stack restricted to frames labeled as sample projections.

        Returns
        -------
        Hdf5Frames
            A lazy view of sample frames, or all frames for an unlabeled stack.
        """
        return self.stack if self.image_key is None else self.stack.frames_at(self.image_key == 0)

    @property
    def raw(self) -> bool:
        """Whether the file holds flat or dark frames: its frames are detector counts.

        Returns
        -------
        bool
            True when flat or dark frames mark the data as detector counts.
        """
        marked = self.image_key is not None and bool(np.isin(self.image_key, (1, 2)).any())
        return marked or self.flats is not None or self.darks is not None


def locate_frames(
    path: str,
    *,
    data_path: str | None = None,
    image_key_path: str | None = None,
    angles_path: str | None = None,
) -> LocatedFrames:
    """Find the stack of frames in an HDF5 file, and its ``image_key`` and angles.

    Each is read from the path given, else the NXtomo (or Data Exchange) path,
    else where the NeXus structure puts it, else the file's only dataset of that
    name (``data`` of two or three dimensions, ``image_key``, ``rotation_angle``)
    with one entry per frame. The NeXus structure gives the frames of the
    default plot (``@default`` to an NXdata group's ``@signal``), the
    ``image_key`` of the NXdetector holding them, and the rotation as the
    NXdata coordinate along the frames with angular units: its own, or those
    recorded for the positioner field of the same name in its entry. A uniquely
    named key or angle dataset with the wrong shape is rejected, not treated as
    missing. Flats and darks kept beside the frames (``data_white``,
    ``data_dark``) are found when ``image_key`` marks none of that kind, and so
    are angles there (``theta``). Angles in radians (a ``units`` attribute) are
    converted to degrees.

    A key or angles paired with the frames by path or structure may hold more
    or fewer entries than the stack holds frames, as when a scan stops early:
    the frames they all cover are read, from the first, with a warning, and the
    acquisition record keeps each count (``frame_records``).

    Parameters
    ----------
    path : str
        HDF5 acquisition file.
    data_path : str or None, optional
        Explicit frame dataset path; None uses discovery.
    image_key_path : str or None, optional
        Explicit frame-label dataset path; None uses discovery.
    angles_path : str or None, optional
        Explicit rotation-angle dataset path; None uses discovery.

    Returns
    -------
    LocatedFrames
        Lazy frame stacks plus frame labels, angles and acquisition metadata.

    Raises
    ------
    KeyError
        If frames or an explicitly requested dataset cannot be found.
    ValueError
        If discovery is ambiguous or per-frame metadata are malformed.
    OSError
        If the file cannot be opened.
    """
    with h5py.File(path, "r") as file:
        where, stack, gda = _stack(file, path, data_path)
        frames = int(stack.shape[0])

        def per_frame(item: h5py.Dataset) -> bool:
            return item.shape == (frames,)

        key_at = image_key_path or (_IMAGE_KEY_PATH if _dataset(file, _IMAGE_KEY_PATH) else None)
        key_at = key_at or (gda[1] if gda is not None else None)
        key_at = key_at or _detector_key(file, stack, path)
        key_by_name = key_at is None
        key_at = key_at or _only(file, "image_key", per_frame, path, validate_unmatched=True)
        angles_at = angles_path or next(
            (p for p in ROTATION_ANGLE_PATHS if _dataset(file, p) is not None), None
        )
        angles_at = angles_at or (gda[2] if gda is not None else None)
        angles_at = angles_at or _only(file, "rotation_angle", per_frame, path)
        parent = where.rsplit("/", 1)[0] or "/"
        beside = {n: f"{parent.rstrip('/')}/{n}" for n in ("data_white", "data_dark", "theta")}
        angles_at = angles_at or (beside["theta"] if _dataset(file, beside["theta"]) else None)
        coordinate = None if angles_at else _rotation_coordinate(file, stack, path)
        angles_at = angles_at or (coordinate[0] if coordinate is not None else None)
        angles_by_name = angles_at is None
        # A known Data Exchange angle path takes priority over unmatched names.
        angles_at = angles_at or _only(
            file, "rotation_angle", per_frame, path, validate_unmatched=True
        )
        raw_key, raw_angles, counts = _records(
            file, path, frames, (key_at, key_by_name), (angles_at, angles_by_name)
        )
        read = min(counts.values())
        key = None if raw_key is None else checked_image_key(raw_key, frames=read, path=path)
        control = (
            _dataset(file, f"{key_at.rsplit('/', 1)[0]}/image_key_control") if key_at else None
        )
        returns = 0
        if key is not None and control is not None and control.ndim == 1 and len(control) >= read:
            # ESRF marks return (alignment) images -1 here while image_key calls them samples.
            marked = np.asarray(control[:read]) == -1
            key, returns = np.where(marked, 3, key), int(np.count_nonzero(marked))
        acquisition, units_override = _gda_metadata(file, gda, where, key_at, angles_at)
        notes: JsonObject = {}
        if coordinate is not None and coordinate[2] is not None:
            units_override = coordinate[1]
            notes |= {
                "angle_units_recorded": "",
                "angle_units": coordinate[1],
                "angle_units_path": coordinate[2],
            }
        if len(set(counts.values())) > 1:
            notes["frame_records"] = dict(counts)
        if notes:
            paths: JsonObject = {
                "data_path": where,
                "image_key_path": key_at,
                "angles_path": angles_at,
            }
            acquisition = {**paths, **(acquisition or {}), **notes}
        angles = (
            None
            if angles_at is None or raw_angles is None
            else _degrees(
                file,
                angles_at,
                raw_angles,
                path,
                frames=read,
                key=key,
                units_override=units_override,
            )
        )
        flats, darks = (
            Hdf5Frames.open(path, beside[n])
            if (key is None or not np.any(key == label)) and _dataset(file, beside[n])
            else None
            for n, label in (("data_white", 1), ("data_dark", 2))
        )
    stack_frames = Hdf5Frames.open(path, where)
    return LocatedFrames(
        stack_frames if read == frames else stack_frames.frames_at(slice(0, read)),
        key,
        angles,
        flats,
        darks,
        acquisition,
        key_at,
        angles_at,
        returns,
    )


def _stack(
    file: h5py.File, path: str, data_path: str | None
) -> tuple[str, h5py.Dataset, tuple[str, str, str] | None]:
    """The frames' path and dataset, and the GDA layout they belong to, if any."""
    where = data_path or next((p for p in _FRAME_PATHS if _dataset(file, p) is not None), None)
    gda = _gda_paths(file) if where is None else None
    where = where or (gda[0] if gda is not None else None)
    gda = gda if gda is not None and where == gda[0] else None
    where = where or _default_frames(file)
    where = where or _only(file, "data", lambda d: len(d.shape) in (2, 3), path)
    stack = None if where is None else _dataset(file, where)
    if data_path is not None and stack is None:
        raise KeyError(f"{path} has no dataset {data_path!r}")
    if where is not None and stack is not None:
        return where, stack, gda
    missing: list[str] = []
    for name in file:
        link = file.get(name, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            try:
                file[name]
            except (KeyError, OSError):
                missing.append(f"/{name} -> {link.filename}:{link.path}")
    if missing:
        raise ValueError(
            f"{path}: no stack of frames found; unresolved external roots: "
            + "; ".join(missing)
            + ". Restore the referenced files and datasets, or supply a readable "
            "detector stack with data_path."
        )
    raise KeyError(f"{path}: no stack of frames found; name its dataset (data_path)")


def _records(
    file: h5py.File,
    path: str,
    frames: int,
    key: tuple[str | None, bool],
    angles: tuple[str | None, bool],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, int]]:
    """The image_key and angles, cut to the frames that the stack and both cover.

    ``key`` and ``angles`` are each a path and whether it was found by its name alone.
    Writers append frames, labels and positions in acquisition order, each in its own
    stream, so a scan that stops early leaves them at different lengths: the frames read
    are those, from the first, that every record covers, with a warning when that is not
    all of them. A record found only by its name is paired with the frames by its shape,
    so it is not cut and must fit. Returns the cut records and each one's length.
    """
    counts = {"data": frames}
    raw: dict[str, np.ndarray | None] = {}
    for name, (where, by_name) in (("image_key", key), ("angles", angles)):
        values = raw[name] = None if where is None else _read(file, where, path)
        if values is not None and values.ndim == 1 and not by_name:
            counts[name] = len(values)
    read = min(counts.values())
    if len(set(counts.values())) > 1:
        nouns = {"data": "frames", "image_key": "image_key entries", "angles": "angles"}
        logging.getLogger(__name__).warning(
            "%s has %s: reading the first %d frames, which all of them cover",
            path,
            ", ".join(f"{count} {nouns[name]}" for name, count in counts.items()),
            read,
        )

    def cut(values: np.ndarray | None) -> np.ndarray | None:
        fits = values is not None and values.ndim == 1 and len(values) >= read
        return values[:read] if fits and values is not None else values

    return cut(raw["image_key"]), cut(raw["angles"]), counts


def _text(value: object) -> str:
    """An HDF5 string attribute or scalar as text."""
    items = np.asarray(value).reshape(-1)
    item = items[0] if items.size else ""
    return item.decode() if isinstance(item, bytes) else str(item)


def _default_frames(file: h5py.File) -> str | None:
    """The frames the NeXus default plot shows: root and entry ``@default``, NXdata ``@signal``.

    Without a root ``@default`` each entry's own is followed; only one stack of frames
    among them is a default.
    """
    entries = [_text(file.attrs["default"])] if "default" in file.attrs else list(file)
    found: list[str] = []
    for name in entries:
        entry = file.get(name)
        if not isinstance(entry, h5py.Group) or "default" not in entry.attrs:
            continue
        plot = entry.get(_text(entry.attrs["default"]))
        if not isinstance(plot, h5py.Group) or "signal" not in plot.attrs:
            continue
        where = f"{plot.name}/{_text(plot.attrs['signal'])}"
        signal = _dataset(file, where)
        if signal is not None and signal.ndim in (2, 3):
            found.append(where)
    return found[0] if len(found) == 1 else None


def _groups_of_class(file: h5py.File, nx_class: str) -> list[h5py.Group]:
    groups: list[h5py.Group] = []

    def visit(_: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Group) and _text(item.attrs.get("NX_class", "")) == nx_class:
            groups.append(item)

    file.visititems(visit)
    return groups


def _detector_key(file: h5py.File, stack: h5py.Dataset, path: str) -> str | None:
    """The ``image_key`` of the NXdetector whose ``data`` is ``stack``.

    NeXus keeps a detector's frame labels beside its frames; another detector's
    labels in the same file describe other frames.
    """
    found: dict[str, h5py.Dataset] = {}
    for detector in _groups_of_class(file, "NXdetector"):
        key = detector.get("image_key")
        if (
            detector.get("data") == stack
            and isinstance(key, h5py.Dataset)
            and all(key != other for other in found.values())
        ):
            found[f"{detector.name}/image_key"] = key
    if len(found) > 1:
        raise KeyError(f"{path} has several 'image_key' datasets ({', '.join(found)}): name one")
    return next(iter(found), None)


_ANGULAR_UNITS = frozenset({"deg", "degree", "degrees", "rad", "radian", "radians"})


def _rotation_coordinate(
    file: h5py.File, stack: h5py.Dataset, path: str
) -> tuple[str, str, str | None] | None:
    """The rotation among the NXdata coordinates along ``stack``'s frames.

    NeXus names a signal's coordinates in its NXdata group: ``@axes`` and
    ``<name>_indices``. A one-dimensional coordinate along the frames with angular
    units is a rotation. A coordinate with no units of its own takes those of the
    fields of the same name elsewhere in its entry, when they agree: GDA writes
    each positioner's per-frame values without units and records its fields, with
    units, once per scan. The scan axis (``@axes``) wins among several rotations.

    Returns the coordinate's path, its units, and where the units came from when
    the coordinate has none (else None).
    """
    found: list[tuple[str, h5py.Dataset, str, str | None, bool]] = []
    for group in _groups_of_class(file, "NXdata"):
        if "signal" not in group.attrs or group.get(_text(group.attrs["signal"])) != stack:
            continue
        scan_axis = _text(group.attrs.get("axes", ".")) or "."  # the first axis: the frames'
        names: set[str] = set() if scan_axis == "." else {scan_axis}
        names |= {
            attribute.removesuffix("_indices")
            for attribute, value in group.attrs.items()
            if attribute.endswith("_indices") and np.asarray(value).reshape(-1).tolist() == [0]
        }
        entry = file[str(group.name).split("/")[1]]
        assert isinstance(entry, h5py.Group)  # the NXdata group lies in it
        for name in sorted(names):
            item = group.get(name)
            if not isinstance(item, h5py.Dataset) or item.ndim != 1:
                continue
            units, units_path = _text(item.attrs.get("units", "")), None
            if not units.strip():
                units, units_path = _recorded_units(entry, name)
            if units.strip().lower() in _ANGULAR_UNITS and all(item != f[1] for f in found):
                found.append((f"{group.name}/{name}", item, units, units_path, name == scan_axis))
    if len(found) > 1:
        found = [f for f in found if f[4]] or found
    if len(found) > 1:
        listed = ", ".join(f[0] for f in found)
        raise KeyError(
            f"{path} has several rotation coordinates ({listed}): name one (angles_path)"
        )
    if not found:
        return None
    where, _, units, units_path, _ = found[0]
    return where, units, units_path


def _recorded_units(entry: h5py.Group, name: str) -> tuple[str, str | None]:
    """The units of the fields named ``name`` in ``entry`` if they agree, and one's path."""
    recorded: dict[str, str] = {}

    def visit(where: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset) and where.rsplit("/", 1)[-1] == name:
            units = _text(item.attrs.get("units", "")).strip()
            if units and units not in recorded:
                recorded[units] = f"{entry.name}/{where}"

    entry.visititems(visit)
    if len(recorded) != 1:
        return "", None
    ((units, where),) = recorded.items()
    return units, where


def _dataset(file: h5py.File, where: str) -> h5py.Dataset | None:
    found = file.get(where)
    if found is None:
        # get() reads a dangling link as absence: missing frame labels would make
        # every flat and dark a sample view.
        link = file.get(where, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError(
                f"{where}: external data unavailable; supply {link.filename!r} "
                f"beside {file.filename!r}, containing dataset {link.path!r}"
            )
        if isinstance(link, h5py.SoftLink):
            raise ValueError(f"{where}: links to {link.path!r}, which {file.filename!r} lacks")
    return found if isinstance(found, h5py.Dataset) else None


def _read(file: h5py.File, where: str, path: str) -> np.ndarray:
    found = _dataset(file, where)
    if found is None:
        raise KeyError(f"{path} has no dataset {where!r}")
    return np.asarray(found[()])


def _degrees(
    file: h5py.File,
    where: str,
    values: np.ndarray,
    path: str,
    *,
    frames: int,
    key: np.ndarray | None,
    units_override: str | None = None,
) -> np.ndarray | None:
    units = file[where].attrs.get("units", file[where].attrs.get("unit", b""))
    units = units.decode() if isinstance(units, bytes) else str(units)
    degrees = angles_in_degrees(
        values, units_override or units, frames=frames, path=path, image_key=key
    )
    if not units_override and not units.strip() and not _degree_convention(file, where):
        # A generic motor dataset has no implied angular unit. Keep the angle
        # record incomplete so load_frames(..., angles=...) can replace it.
        return None
    return degrees


def _degree_convention(file: h5py.File, where: str) -> bool:
    """NXtomo and Data Exchange prescribe degrees; arbitrary HDF5 names do not."""
    if where == "/exchange/theta" or where in ROTATION_ANGLE_PATHS:
        return True
    for suffix in (
        "/sample/rotation_angle",
        "/sample/transformations/rotation_angle",
        "/data/rotation_angle",
    ):
        if not where.endswith(suffix):
            continue
        root = where[: -len(suffix)]
        while root:
            found = file.get(root + "/definition")
            # NXtomo is declared by a definition dataset, or by an attribute of the entry.
            definition = (
                found[()] if isinstance(found, h5py.Dataset) else file[root].attrs.get("definition")
            )
            if definition is not None:
                value = np.asarray(definition).item()
                value = value.decode() if isinstance(value, bytes) else str(value)
                if value == "NXtomo":
                    return True
            root = root.rsplit("/", 1)[0]
    return False


def _only(
    file: h5py.File,
    name: str,
    fits: Callable[[h5py.Dataset], bool],
    path: str,
    *,
    validate_unmatched: bool = False,
) -> str | None:
    """Find a fitting named dataset, or one unmatched candidate to validate.

    Shape still disambiguates different detectors. With ``validate_unmatched``,
    a malformed uniquely named acquisition dataset is not mistaken for absent
    metadata; its caller validates it. More than one candidate is ambiguous.
    """
    found: list[str] = []
    named: list[str] = []

    seen = set()

    def visit(where: str) -> None:
        if where.rsplit("/", 1)[-1] != name:
            return
        item = _dataset(file, where)
        if item is not None and item.id not in seen:
            # NXdata and NXdetector commonly link to the same pixels. A link
            # alias is not a second acquisition, even across external files.
            seen.add(item.id)
            named.append("/" + where)
            if fits(item):
                found.append("/" + where)

    file.visit_links(visit)
    if validate_unmatched and not found:
        found = named
    if len(found) > 1:
        raise KeyError(f"{path} has several {name!r} datasets ({', '.join(found)}): name one")
    return found[0] if found else None
