"""Registered worker reads and exact slab-wise statistics for local arrays."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from itertools import product
import math
from pathlib import Path
import tempfile
from typing import Any

import h5py
import msgspec
import numpy as np

import tomojax as tj
from tomojax._files import saved_volume_grid
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._frame_reads import frame_pixels
from tomojax.cli._handles import stage_array
from tomojax.cli._results import disk_volume, scan_axes, stored_grid
from tomojax.core.registration import recorded_length_unit
from tomojax.errors import InvalidParameter
from tomojax.io import build_geometry_from_dataset_metadata
from tomojax.io.api import load_nxtomo, mapped_npz
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Member, Storage, Uniform

READ_BYTES = 16 << 20


class SavedVolume:
    """Translate logical (z, y, x) slices to HDF5 hyperslabs before reading."""

    def __init__(self, data: h5py.Dataset, axes: str) -> None:
        self.data = data
        self.axes = axes
        self.to_zyx = tuple(axes.index(axis) for axis in "zyx")
        self.shape = tuple(data.shape[i] for i in self.to_zyx)

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        stored = tuple(key["zyx".index(axis)] for axis in self.axes)
        return np.asarray(self.data[stored]).transpose(self.to_zyx)


@contextmanager
def source_array(
    source: Storage | op.PathInput, context: ex.ExecutionContext
) -> Iterator[tuple[Any, Array, dict[str, np.ndarray]]]:
    """Keep HDF5 open only while its selected datasets are in use."""
    if isinstance(source, Member):
        registered = context.leases[source].value
        if not isinstance(registered, ex.RegisteredArray):
            raise InvalidParameter("source", "choose a registered array member")
        yield registered.data, registered.metadata, {}
        return
    path = Path(source.path)
    if isinstance(source, op.PathInput) and h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            if p.VOLUME_DATASET in file:
                source = Hdf5(str(path), p.VOLUME_DATASET)
    if isinstance(source, op.PathInput):
        yield from saved_array(path)
        return
    if isinstance(source, Hdf5):
        with h5py.File(path, "r") as file:
            data = file.get(source.dataset)
            if not isinstance(data, h5py.Dataset):
                raise InvalidParameter("source", f"no dataset {source.dataset} in {path}")
            if "/" + source.dataset.lstrip("/") == p.VOLUME_DATASET:
                grid, _, geom_meta, axes = stored_grid(file, path)
                volume = SavedVolume(data, axes)
                yield (
                    volume,
                    disk_volume(path, grid, "zyx", volume.shape, data.dtype.str, geom_meta),
                    {},
                )
            else:
                yield data, indexed(source, data), {}
        return
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    lease = context.leases.get(source)
    yield data, (lease and lease.metadata) or indexed(source, data), {}


def saved_array(path: Path) -> Iterator[tuple[Any, Array, dict[str, np.ndarray]]]:
    """Read the primary member of a saved volume or projection stack."""
    with ExitStack() as files:
        saved = (
            files.enter_context(mapped_npz(str(path)))
            if path.suffix.lower() == ".npz"
            else load_nxtomo(str(path), lazy=True)
        )
        meta = saved.metadata
        if meta.volume is not None:
            grid_meta = meta.grid.to_dict() if isinstance(meta.grid, tj.Grid) else meta.grid
            grid = saved_volume_grid(grid_meta, meta.geometry_meta)
            data = meta.volume
            yield (
                data,
                disk_volume(
                    path,
                    grid,
                    "zyx",
                    data.shape,
                    np.dtype(data.dtype).str,
                    meta.geometry_meta or {},
                ),
                {},
            )
        else:
            data = saved.projections
            if data is None:
                raise InvalidParameter("source", f"No array in {path}")
            _, detector, geometry = build_geometry_from_dataset_metadata(
                saved.geometry_inputs(), apply_poses=True
            )
            angles = np.asarray(geometry.angles)
            view = Axis(
                "view",
                "acquisition",
                "degree",
                Coordinates(Hdf5(str(path), "/entry/sample/transformations/rotation_angle")),
            )
            axes = scan_axes(
                detector, view, recorded_length_unit(meta.geometry_meta or {}) or "scan_length"
            )
            yield (
                data,
                Array(
                    Hdf5(str(path), "/entry/instrument/detector/data"),
                    data.shape,
                    np.dtype(data.dtype).str,
                    axes,
                    "1",
                    "line_integrals",
                ),
                {"view": angles},
            )


def indexed(storage: Storage, data: Any) -> Array:
    axes = tuple(Axis(f"axis{i}", "array", "index", Uniform(0, 1)) for i in range(data.ndim))
    return Array(storage, data.shape, data.dtype.str, axes, "unspecified", "samples")


def window(shape: tuple[int, ...], request: op.Window) -> tuple[range, ...]:
    roi = request.roi if request.roi is not None else tuple((0, n) for n in shape)
    stride = request.stride if request.stride is not None else (1,) * len(shape)
    if len(stride) != len(shape) or any(s < 1 for s in stride):
        raise InvalidParameter("stride", "supply one positive stride per axis")
    if len(roi) != len(shape) or any(
        not 0 <= a < b <= n for (a, b), n in zip(roi, shape, strict=False)
    ):
        raise InvalidParameter("roi", f"expected nonempty half-open intervals inside {shape}")
    return tuple(range(a, b, s) for (a, b), s in zip(roi, stride, strict=True))


def pixels(data: Any, ranges: tuple[range, ...]) -> np.ndarray:
    if hasattr(data, "window"):
        return frame_pixels(data, ranges, READ_BYTES)
    keys = tuple(slice(r.start, r[-1] + 1, r.step) for r in ranges)
    return np.asarray(data[keys])


def browse(
    request: op.Read | op.Statistics, context: ex.ExecutionContext
) -> p.Layout | p.ArrayResult | p.StatisticsResult:
    with source_array(request.source, context) as (data, meta, vectors):
        if np.dtype(meta.dtype).kind not in "biuf":
            raise InvalidParameter("source", f"expected real numeric data, got {meta.dtype}")
        ranges = window(meta.shape, request)
        if isinstance(request, op.Statistics):
            return statistics(request, context, data, meta, ranges)
        if request.metadata_only:
            layout = msgspec.to_builtins(meta)
            del layout["storage"]
            for axis in layout["axes"]:
                axis["coordinates"].pop("values", None)
            return p.Layout(layout)
        shape = tuple(map(len, ranges))
        if math.prod(shape) * np.dtype(meta.dtype).itemsize > READ_BYTES:
            raise InvalidParameter("roi", "read exceeds 16 MiB; use a stride or a smaller ROI")
        axes = []
        for axis, r in zip(meta.axes, ranges, strict=True):
            coord = axis.coordinates
            if isinstance(coord, Uniform):
                selected = Uniform(coord.origin + r.start * coord.spacing, r.step * coord.spacing)
            else:
                if axis.name in vectors:
                    values = vectors[axis.name][r.start : r.stop : r.step]
                else:
                    with source_array(coord.values, context) as (v, _, _):
                        values = pixels(v, (r,))
                selected = Coordinates(stage_array(context, values))
            axes.append(msgspec.structs.replace(axis, coordinates=selected))
        columns = meta.columns
        domain = meta.domain
        if columns and len(ranges[-1]) != len(columns):
            columns, domain = (), "pose_components"
        return p.ArrayResult(
            Array(
                stage_array(context, pixels(data, ranges)),
                shape,
                meta.dtype,
                tuple(axes),
                meta.unit,
                domain,
                meta.fixed,
                columns,
            )
        )


def statistics(request, context, data, meta, ranges):
    """Spool finite samples for exact quantiles while bounding source reads and resident buffers."""
    if not 1 <= request.bins <= 4096:
        raise InvalidParameter("bins", "choose 1 to 4096 histogram bins")
    if any(not 0 <= q <= 100 for q in request.percentiles):
        raise InvalidParameter("percentiles", "percentiles must be between 0 and 100")
    block = [1] * len(ranges)
    capacity = READ_BYTES // 8
    for i in reversed(range(len(ranges))):
        block[i] = max(1, min(capacity, len(ranges[i])))
        capacity //= block[i]
    starts = (range(0, len(r), step) for r, step in zip(ranges, block, strict=True))
    count = 0
    mean = m2 = 0.0
    with tempfile.TemporaryFile() as stream:
        for origin in product(*starts):
            context.control.checkpoint()
            tile = tuple(
                r[start : start + n] for r, start, n in zip(ranges, origin, block, strict=True)
            )
            chunk = pixels(data, tile).ravel()
            finite = chunk[np.isfinite(chunk)].astype(np.float64)
            n = len(finite)
            if n:
                finite.tofile(stream)
                delta = float(finite.mean()) - mean
                mean += delta * n / (count + n)
                m2 += float(finite.var()) * n + delta**2 * count * n / (count + n)
                count += n
        edges, histogram = (), ()
        values = dict.fromkeys(map(str, request.percentiles))
        if count:
            samples = np.memmap(stream, np.float64, "r+", shape=(count,))
            if values:
                qvalues = np.percentile(samples, request.percentiles, overwrite_input=True)
                values = dict(zip(map(str, request.percentiles), map(float, qvalues), strict=True))
            edges = tuple(
                map(
                    float, np.histogram_bin_edges([samples.min(), samples.max()], bins=request.bins)
                )
            )
            bins = np.zeros(request.bins, np.int64)
            for i in range(0, count, READ_BYTES // 8):
                context.control.checkpoint()
                bins += np.histogram(samples[i : i + READ_BYTES // 8], bins=edges)[0]
            histogram = tuple(map(int, bins))
    return p.StatisticsResult(
        count,
        math.prod(map(len, ranges)) - count,
        mean if count else None,
        math.sqrt(m2 / count) if count else None,
        values,
        edges,
        histogram,
        meta.unit,
    )
