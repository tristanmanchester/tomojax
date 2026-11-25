"""Registered worker reads and exact slab-wise statistics for local arrays."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
import math
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any

import h5py
import msgspec
import numpy as np

from tomojax._files import saved_volume_grid
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._arguments import explicit_dataset
from tomojax.cli._frame_reads import WindowedFrames, frame_pixels
from tomojax.cli._handles import stage_array
from tomojax.cli._planes import plane_windows
from tomojax.cli._results import disk_volume, stored_grid
from tomojax.core.host_volume import host_array, release
from tomojax.errors import InvalidParameter
from tomojax.io.projection_layout import saved_npz_layout
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Member, Storage, Uniform

if TYPE_CHECKING:
    import jax

    from tomojax._data import FrameStack

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
) -> Iterator[tuple[Any, Array]]:
    """Keep HDF5 open only while its selected datasets are in use."""
    if isinstance(source, Member):
        registered = context.leases[source].value
        if not isinstance(registered, ex.RegisteredArray):
            raise InvalidParameter("source", "choose a registered array member")
        yield registered.data, registered.metadata
        return
    path = Path(source.path)
    if isinstance(source, op.PathInput):
        with saved_volume(path, "source") as volume:
            yield volume
        return
    if isinstance(source, Hdf5):
        with h5py.File(path, "r") as file:
            data = explicit_dataset(file, source, "source")
            if (
                "/" + source.dataset.lstrip("/") == p.VOLUME_DATASET
                and "volume_axes_order" in file["/entry/processing/tomojax"].attrs
            ):
                grid, _, geom_meta, axes = stored_grid(file, path)
                yield data, disk_volume(path, grid, axes, data.shape, data.dtype.str, geom_meta)
            else:
                yield data, indexed(source, data)
        return
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    lease = context.leases.get(source)
    yield data, (lease and lease.metadata) or indexed(source, data)


@contextmanager
def saved_volume(path: Path, field: str) -> Iterator[tuple[Any, Array]]:
    """Open only the volume and its registration in logical voxel order."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            data = file.get(p.VOLUME_DATASET)
            if isinstance(data, h5py.Dataset):
                grid, _, geom_meta, axes = stored_grid(file, path)
                volume = SavedVolume(data, axes)
                yield (
                    volume,
                    disk_volume(path, grid, "zyx", volume.shape, volume.data.dtype.str, geom_meta),
                )
                return
    elif path.is_file() and path.suffix.lower() == ".npz":
        record, arrays = saved_npz_layout(path, volume_only=True)
        if (volume := arrays.get("volume")) is not None:
            grid = saved_volume_grid(
                record.grid.to_dict() if record.grid else None, record.geometry_metadata
            )
            yield (
                volume,
                disk_volume(
                    path, grid, "zyx", volume.shape, volume.dtype.str, record.geometry_metadata
                ),
            )
            return
    raise InvalidParameter(field, op.ACQUISITION_ARRAY_ERROR)


def project_volume(path: Path, context: ex.ExecutionContext, *, shape_only: bool) -> np.ndarray:
    """Read the registered volume in bounded tiles, or just its shape for preflight."""
    with saved_volume(path, "volume") as (data, meta):
        if shape_only:
            return np.broadcast_to(np.zeros((), meta.dtype), meta.shape)
        # The projector consumes the full array. Stage it in bounded reads,
        # using the library's file-backed allocation for volumes larger than RAM.
        volume = host_array(meta.shape, dtype=meta.dtype)
        ranges = tuple(range(n) for n in meta.shape)
        for tile in ordered_tiles(ranges, READ_BYTES // volume.itemsize):
            context.control.checkpoint()
            key = tuple(slice(r.start, r.stop) for r in tile)
            volume[key] = data[key]
            release(volume)
        return volume


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
    if isinstance(data, WindowedFrames):
        return frame_pixels(data, ranges, READ_BYTES)
    keys = tuple(slice(r.start, r[-1] + 1, r.step) for r in ranges)
    return np.asarray(data[keys])


def browse(
    request: op.Read | op.Statistics, context: ex.ExecutionContext
) -> p.Layout | p.ArrayResult | p.DisplayPlanes | p.StatisticsResult:
    with source_array(request.source, context) as (data, meta):
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
        if request.mode is None:
            return p.ArrayResult(read_window(context, data, meta, ranges))
        if request.mode in ("projection", "sinogram") and meta.domain == "attenuation":
            raise InvalidParameter(
                "source",
                "Source is a volume. Select .projections on its scan or use a detector dataset.",
            )
        planes = []
        windows = list(
            plane_windows(
                [a.name for a in meta.axes],
                meta.shape,
                request.mode,
                request.axis,
                request.index,
                tuple((r.start, r.stop) for r in ranges),
                tuple(r.step for r in ranges),
            )
        )
        tiles = [
            tuple(range(lo, hi, s) for (lo, hi), s in zip(roi, stride, strict=True))
            for _, _, roi, stride, _ in windows
        ]
        for (dim, at, _, _, sampling), tile in zip(windows, tiles, strict=True):
            planes.append(p.DisplayPlane(read_window(context, data, meta, tile), dim, at, sampling))
        return p.DisplayPlanes(tuple(planes))


def read_window(
    context: ex.ExecutionContext,
    data: ex.Readable | SavedVolume | h5py.Dataset | jax.Array | np.ndarray | FrameStack,
    meta: Array,
    ranges: tuple[range, ...],
) -> Array:
    shape = tuple(map(len, ranges))
    if math.prod(shape) * np.dtype(meta.dtype).itemsize > READ_BYTES:
        raise InvalidParameter("roi", "read exceeds 16 MiB; use a stride or a smaller ROI")
    axes = []
    for axis, r in zip(meta.axes, ranges, strict=True):
        coord = axis.coordinates
        if isinstance(coord, Uniform):
            selected = Uniform(coord.origin + r.start * coord.spacing, r.step * coord.spacing)
        else:
            with source_array(coord.values, context) as (v, _):
                values = pixels(v, (r,))
            selected = Coordinates(stage_array(context, values))
        axes.append(msgspec.structs.replace(axis, coordinates=selected))
    columns = meta.columns
    domain = meta.domain
    if columns and len(ranges[-1]) != len(columns):
        columns, domain = (), "pose_components"
    return Array(
        stage_array(context, pixels(data, ranges)),
        shape,
        meta.dtype,
        tuple(axes),
        meta.unit,
        domain,
        meta.fixed,
        columns,
    )


def ordered_tiles(ranges: tuple[range, ...], capacity: int) -> Iterator[tuple[range, ...]]:
    """Partition a window into C-order spans of at most capacity elements."""
    block = [1] * len(ranges)
    for i in reversed(range(len(ranges))):
        block[i] = max(1, min(capacity, len(ranges[i])))
        capacity //= block[i]
    starts = (range(0, len(r), step) for r, step in zip(ranges, block, strict=True))
    for origin in product(*starts):
        yield tuple(r[start : start + n] for r, start, n in zip(ranges, origin, block, strict=True))


def exact_percentiles(
    samples: np.ndarray,
    percentiles: tuple[float, ...],
    low: float,
    high: float,
    control: ex.Control,
) -> np.ndarray:
    """Find bracketing order statistics by counting in independently sorted runs."""
    step = READ_BYTES // samples.itemsize
    if samples.size <= step:
        control.checkpoint()
        return np.percentile(samples, percentiles, overwrite_input=True)
    starts = range(0, samples.size, step)
    for start in starts:
        control.checkpoint()
        samples[start : start + step].sort()

    positions = (np.asarray(percentiles) / 100) * (samples.size - 1)
    ranks = np.unique(np.concatenate((np.floor(positions), np.ceil(positions)))).astype(np.int64)
    lower = np.full(ranks.shape, low)
    upper = np.full(ranks.shape, high)
    while np.any(lower < upper):
        # Keep each pivot below its upper endpoint even if the midpoint rounds up.
        pivots = np.minimum(lower * 0.5 + upper * 0.5, np.nextafter(upper, lower))
        counts = np.zeros_like(ranks)
        below = np.full_like(lower, -np.inf)
        above = np.full_like(upper, np.inf)
        for start in starts:
            control.checkpoint()
            run = samples[start : start + step]
            split = np.searchsorted(run, pivots, side="right")
            counts += split
            below = np.maximum(below, np.where(split > 0, run[split - 1], -np.inf))
            above = np.minimum(
                above, np.where(split < run.size, run[split.clip(max=run.size - 1)], np.inf)
            )
        # Snap to actual samples on either side. The retained interval always
        # contains the requested rank and loses at least one distinct value.
        move_lower = counts <= ranks
        lower = np.where(move_lower, above, lower)
        upper = np.where(move_lower, upper, below)
    return np.interp(positions, ranks, lower)


def statistics(
    request: op.Statistics,
    context: ex.ExecutionContext,
    data: ex.Readable | SavedVolume | h5py.Dataset | jax.Array | np.ndarray | FrameStack,
    meta: Array,
    ranges: tuple[range, ...],
) -> p.StatisticsResult:
    """Spool finite samples for exact quantiles while bounding source reads and resident buffers."""
    if not 1 <= request.bins <= 4096:
        raise InvalidParameter("bins", "choose 1 to 4096 histogram bins")
    if any(not 0 <= q <= 100 for q in request.percentiles):
        raise InvalidParameter("percentiles", "percentiles must be between 0 and 100")
    count = 0
    mean = m2 = 0.0
    low, high = math.inf, -math.inf
    with tempfile.TemporaryFile() as stream:
        for tile in ordered_tiles(ranges, READ_BYTES // 8):
            context.control.checkpoint()
            chunk = pixels(data, tile)
            finite = chunk[np.isfinite(chunk)].astype(np.float64)
            n = len(finite)
            if n:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
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
                qvalues = exact_percentiles(
                    samples, request.percentiles, low, high, context.control
                )
                values = dict(zip(map(str, request.percentiles), map(float, qvalues), strict=True))
            edges = tuple(map(float, np.histogram_bin_edges([low, high], bins=request.bins)))
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
