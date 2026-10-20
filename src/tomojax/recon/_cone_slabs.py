"""Cone-beam volumes kept in host memory and streamed through the device in z slabs.

A slab of whole z planes reaches a band of detector rows in every view, its
row window. Each slab is projected onto, and transposed from, that window
with one set of per-view coefficients, so the sum of the slab operators is a
cone projector whose transpose stays exact. A sweep projects the slabs in
window order into a ring of detector rows, finishes each row once every slab
reaching it has been projected, and transposes each slab as soon as its whole
window is finished. The ring, the slabs projected but not yet transposed and
the transposes in progress occupy the device; host volumes are read and
written one slab at a time.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
import math
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from tomojax.core.cone import ConeModel
from tomojax.core.host_volume import (
    check_rows,
    fill,
    host_volume,
    release,
    release_planes,
    scratch_directory,
)
from tomojax.geometry import grid_volume_origin

from ._host_arrays import validate_host_arrays
from ._slab_engines import CudaEngine, Engine, JaxEngine, engine_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.geometry import Detector, Geometry, Grid

_BAND = 64  # rows are finished in bands of [k _BAND, (k + 1) _BAND)


def slab_windows(
    frames: np.ndarray, poses: np.ndarray, grid: Grid, detector: Detector, depth: int
) -> list[tuple[int, int]]:
    """Detector rows ``[first, stop)`` whose rays can sample each ``depth``-plane slab.

    ``frames`` are the views' lab detector centre, u and v directions and
    source (see :meth:`ConeModel.frames`). Joseph taps reach a voxel from
    sample points less than one voxel away, so each slab's box is widened by
    one voxel on every side; a box projects inside the hull of its projected
    corners. One row of margin covers rounding.
    """
    origin = np.asarray(grid_volume_origin(grid))
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    lo, hi = origin - spacing, origin + np.asarray([grid.nx, grid.ny, grid.nz]) * spacing
    starts = np.arange(0, grid.nz, depth)
    stops = np.minimum(starts + depth, grid.nz)
    z = origin[2] + np.stack([starts - 1, stops], axis=1) * grid.vz
    boxes = np.array(
        [[(x, y, h) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for h in pair] for pair in z]
    )
    frames = np.asarray(frames, np.float64)
    centre, u_dir, v_dir, source = (frames[:, i] for i in range(4))
    normal = np.cross(u_dir, v_dir)
    world = np.einsum("nij,kcj->nkci", poses[:, :3, :3], boxes) + poses[:, None, None, :3, 3]
    ray = world - source[:, None, None]
    depth_along = np.einsum("nkci,ni->nkc", ray, normal)
    scale = np.einsum("ni,ni->n", centre - source, normal)[:, None, None]
    behind = np.any(depth_along * scale <= 0, axis=(0, 2))
    hit = (
        source[:, None, None]
        + ray * (scale / np.where(depth_along == 0, 1, depth_along))[..., None]
    )
    rows = np.einsum("nkci,ni->nkc", hit - centre[:, None, None], v_dir) / detector.dv
    rows += (detector.nv - 1) / 2
    windows = []
    for low, high, unbounded in zip(
        rows.min(axis=(0, 2)), rows.max(axis=(0, 2)), behind, strict=True
    ):
        if unbounded:
            windows.append((0, detector.nv))
            continue
        first = max(0, min(detector.nv - 1, math.floor(low) - 1))
        windows.append((first, max(first + 1, min(detector.nv, math.ceil(high) + 2))))
    return windows


class RowReader:
    """Projections read a band of detector rows at a time, for every view.

    ``read(first, stop)`` returns rows ``[first, stop)`` of all views, as an
    array of shape ``(views, stop - first, nu)`` or an object holding one as
    ``projections``, such as the reader of
    :func:`tomojax.corrections.corrected_row_source`; ``shape`` is the whole
    stack's ``(views, nv, nu)``. The cone slab solvers accept one in place of
    a projection array and read each row once per sweep, so a stack larger
    than memory is corrected as it is needed. Rows must be finite in FP32.

    Rows are read in bands of 64 by one worker thread, which reads the band
    after the one in use (after the last, the first again) while the solver
    works on it; at most two bands are held.

    Parameters
    ----------
    read : callable
        ``read(first, stop)`` returning every view's rows ``[first, stop)``.
    shape : tuple of int
        Shape ``(views, nv, nu)`` of the whole projection stack.
    """

    def __init__(self, read: Callable[[int, int], Any], shape: tuple[int, int, int]) -> None:
        self.read, self.shape, self.ndim, self.dtype = read, tuple(shape), 3, np.dtype(np.float32)
        self._band: tuple[int, np.ndarray] | None = None
        self._ahead: tuple[int, Future[np.ndarray]] | None = None
        self._first: int | None = None
        self._worker: ThreadPoolExecutor | None = None

    def _read(self, k: int) -> np.ndarray:
        value = self.read(k * _BAND, min(self.shape[1], (k + 1) * _BAND))
        rows = np.asarray(getattr(value, "projections", value), np.float32)
        check_rows(rows, "projections")
        return rows

    def band(self, first: int, stop: int) -> tuple[np.ndarray, int, int]:
        """The band holding rows ``[first, stop)``, all in one band, and their offsets in it."""
        k = first // _BAND
        if self._band is None or self._band[0] != k:
            self._band = None
            worker = self._worker = self._worker or ThreadPoolExecutor(1)
            ahead, self._ahead = self._ahead, None
            if ahead is None or ahead[0] != k:
                if ahead is not None:
                    ahead[1].cancel()
                ahead = (k, worker.submit(self._read, k))
            self._band = (k, ahead[1].result())
            self._first = k if self._first is None else self._first
            following = k + 1 if (k + 1) * _BAND < self.shape[1] else self._first
            self._ahead = (following, worker.submit(self._read, following))
        return self._band[1], first - k * _BAND, stop - k * _BAND

    def close(self) -> None:
        """Drop the held bands and stop the worker, after any read in progress."""
        if self._ahead is not None:
            self._ahead[1].cancel()
        if self._worker is not None:
            self._worker.shutdown()
        self._band = self._ahead = self._worker = None


def host_projections(projections: Any, context: str, *, nonnegative: bool = False) -> Any:
    """A :class:`RowReader` as it is, otherwise a NumPy stack checked a few views at a time."""
    if isinstance(projections, RowReader):
        return projections
    data = np.asarray(projections)
    check_rows(data, context, nonnegative=nonnegative)
    return data


def window_coefficients(coefficients: Any, row: int, z: int) -> Any:
    """Coefficients of the window starting at detector ``row`` on the slab starting at ``z``.

    ``coefficients`` is a NumPy or CuPy ``(views, 25)`` FP32 array; the result is a new one.
    """
    shifted = coefficients.astype(np.float32, copy=True)
    shifted[:, 2] -= np.float32(z)
    shifted[:, 5] -= np.float32(z)
    shifted[:, 3:6] += np.float32(row) * shifted[:, 9:12]
    shifted[:, 24] -= np.float32(row)
    return shifted


@dataclass(frozen=True)
class SlabPlan:
    """Slabs of ``depth`` z planes, each with its row window, in sweep order.

    ``windows[k]`` is slab ``k``'s rows ``[first, stop)``. Windows are made
    monotone (first and stop never decrease along ``order``), which only widens
    them. ``ring`` is the number of rows a sweep holds at once, and ``lag``
    the most slabs projected but not yet transposed.
    """

    grid: Grid
    detector: Detector
    depth: int
    windows: tuple[tuple[int, int], ...]
    order: tuple[int, ...]
    ring: int
    lag: int

    @property
    def count(self) -> int:
        """The number of slabs."""
        return len(self.windows)

    def planes(self, k: int) -> tuple[int, int]:
        """Slab ``k``'s z planes ``[z0, z1)``."""
        return k * self.depth, min(self.grid.nz, (k + 1) * self.depth)


def plan_slabs(
    frames: np.ndarray,
    poses: np.ndarray,
    grid: Grid,
    detector: Detector,
    depth: int,
    *,
    common_height: bool = False,
) -> SlabPlan:
    """Plan ``depth``-plane slabs; ``common_height`` widens every window to the tallest."""
    windows = slab_windows(frames, poses, grid, detector, depth)
    count = len(windows)
    order = tuple(sorted(range(count), key=lambda k: windows[k][0]))
    # Monotone windows: suffix minimum of first and prefix maximum of stop.
    lowest = detector.nv
    for k in reversed(order):
        lowest = min(lowest, windows[k][0])
        windows[k] = (lowest, windows[k][1])
    highest = 0
    for k in order:
        highest = max(highest, windows[k][1])
        windows[k] = (windows[k][0], highest)
    if common_height:
        # One common height, keeping each window inside the detector.
        height = max(stop - first for first, stop in windows)
        windows = [
            (first, first + height)
            for first in (min(first, detector.nv - height) for first, _ in windows)
        ]
    ring = lag = projected = 0
    for position, k in enumerate(order):
        while projected < count and windows[order[projected]][0] < windows[k][1]:
            projected += 1
        top = max(windows[j][1] for j in order[: max(projected, position + 1)])
        ring = max(ring, top - windows[k][0])
        lag = max(lag, projected - position)
    return SlabPlan(grid, detector, depth, tuple(windows), order, ring, lag)


class HostVolume:
    """Slab access to a host ``(nx, ny, nz)`` array, preferably Fortran ordered.

    In Fortran order a slab is one contiguous block, which also suits
    memory-mapped files, whose touched pages are released after each access.
    A file-backed volume is read ahead (:meth:`prefetch`) and written behind
    by one worker thread, so its disk traffic overlaps device work; reads and
    writes run in the order they were requested. :meth:`close` waits for them.
    """

    _PENDING = 2  # slabs written behind before the caller waits

    def __init__(self, array: np.ndarray, engine: Engine, depth: int) -> None:
        self.array, self.engine, self.depth = array, engine, depth
        mapped = getattr(array, "_mmap", None) is not None
        self._worker = ThreadPoolExecutor(1) if mapped else None
        self._reads: dict[tuple[int, int], Future[np.ndarray]] = {}
        self._writes: list[Future[None]] = []
        self._held: dict[tuple[int, int], np.ndarray] = {}

    def _read(self, z0: int, z1: int) -> np.ndarray:
        block = np.ascontiguousarray(self.array[:, :, z0:z1].T)
        release_planes(self.array, z0, z1)
        return block

    def _write(self, z0: int, z1: int, block: np.ndarray) -> None:
        self.array[:, :, z0:z1].T[...] = block
        release_planes(self.array, z0, z1)

    def prefetch(self, z0: int, z1: int) -> None:
        """Start reading planes ``[z0, z1)`` of a file-backed volume."""
        if self._worker is not None and (z0, z1) not in self._reads:
            self._reads[z0, z1] = self._worker.submit(self._read, z0, z1)

    def close(self) -> None:
        """Wait for every write behind and stop the worker; a failed write raises here."""
        while self._writes:
            self._writes.pop(0).result()
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None

    def view(self, z0: int, z1: int) -> Any:
        """Planes ``[z0, z1)`` on the device as a strided ``(nx, ny, z1 - z0)`` view.

        It holds one copy of the slab, for elementwise use; kernels need :meth:`get`.
        """
        return self.engine.upload(self._block(z0, z1)).transpose(2, 1, 0)

    def _block(self, z0: int, z1: int) -> np.ndarray:
        """Host planes ``[z0, z1)`` as a contiguous ``(z, y, x)`` block, held or read."""
        block = self._held.pop((z0, z1), None)
        if block is None:
            self.prefetch(z0, z1)
            read = self._reads.pop((z0, z1), None)
            block = self._read(z0, z1) if read is None else read.result()
        return block

    def get(self, z0: int, z1: int, *, retain: bool = False) -> Any:
        """Planes ``[z0, z1)`` on the device in kernel layout, zero-padded to the slab depth.

        With ``retain`` the host copy is held for the next read of these planes.
        """
        block = self._block(z0, z1)
        if retain:
            self._held[z0, z1] = block
        return self.kernel(self.engine.upload(block).transpose(2, 1, 0))

    def kernel(self, value: Any) -> Any:
        """A device slab view in kernel layout, zero-padded to the slab depth."""
        xp = self.engine.xp
        volume = xp.ascontiguousarray(value)
        if volume.shape[2] < self.depth and isinstance(self.engine, JaxEngine):
            volume = xp.pad(volume, ((0, 0), (0, 0), (0, self.depth - volume.shape[2])))
        return volume

    def put(self, z0: int, z1: int, value: Any, scratch: Any = None) -> None:
        """Store the first ``z1 - z0`` planes of a device slab.

        ``scratch``, a contiguous device array of at least the slab's size whose
        contents are no longer needed, holds the host-ordered copy.
        """
        for key in [key for key in self._held if key[0] < z1 and z0 < key[1]]:
            del self._held[key]
        xp = self.engine.xp
        planes = value[:, :, : z1 - z0].transpose(2, 1, 0)
        if scratch is None:
            planes = xp.ascontiguousarray(planes)
        else:
            buffer = scratch.reshape(-1)[: planes.size].reshape(planes.shape)
            buffer[...] = planes
            planes = buffer
        if self._worker is None:
            self.engine.download(planes, self.array[:, :, z0:z1].T)
            return
        for start, stop in [key for key in self._reads if key[0] < z1 and z0 < key[1]]:
            del self._reads[start, stop]  # read before this write: stale
        block = np.empty(planes.shape, np.float32)
        self.engine.download(planes, block)
        while len(self._writes) >= self._PENDING:
            self._writes.pop(0).result()
        self._writes.append(self._worker.submit(self._write, z0, z1, block))


class DeviceVolume(HostVolume):
    """Slab access to a volume kept on the device in kernel layout ``(nx, ny, nz)``."""

    def view(self, z0: int, z1: int) -> Any:
        """A writable copy of planes ``[z0, z1)``."""
        return self.array[:, :, z0:z1].copy()

    def get(self, z0: int, z1: int, *, retain: bool = False) -> Any:
        """A contiguous copy of planes ``[z0, z1)``; copies are cheap, so none is held."""
        del retain
        return self.view(z0, z1)

    def put(self, z0: int, z1: int, value: Any, scratch: Any = None) -> None:
        """Store the first ``z1 - z0`` planes of a device slab; no scratch is needed."""
        del scratch
        self.array[:, :, z0:z1] = value[:, :, : z1 - z0]


@dataclass(frozen=True)
class Batch:
    """A batch of a sweep's views: their indices on the host and on the device."""

    views: np.ndarray
    index: Any


@dataclass
class Sweep:
    """One pass over the slabs for a set of views, with the ring of detector rows.

    ``coefficients`` are the full-grid, full-detector rows of every view.
    """

    plan: SlabPlan
    engine: Engine
    coefficients: np.ndarray
    batch: int
    _slab_coefficients: dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._device_coefficients = self.engine.upload(self.coefficients)

    def slab_coefficients(self, k: int) -> Any:
        """Device coefficients of slab ``k``'s window, for every view."""
        if k not in self._slab_coefficients:
            first, _ = self.plan.windows[k]
            z0, _ = self.plan.planes(k)
            coeff = window_coefficients(self._device_coefficients, first, z0)
            self._slab_coefficients[k] = coeff
        return self._slab_coefficients[k]

    def row_coefficients(self, batch: Batch, row: int) -> Any:
        """Full-grid coefficients of a batch's views for a band starting at detector ``row``."""
        return window_coefficients(self._device_coefficients[batch.index], row, 0)

    def run(
        self,
        views: np.ndarray,
        *,
        source: Callable[[int], tuple[Any, ...]],
        finish: Callable[[Batch, int, int, tuple[Any, ...]], None],
        sink: Callable[[int, tuple[Any, ...], tuple[Any, ...]], None],
        channels: int = 1,
        transposed: int | None = None,
        ones: bool = False,
        keep: bool = True,
        prefetch: Callable[[int], None] | None = None,
    ) -> None:
        """Project ``source(k)``, finish each row once, transpose and ``sink`` every slab.

        ``source(k)`` returns device volumes of slab ``k``, projected into ring
        channels 0, 1, ...; None (a zero slab) or an empty tuple projects nothing. ``finish(batch,
        first, stop, rows)`` receives every channel's ring rows ``[first, stop)``
        for a :class:`Batch` of views and replaces them in place by the images to
        transpose. ``sink(k, accumulators, sources)`` receives slab ``k``'s
        transposes of the first ``transposed`` channels (default all; none skips
        the transposes) and, with ``ones``, of unit images (the column sums),
        then the volumes ``source(k)`` returned, kept on the device until then
        (with ``keep``; otherwise an empty tuple). ``prefetch(k)`` is called
        just before the slab projected ahead of ``k``, to start reading it.
        """
        ring = _Ring(self, views, channels, keep=keep)
        count = channels if transposed is None else transposed
        for k in self.plan.order:
            ring.project_through(self.plan.windows[k][1], source, prefetch)
            ring.finish(self.plan.windows[k], finish)
            sink(k, ring.transpose(k, count, ones=ones), ring.sources.pop(k))


class _Ring:
    """One sweep's ring of detector rows: row ``r`` lives in slot ``r % plan.ring``."""

    def __init__(self, sweep: Sweep, views: np.ndarray, channels: int, *, keep: bool) -> None:
        plan, xp = sweep.plan, sweep.engine.xp
        self.sweep, self.plan, self.xp, self.engine = sweep, plan, xp, sweep.engine
        self.rows = xp.zeros((channels, len(views), plan.ring, plan.detector.nu), xp.float32)
        self.top = self.finished = plan.windows[plan.order[0]][0]
        self.projected, self.sources, self.keep = 0, {}, keep
        index, size = xp.asarray(views), sweep.batch
        self.batches = [
            (b, min(b + size, len(views)), Batch(views[b : b + size], index[b : b + size]))
            for b in range(0, len(views), size)
        ]

    def spans(self, first: int, stop: int) -> list[tuple[int, int, int]]:
        """``(row, slot, length)`` runs of rows ``[first, stop)`` in the ring."""
        parts = []
        while first < stop:
            slot = first % self.plan.ring
            length = min(stop - first, self.plan.ring - slot)
            parts.append((first, slot, length))
            first += length
        return parts

    def project_through(
        self,
        stop: int,
        source: Callable[[int], tuple[Any, ...]],
        prefetch: Callable[[int], None] | None,
    ) -> None:
        """Project every slab whose window starts before row ``stop``, in order."""
        plan = self.plan
        while self.projected < plan.count and plan.windows[plan.order[self.projected]][0] < stop:
            if prefetch is not None and self.projected + 1 < plan.count:
                prefetch(plan.order[self.projected + 1])
            self.project(plan.order[self.projected], source)
            self.projected += 1

    def project(self, j: int, source: Callable[[int], tuple[Any, ...]]) -> None:
        first, stop = self.plan.windows[j]
        for _, slot, length in self.spans(max(self.top, first), stop):
            self.rows[:, :, slot : slot + length] = 0
        self.top = max(self.top, stop)
        coeff = self.sweep.slab_coefficients(j)
        volumes = source(j)
        self.sources[j] = volumes if self.keep else ()
        for channel, volume in enumerate(volumes):
            for b0, b1, batch in self.batches if volume is not None else ():
                values = self.engine.project(volume, coeff[batch.index], stop - first)
                for row, slot, length in self.spans(first, stop):
                    part = values[:, row - first : row - first + length]
                    self.rows[channel, b0:b1, slot : slot + length] += part

    def finish(self, window: tuple[int, int], finish: Callable[..., None]) -> None:
        """Finish the rows of ``window`` not finished yet: every slab reaching them is in."""
        first, stop = window
        for row, slot, length in self.spans(max(self.finished, first), stop):
            # a band at a time: a RowReader holds one band
            bounds = range(row // _BAND * _BAND + _BAND, row + length, _BAND)
            for start, end in zip((row, *bounds), (*bounds, row + length), strict=True):
                part = self.rows[:, :, slot + start - row : slot + end - row]
                for b0, b1, batch in self.batches:
                    finish(batch, start, end, tuple(part[:, b0:b1]))
        self.finished = max(self.finished, stop)

    def window(self, channel: int, b0: int, b1: int, first: int, stop: int) -> Any:
        """A contiguous copy of rows ``[first, stop)`` of a batch."""
        parts = [self.rows[channel, b0:b1, s : s + n] for _, s, n in self.spans(first, stop)]
        return self.xp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0].copy()

    def transpose(self, k: int, count: int, *, ones: bool) -> tuple[Any, ...]:
        """Slab ``k``'s transposes of the first ``count`` channels, then of unit images."""
        xp, plan, (first, stop) = self.xp, self.plan, self.plan.windows[k]
        z0, z1 = plan.planes(k)
        depth = plan.depth if isinstance(self.engine, JaxEngine) else z1 - z0
        shape = (plan.grid.nx, plan.grid.ny, depth)
        accumulate = tuple(xp.zeros(shape, xp.float32) for _ in range(count + ones))
        coeff = self.sweep.slab_coefficients(k)
        for b0, b1, batch in self.batches if accumulate else ():
            images = tuple(self.window(c, b0, b1, first, stop) for c in range(count))
            if ones:
                images += (xp.ones((b1 - b0, stop - first, plan.detector.nu), xp.float32),)
            accumulate = self.engine.transpose(images, coeff[batch.index], accumulate)
        return accumulate


def host_volumes(grid: Grid, volumes: int, backend: str) -> bool:
    """Whether ``volumes`` FP32 copies of ``grid`` exceed 85% of free CUDA device memory."""
    if backend != "cuda":
        return False
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    return free is not None and volumes * 4 * grid.nx * grid.ny * grid.nz > 0.85 * free


def _free_bytes() -> int:
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


def _device_need(
    plan: SlabPlan, views: int, channels: int, batch: int, engine: Engine, *, kept: int
) -> int:
    """Device bytes of a sweep.

    Its ring, the slab volumes alive at once (``kept`` projected slabs per
    channel and one being uploaded, or two transposes and the slab they
    update) and view-batch scratch.
    """
    plane, row = 4 * engine.grid.nx * engine.grid.ny, 4 * engine.detector.nu
    height = max(stop - first for first, stop in plan.windows)
    return (
        channels * views * plan.ring * row
        + max(3, 2 + channels * kept) * plane * plan.depth
        + 6 * batch * height * row
    )


def slab_depth(
    plans: Callable[[int], SlabPlan],
    views: int,
    channels: int,
    batch: int,
    engine: Engine,
    resident: int = 0,
) -> int:
    """Planes per slab: a few slabs on the JAX engine, otherwise the deepest that fits.

    The device holds ``resident`` whole volumes beside one sweep (see
    :func:`_device_need`). The ring grows with the cone's spread more than
    with the depth, so deeper slabs cost little ring and save work.
    """
    if isinstance(engine, JaxEngine):
        return max(1, -(-engine.grid.nz // 4))
    free = _free_bytes() - resident * 4 * engine.grid.nx * engine.grid.ny * engine.grid.nz
    for depth in (128, 64, 32, 16, 8, 4, 2, 1):
        if depth >= 2 * engine.grid.nz:
            continue
        if _device_need(plans(depth), views, channels, batch, engine, kept=1) <= 0.8 * free:
            return depth
    raise MemoryError(
        "the detector rows reached by one z plane do not fit on the device; "
        "use more subsets or fewer views per batch"
    )


@dataclass
class ConeProblem:
    """A cone scan's slabs, engine and host measurements for slab-swept solvers.

    With ``on_device`` the solver's volumes stay on the device; otherwise in
    host memory or in a caller's ``out``. ``keep`` tells sweeps whether the
    projected slabs fit on the device until their transposes; otherwise a
    host volume holds their copies (``get(..., retain=True)``) for re-upload.
    """

    sweep: Sweep
    data: np.ndarray
    on_device: bool = False
    keep: bool = True

    def iterate(
        self, initial: Any, out: np.ndarray | None, context: str, *, written: bool = False
    ) -> Any:
        """The solver's iterate holding ``initial`` (zero for None): ``out``, device or host.

        With ``written`` the solver writes every voxel before reading any, so
        a zero start is not written to host storage.
        """
        grid = self.plan.grid
        shape = (grid.nx, grid.ny, grid.nz)
        if out is not None or not self.on_device:
            # A row reader holds no storage that could overlap ``out``.
            data = np.empty(0, np.float32) if isinstance(self.data, RowReader) else self.data
            target = validate_host_arrays(data, out, shape, context, order="F")
            if initial is None and written:
                return target
            return fill(target, None if initial is None else np.asarray(initial))
        xp = self.engine.xp
        target = xp.zeros(shape, xp.float32)
        if initial is not None:
            for z0, z1 in map(self.plan.planes, range(self.plan.count)):
                target[:, :, z0:z1] = xp.asarray(np.asarray(initial[:, :, z0:z1], np.float32))
        return target

    def work(self, out: np.ndarray | None) -> Any:
        """A zeroed work volume beside the iterate: device, file beside ``out``, or host."""
        grid = self.plan.grid
        shape = (grid.nx, grid.ny, grid.nz)
        if self.on_device and out is None:
            return self.engine.xp.zeros(shape, self.engine.xp.float32)
        return host_volume(shape, scratch_directory(out))

    def result(self, volume: Any) -> Any:
        """The returned volume: a JAX array for a device volume, otherwise ``volume``.

        Device scratch the solve no longer uses goes back to the driver, so
        later work in the process (JAX's own allocator included) can use it,
        and a :class:`RowReader` stops reading ahead.
        """
        if isinstance(self.engine, CudaEngine):
            self.engine.xp.get_default_memory_pool().free_all_blocks()
        if isinstance(self.data, RowReader):
            self.data.close()
        return jax.dlpack.from_dlpack(volume) if self.on_device else volume

    @property
    def plan(self) -> SlabPlan:
        """The slab plan."""
        return self.sweep.plan

    @property
    def engine(self) -> Engine:
        """The device engine."""
        return self.sweep.engine

    def volume(self, array: Any) -> HostVolume:
        """Slab access to a host or device volume."""
        kind = HostVolume if isinstance(array, np.ndarray) else DeviceVolume
        return kind(array, self.engine, self.plan.depth)

    def measured(self, batch: Batch, first: int, stop: int) -> Any:
        """Measured rows ``[first, stop)`` of a batch's views on the device."""
        views, data = batch.views, self.data
        if isinstance(data, RowReader):
            data, first, stop = data.band(first, stop)
        v0, v1 = int(views[0]), int(views[-1]) + 1
        if v1 - v0 == len(views) and np.all(np.diff(views) == 1):
            rows = np.ascontiguousarray(data[v0:v1, first:stop], np.float32)
        else:
            rows = np.ascontiguousarray(data[views, first:stop], np.float32)
        release(data)
        return self.engine.upload(rows)


def cone_problem(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    data: np.ndarray,
    *,
    model: ConeModel,
    backend: str,
    poses: jax.Array,
    batch: int,
    channels: int,
    views: int | None = None,
    resident: int = 0,
) -> ConeProblem:
    """Plan slabs for sweeps of ``views`` views (default all) holding ``channels`` ring channels.

    The solver's ``resident`` volumes stay on the device when they fit beside
    the sweep (see :attr:`ConeProblem.on_device`).
    """
    from tomojax.core.geometry.views import host_view_poses

    count = int(poses.shape[0])
    coefficients = np.asarray(model.coefficients(poses, grid), np.float32)
    host_poses = host_view_poses(geometry, count, dtype=np.float64)
    host_poses = np.asarray(poses if host_poses is None else host_poses, np.float64)
    engine = engine_for(grid, detector, backend)
    batch = min(batch, count)
    frames = model.frames(count)

    @cache
    def plans(depth: int) -> SlabPlan:
        return plan_slabs(
            frames, host_poses, grid, detector, depth, common_height=isinstance(engine, JaxEngine)
        )

    sweeps = count if views is None else views
    on_device, keep = False, isinstance(engine, JaxEngine)
    if resident and isinstance(engine, CudaEngine):
        try:
            depth = slab_depth(plans, sweeps, channels, batch, engine, resident)
            on_device = depth >= min(64, grid.nz)  # thin slabs would cost more than transfers
        except MemoryError:
            pass
    if not on_device:
        depth = slab_depth(plans, sweeps, channels, batch, engine)
        if isinstance(engine, CudaEngine):
            plan = plans(depth)
            need = _device_need(plan, sweeps, channels, batch, engine, kept=plan.lag + 1)
            keep = need <= 0.8 * _free_bytes()
    sweep = Sweep(plans(depth), engine, coefficients, batch)
    return ConeProblem(sweep, data, on_device, keep)
