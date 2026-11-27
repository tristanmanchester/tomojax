"""Views shared among several devices, each holding the whole volume.

Each device projects its share of the views and backprojects them into its own
volume, and the volumes are summed, so the transpose stays exact; only the order
of that sum differs from one device. So that the shares are equal, the views are
padded with copies of the last one that count for nothing. Compiled solvers run
the shares in one ``shard_map``; FBP and FDK, host-driven loops, run each share
in a thread of its own (:func:`sum_batches`).

Whether one block of a size can be had on the GPU, from the allocator's figures or by
allocating it, is answered here once for every layer (:func:`block_fits`,
:func:`block_available`).
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import jax
from jax.core import Tracer
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

from tomojax import progress
from tomojax._typed_arrays import device_id
from tomojax.errors import InvalidParameter, device_out_of_memory

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from tomojax._typed_arrays import Device

VIEWS = "views"  # the mesh axis the views are split along


@dataclass(frozen=True)
class ViewSplit:
    """``n`` views shared evenly among the devices of ``mesh``.

    Sinogram-sized arrays of a split have :attr:`total` views, the padding views
    zero: :meth:`place` lays one out and :meth:`take` returns its ``n`` views.
    """

    mesh: Mesh
    n: int

    @property
    def total(self) -> int:
        """The views padded to a multiple of the device count."""
        return -(-self.n // self.mesh.size) * self.mesh.size

    @property
    def first(self) -> Device:
        """The device results come back on, as from a one-device call."""
        return self.mesh.devices.flat[0]

    def counted(self) -> jax.Array:
        """Which of the :attr:`total` views count: all but the padding."""
        return jnp.arange(self.total) < self.n

    def place(
        self,
        stack: jax.Array | np.ndarray,
        *,
        repeat: bool = False,
        dtype: np.dtype | type[np.generic] = np.float32,
    ) -> jax.Array:
        """``stack`` of ``n`` views, padded and laid out with each device's views on it.

        The padding views are zeros, or with ``repeat`` copies of the last view
        (poses, say, which must stay valid). Each device receives only its own
        views, read from the host or copied from the device holding ``stack``;
        no whole padded copy is made anywhere.
        """
        xp = jnp if isinstance(stack, jax.Array) else np

        def views(index: tuple[slice, ...] | None) -> np.ndarray | jax.Array:
            assert index is not None
            start, stop, _ = index[0].indices(self.total)
            part = xp.asarray(stack[start : min(stop, self.n)], dtype)
            missing = stop - start - int(part.shape[0])
            if not missing:
                return part
            if repeat:
                last = xp.asarray(stack[self.n - 1 : self.n], dtype)
                filler = xp.repeat(last, missing, axis=0)
            else:
                filler = xp.zeros((missing, *stack.shape[1:]), dtype)
            return xp.concatenate([part, filler])

        sharding = NamedSharding(self.mesh, PartitionSpec(VIEWS))
        return jax.make_array_from_callback((self.total, *stack.shape[1:]), sharding, views)

    def everywhere[T](self, arrays: T) -> T:
        """``arrays`` (an array or a pytree of them) whole on every device, wherever they were."""
        return jax.device_put(arrays, NamedSharding(self.mesh, PartitionSpec()))

    def take(self, stack: jax.Array | np.ndarray) -> jax.Array:
        """The ``n`` real views of a :meth:`place`-d stack, on :attr:`first`."""
        return jax.device_put(stack[: self.n], self.first)

    def gather(self, volume: ArrayLike) -> jax.Array:
        """A volume every device holds, on :attr:`first`."""
        return jax.device_put(volume, self.first)

    def shares(self) -> list[tuple[Device, range]]:
        """Each device with views to handle, and those views: the ones :meth:`place` gives it."""
        per = self.total // self.mesh.size
        devices = list(self.mesh.devices.flat)
        return [
            (device, range(k * per, min(self.n, (k + 1) * per)))
            for k, device in enumerate(devices)
            if k * per < self.n
        ]


type BatchRanges = Callable[[int], Iterator[range]]


def sum_batches(
    split: ViewSplit | None,
    views: int,
    batch_size: int,
    partials: Callable[[Device | None, range, BatchRanges], Generator[jax.Array]],
) -> jax.Array:
    """Drive independent device accumulators to the same work boundary, then report on the owner.

    A share keeps its original view range, numerical batch and summation order.
    If a share retries with smaller batches, it replays to the current view
    boundary before it can yield another partial volume.
    """
    shares: list[tuple[Device | None, range]] = [(None, range(views))]
    if split is not None:
        placed = split.shares()
        announce(tuple(device for device, _ in placed))
        shares = list(placed)
    counts = [math.ceil(len(share) / batch_size) for _, share in shares]
    task = progress.for_unit("batches")
    end = max(counts)
    runtime = progress.current_runtime.get()

    def ranges(length: int) -> BatchRanges:
        def groups(size: int) -> Iterator[range]:
            first, total = 0, math.ceil(length / size)
            while first < total:
                stop = min(total, math.ceil(end * batch_size / size))
                yield range(first, stop)
                first = stop

        return groups

    active = [partials(device, share, ranges(len(share))) for device, share in shares]
    volumes: dict[int, jax.Array] = {}

    def dispatch(index: int) -> tuple[int, jax.Array]:
        device = shares[index][0]
        token = progress.current_runtime.set(runtime)
        try:
            with jax.default_device(device) if device is not None else nullcontext():
                return index, next(active[index])
        finally:
            progress.current_runtime.reset(token)

    with ExitStack() as jobs:
        for accumulator in active:
            _ = jobs.callback(accumulator.close)
        pool = jobs.enter_context(ThreadPoolExecutor(len(shares))) if split is not None else None
        for first, end in progress.chunks(max(counts)) if task is not None else [(0, max(counts))]:
            indices = [index for index, count in enumerate(counts) if count > first]
            with progress.preparing("dispatch backprojection", first):
                results = map(dispatch, indices) if pool is None else pool.map(dispatch, indices)
                for index, volume in results:
                    if counts[index] <= end:
                        volumes[index] = volume
                    del volume
            if task is not None:
                task.total = sum(counts)
                task.advance(sum(min(count, end) for count in counts), {})
    total = volumes[0]
    for index in range(1, len(shares)):
        total = total + jax.device_put(volumes[index], shares[0][0])
    return total


# The devices a run's solvers share their views among, set by :func:`sharing`.
_SHARING: ContextVar[tuple[Device, ...] | None] = ContextVar("tomojax_devices", default=None)


@contextmanager
def sharing(devices: Device | Sequence[Device] | None) -> Iterator[None]:
    """Within, solvers that read :func:`shared_devices` share their views among ``devices``.

    Alignment's solvers are many calls below :func:`tomojax.align`; as with
    ``jax.default_device``, the devices are a context of the run, not a setting
    that changes its result or its checkpoints.
    """
    token = _SHARING.set(as_devices(devices))
    try:
        yield
    finally:
        _SHARING.reset(token)


def shared_devices() -> tuple[Device, ...] | None:
    """The devices :func:`sharing` names, None outside it."""
    return _SHARING.get()


def current_device() -> Device:
    """The local dispatch device selected by JAX's default-device context."""
    selected = jax.config.jax_default_device
    if selected is None or isinstance(selected, str):
        return next(iter(jax.local_devices(backend=selected)))
    return selected


def announce(devices: Iterable[Device] | None = None, *, array: object = None) -> None:
    """Announce concrete placement on the coordinator, including cache hits."""
    if not progress.tracking_placement() or isinstance(array, Tracer):
        return
    if devices is None:
        devices = (
            sorted(array.devices(), key=device_id)
            if isinstance(array, jax.Array) and array.committed
            else (current_device(),)
        )
    progress.placed(tuple(map(device_id, devices)))


def as_devices(devices: Device | Sequence[Device] | None) -> tuple[Device, ...] | None:
    """``devices`` (one device, several, or None) as a tuple, checked.

    Raises for no devices, a device given twice or devices of different platforms.
    """
    if devices is None:
        return None
    devices = tuple(devices) if isinstance(devices, Sequence) else (devices,)
    if not devices:
        raise InvalidParameter("devices", "devices must name at least one device")
    if len(set(devices)) != len(devices):
        raise InvalidParameter("devices", "devices must not name a device twice")
    if len({d.platform for d in devices}) > 1:
        raise InvalidParameter("devices", "devices must all be of one platform (all GPUs, say)")
    return devices


def view_split(devices: Device | Sequence[Device] | None, n_views: int) -> ViewSplit | None:
    """The split of ``n_views`` among ``devices`` (see :func:`as_devices`), None for None."""
    devices = as_devices(devices)
    if devices is None:
        return None
    return ViewSplit(Mesh(np.array(devices), (VIEWS,)), int(n_views))


def refuse_streaming(
    devices: Device | Sequence[Device] | None, stream: bool | None, context: str
) -> None:
    """Raise if projections shared among devices are asked to stream: each holds its share."""
    if devices is not None and stream:
        raise InvalidParameter(
            "stream", f"{context}: projections shared among devices cannot be streamed"
        )


def pad_views(array: jax.Array, total: int, *, repeat: bool = False) -> jax.Array:
    """``array`` padded to ``total`` views: with zeros, or repeating its last view."""
    extra = total - int(array.shape[0])
    if extra == 0:
        return array
    widths = [(0, extra)] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, widths, mode="edge" if repeat else "constant")


__all__ = [
    "VIEWS",
    "ViewSplit",
    "as_devices",
    "block_available",
    "block_fits",
    "pad_views",
    "pool_stats",
    "refuse_streaming",
    "shared_devices",
    "sharing",
    "view_split",
]


def pool_stats() -> dict[str, int] | None:
    """The default GPU allocator's statistics; None off a GPU or without them."""
    if jax.default_backend() != "gpu":
        return None
    return jax.local_devices()[0].memory_stats()


def block_fits(size: int, stats: Mapping[str, int] | None) -> bool:
    """Whether a pool with allocator figures ``stats`` could hand out one block of ``size`` bytes.

    This uses the largest free block and the room to grow under the pool limit.
    The snapshot excludes space the allocator might reclaim by releasing regions.
    ``None`` (off a GPU) always fits.
    """
    if not stats:
        return True
    return size <= max(
        stats["largest_free_block_bytes"], stats["bytes_limit"] - stats["pool_bytes"]
    )


def block_available(size: int) -> bool:
    """Whether one block of ``size`` bytes can be had now, confirmed by allocating it.

    A block the pool's figures rule out (:func:`block_fits`) is refused without an
    allocation, so a certain no costs neither the allocator's retry nor its report. Otherwise
    the block is allocated and freed: its region stays in the pool, free, for the arrays that
    follow.
    """
    if not block_fits(size, pool_stats()):
        return False
    if jax.default_backend() != "gpu":
        return True
    try:
        block = jnp.zeros((-(-size // 4),), jnp.float32)
        _ = block.block_until_ready()
    except Exception as exc:
        if device_out_of_memory(exc):
            return False
        raise
    block.delete()
    return True
