"""Views shared among several devices, each holding the whole volume.

Each device projects its share of the views and backprojects them into its own
volume, and the volumes are summed, so the transpose stays exact; only the order
of that sum differs from one device. So that the shares are equal, the views are
padded with copies of the last one that count for nothing. Compiled solvers run
the shares in one ``shard_map``; FBP and FDK, host-driven loops, run each share
in a thread of its own (:meth:`ViewSplit.summed`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np

if TYPE_CHECKING:
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

    def place(self, stack: jax.Array | np.ndarray, *, repeat: bool = False) -> jax.Array:
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
            part = xp.asarray(stack[start : min(stop, self.n)], xp.float32)
            missing = stop - start - int(part.shape[0])
            if not missing:
                return part
            if repeat:
                last = xp.asarray(stack[self.n - 1 : self.n], xp.float32)
                filler = xp.repeat(last, missing, axis=0)
            else:
                filler = xp.zeros((missing, *stack.shape[1:]), xp.float32)
            return xp.concatenate([part, filler])

        sharding = NamedSharding(self.mesh, PartitionSpec(VIEWS))
        return jax.make_array_from_callback((self.total, *stack.shape[1:]), sharding, views)

    def everywhere[T](self, arrays: T) -> T:
        """``arrays`` (an array or a pytree of them) whole on every device, wherever they were."""
        return jax.device_put(arrays, NamedSharding(self.mesh, PartitionSpec()))

    def take(self, stack: jax.Array) -> jax.Array:
        """The ``n`` real views of a :meth:`place`-d stack, on :attr:`first`."""
        return jax.device_put(stack[: self.n], self.first)

    def gather(self, volume: jax.Array) -> jax.Array:
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

    def summed(self, run: Callable[[Device, range], jax.Array]) -> jax.Array:
        """The sum, on :attr:`first`, of ``run(device, views)`` over :meth:`shares`.

        Each device's ``run`` is a thread of its own, with that device JAX's
        default, so host-driven loops (batches streamed from the host, CUDA
        launches) keep every device busy at once.
        """

        def on(share: tuple[Device, range]) -> jax.Array:
            device, views = share
            with jax.default_device(device):
                return run(device, views)

        shares = self.shares()
        with ThreadPoolExecutor(len(shares)) as pool:
            parts = list(pool.map(on, shares))
        total = parts[0]
        for part in parts[1:]:
            total = total + jax.device_put(part, self.first)
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


def as_devices(devices: Device | Sequence[Device] | None) -> tuple[Device, ...] | None:
    """``devices`` (one device, several, or None) as a tuple, checked.

    Raises for no devices, a device given twice or devices of different platforms.
    """
    if devices is None:
        return None
    devices = tuple(devices) if isinstance(devices, Sequence) else (devices,)
    if not devices:
        raise ValueError("devices must name at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("devices must not name a device twice")
    if len({d.platform for d in devices}) > 1:
        raise ValueError("devices must all be of one platform (all GPUs, say)")
    return devices


def view_split(devices: Device | Sequence[Device] | None, n_views: int) -> ViewSplit | None:
    """The split of ``n_views`` among ``devices`` (see :func:`as_devices`), None for None."""
    devices = as_devices(devices)
    if devices is None:
        return None
    return ViewSplit(Mesh(np.array(devices), (VIEWS,)), int(n_views))


def refuse_streaming(
    split: ViewSplit | None, stream_projections: bool | None, context: str
) -> None:
    """Raise if projections shared among devices are asked to stream: each holds its share."""
    if split is not None and stream_projections:
        raise ValueError(f"{context}: projections shared among devices cannot be streamed")


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
    "pad_views",
    "refuse_streaming",
    "shared_devices",
    "sharing",
    "view_split",
]
