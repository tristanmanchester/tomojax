"""Views shared among several devices, each holding the whole volume.

Each device projects its share of the views and backprojects them into its own
volume, and the volumes are summed, so the transpose stays exact; only the order
of that sum differs from one device. So that the shares are equal, the views are
padded with copies of the last one that count for nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    def place(self, stack: jax.Array | np.ndarray) -> jax.Array:
        """``stack`` of ``n`` views, padded and laid out with each device's views on it.

        Each device receives only its own views, read from the host or copied from
        the device holding ``stack``; no whole padded copy is made anywhere.
        """
        xp = jnp if isinstance(stack, jax.Array) else np

        def views(index: tuple[slice, ...] | None) -> np.ndarray | jax.Array:
            assert index is not None
            start, stop, _ = index[0].indices(self.total)
            part = xp.asarray(stack[start : min(stop, self.n)], xp.float32)
            missing = stop - start - int(part.shape[0])
            return xp.pad(part, [(0, missing)] + [(0, 0)] * (part.ndim - 1)) if missing else part

        sharding = NamedSharding(self.mesh, PartitionSpec(VIEWS))
        return jax.make_array_from_callback((self.total, *stack.shape[1:]), sharding, views)

    def take(self, stack: jax.Array) -> jax.Array:
        """The ``n`` real views of a :meth:`place`-d stack, on :attr:`first`."""
        return jax.device_put(stack[: self.n], self.first)

    def gather(self, volume: jax.Array) -> jax.Array:
        """A volume every device holds, on :attr:`first`."""
        return jax.device_put(volume, self.first)


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


__all__ = ["VIEWS", "ViewSplit", "as_devices", "pad_views", "refuse_streaming", "view_split"]
