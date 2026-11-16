"""The level's stack-sized arrays, each on the device or in a host buffer for the level's life.

A level holds up to six arrays with one image per view: the data ``y`` and its
``precision`` (when not uniform), the linearisation's ``residual``, IRLS weights
``sw`` and pose columns ``j`` (FP16), and ``r``, the weighted projection that
the Schur matvec and then the step's line search use in turn (they never
coexist, so one buffer serves both). Together they are what a large scan cannot
fit on the device; everything else in a level is volume-sized.

At level setup a plan is made from the device's total room under its limit,
less the level's other arrays, keeping resident the stacks that most deserve the
device (read in every matvec first) and putting the rest on the host. The plan
is a prediction: the total room is an upper bound, blind to fragmentation, and
the allocator's statistics cannot say more (the pool returns wholly free regions
to the driver and retries). So the resident stacks are then really allocated, in
the same order, each waited for; the first that fails goes to the host with every
stack after it, and the level records the misprediction. Little is in use at
setup, which keeps the plan close to exact; releasing every buffer at the level's
end keeps that true for the next level.

Compiled functions take :class:`Stack` handles and read and write a batch at a
time. A handle's placement is static, so a function compiles once per placement:
the resident path is a slice of a device array and costs nothing extra; only a
stack held on the host pays the transfer (``read_block`` / ``write_block``).
Written stacks are donated, so a level rewrites its buffers in place instead of
allocating fresh stacks at every linearisation and matvec.
"""

from __future__ import annotations

from contextlib import ExitStack
import math
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source, read_block, write_block
from tomojax.core.host_volume import host_array

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

Place = Literal["device", "host"]

# Who most deserves the device: read in every matvec (sw, r, j), then the data
# (each linearisation and line-search trial), then the residual (the trials at
# unchanged poses only).
ORDER = ("sw", "r", "j", "y", "precision", "residual")


@jax.tree_util.register_pytree_node_class
class Stack:
    """One stack-sized array, views first: a device array, or the key of a registered host buffer.

    The placement, shape, dtype and batch size are static.
    """

    def __init__(
        self, data: Any, *, host: bool, shape: tuple[int, ...], dtype: Any, batch: int
    ) -> None:
        self.data, self.host, self.shape, self.batch = data, host, tuple(shape), int(batch)
        self.dtype = np.dtype(dtype)

    def tree_flatten(self):
        return (self.data,), (self.host, self.shape, self.dtype, self.batch)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: tuple) -> Stack:
        host, shape, dtype, batch = aux
        return cls(children[0], host=host, shape=shape, dtype=dtype, batch=batch)

    def read(self, i: jax.Array) -> jax.Array:
        """Batch ``i`` (``batch`` views) on the device.

        Behind an optimization barrier whatever the placement: a slice of a device
        array would otherwise fuse into its consumers and a host read could not, and
        the FP64 reductions downstream (scores, moments) would round differently by
        placement. With the barrier every consumer compiles the same, bit for bit.
        """
        start = i * self.batch
        if self.host:
            block = read_block(self.data, start, (self.batch, *self.shape[1:]), self.dtype)
        else:
            block = jax.lax.dynamic_slice_in_dim(self.data, start, self.batch, 0)
        return jax.lax.optimization_barrier(block)

    def write(self, i: jax.Array, value: jax.Array) -> Stack:
        """The stack with batch ``i`` replaced by ``value``."""
        value = value.astype(self.dtype)
        if self.host:
            write_block(self.data, i * self.batch, value)
            return self
        data = jax.lax.dynamic_update_slice_in_dim(self.data, value, i * self.batch, 0)
        return Stack(data, host=False, shape=self.shape, dtype=self.dtype, batch=self.batch)


def _exhausted(err: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(err)


class LevelStacks:
    """A level's stacks: planned, allocated in device-priority order, and released at its end.

    ``y`` and ``precision`` are the level's padded host data (``precision`` None
    when uniform); ``names`` the stacks the level uses; ``room`` the device's total
    room and ``reserve`` the bytes the level needs besides its stacks. ``placement``
    forces named stacks to "device" or "host" (the tests' and, until the public
    setting, the only override); a forced stack is still really allocated.
    """

    def __init__(
        self,
        specs: Mapping[str, tuple[tuple[int, ...], Any]],
        batch: int,
        *,
        data: Mapping[str, np.ndarray],
        room: int | None,
        reserve: int,
        placement: Mapping[str, Place] | None = None,
    ) -> None:
        self.names = [name for name in ORDER if name in specs]
        self.batch = batch
        self.specs = {
            name: (tuple(specs[name][0]), np.dtype(specs[name][1])) for name in self.names
        }
        self.sizes = {n: math.prod(s) * d.itemsize for n, (s, d) in self.specs.items()}
        self.planned = self._plan(room, reserve, placement or {})
        self.room, self.reserve = room, reserve
        self.stacks: dict[str, Stack] = {}
        self.placed: dict[str, Place] = {}
        self._host: dict[str, np.ndarray] = {}
        self._keys = ExitStack()
        try:
            self._allocate(data)
        except BaseException:
            self.close()
            raise

    def _plan(self, room, reserve, forced) -> dict[str, Place]:
        unknown = set(forced) - set(ORDER)
        if unknown or any(place not in ("device", "host") for place in forced.values()):
            raise ValueError(f"placement takes {ORDER} mapped to 'device' or 'host', not {forced}")
        budget = math.inf if room is None else room - reserve
        plan, used, spilled = {}, 0, False
        for name in self.names:
            if name in forced:
                plan[name] = forced[name]
            elif not spilled and used + self.sizes[name] <= budget:
                plan[name] = "device"
            else:
                plan[name], spilled = "host", True
            used += self.sizes[name] if plan[name] == "device" else 0
        return plan

    def _allocate(self, data: Mapping[str, np.ndarray]) -> None:
        failed = False
        for name in self.names:
            shape, dtype = self.specs[name]
            if self.planned[name] == "device" and not failed:
                try:
                    array = self._device(shape, dtype, data.get(name))
                except jax.errors.JaxRuntimeError as err:
                    if not _exhausted(err):
                        raise
                    failed = True  # this stack and every one after it go to the host
                else:
                    self.stacks[name] = Stack(
                        array, host=False, shape=shape, dtype=dtype, batch=self.batch
                    )
                    self.placed[name] = "device"
                    continue
            # The host stacks are held together, so the storage rule counts them together.
            beside = sum(self.sizes[n] for n in self.names if n not in self.placed)
            array = host_array(shape, dtype=dtype, copies=beside / max(1, self.sizes[name]))
            if name in data:
                array[...] = data[name]
            self._host[name] = array
            key = self._keys.enter_context(host_source(array))
            self.stacks[name] = Stack(key, host=True, shape=shape, dtype=dtype, batch=self.batch)
            self.placed[name] = "host"

    @staticmethod
    def _device(shape: tuple[int, ...], dtype: Any, value: np.ndarray | None) -> jax.Array:
        """Allocate one stack on the device and wait for it: the allocator is the judge."""
        array = jnp.zeros(shape, dtype) if value is None else jax.device_put(value)
        return array.block_until_ready()

    def __getitem__(self, name: str) -> Stack:
        return self.stacks[name]

    def get(self, name: str) -> Stack | None:
        return self.stacks.get(name)

    def update(self, stacks: Iterable[tuple[str, Stack]]) -> None:
        """Hold the stacks a donating call returned in place of the ones it consumed."""
        for name, stack in stacks:
            self.stacks[name] = stack

    def rows(self, name: str, start: int, stop: int) -> jax.Array:
        """Views ``start`` to ``stop`` of a stack, on the device, outside compiled code."""
        if self.placed[name] == "host":
            return jnp.asarray(self._host[name][start:stop])
        return self.stacks[name].data[start:stop]

    def record(self) -> dict[str, Any]:
        """What ``info`` reports: each stack's placement and size, and the plan it came from."""
        return {
            "placement": dict(self.placed),
            "planned": dict(self.planned),
            "mispredicted": [n for n in self.names if self.planned[n] != self.placed[n]],
            "stack_bytes": dict(self.sizes),
            "room_bytes": self.room,
            "reserve_bytes": self.reserve,
        }

    def close(self) -> None:
        """Release every buffer now, so the next level starts with nothing of this one in use."""
        for stack in self.stacks.values():
            if not stack.host and not stack.data.is_deleted():
                stack.data.delete()
        self.stacks.clear()
        self._keys.close()
        self._host.clear()


def device_room() -> int | None:
    """The device's total room under its limit (an upper bound); None off the GPU."""
    from tomojax.backends import device_free_memory_bytes

    return device_free_memory_bytes() if jax.default_backend() == "gpu" else None


def level_reserve(ops: Any) -> int:
    """Device bytes a level needs besides its stacks, from its shapes: 25 volumes and the batches.

    Nothing a level compiles keeps a temporary that grows with the views (tested at two view
    counts), so the rest is volumes and per-batch terms. Measured at the peak beside the
    stacks: 24.5 volumes on a 192 x 192 x 128 level with 480 views, about 20 on K11 54013's
    fine level (2f735f89): the level's state between solves (the volume and the previous one,
    the linearisation's gradient and diagonal, the damping, the stepped gradient, TV's
    conductance, the free-voxel mask, the views' two shared images, the Schur right-hand side
    and preconditioner, the clipped step) and the Schur solve's temporaries (conjugate
    gradients' vectors, their FP64 dot-product copies, the shared images in FP64, the
    matvec's product and TV term). Then one batch's projections and pose columns at FP32, four
    times over, for the scans' per-batch temporaries.
    """
    d, (nz, ny, nx) = ops.detector, ops.shape
    volume = 4 * nz * ny * nx
    batch = 4 * ops.batch * (1 + ops.dofs) * d.nv * d.nu
    return 25 * volume + 4 * batch


def level_stacks(
    ops: Any,
    y: np.ndarray,
    precision: np.ndarray | None,
    *,
    names: Iterable[str] | None = None,
    placement: Mapping[str, Place] | None = None,
) -> LevelStacks:
    """Place and allocate a level's stacks (``names``; all of them by default) for its data."""
    specs = ops.stack_specs(precision)
    if names is not None:
        specs = {name: specs[name] for name in names if name in specs}
    data = {"y": ops.pad_images(y)}
    if precision is not None:
        data["precision"] = ops.pad_images(precision)
    return LevelStacks(
        specs,
        ops.batch,
        data={name: value for name, value in data.items() if name in specs},
        room=device_room(),
        reserve=level_reserve(ops),
        placement=placement,
    )


def unplaced(err: Exception, stacks: LevelStacks) -> Exception:
    """A plain error for a level that ran out of device memory after its stacks were placed."""
    if not _exhausted(err):
        return err
    gib = 2**30
    held = ", ".join(f"{n} on the {p}" for n, p in stacks.placed.items())
    room = "unknown" if stacks.room is None else f"{stacks.room / gib:.2f} GiB"
    return MemoryError(
        f"tj.align ran out of device memory during a level, after placing its stacks ({held}; "
        f"room at setup {room}, {stacks.reserve / gib:.2f} GiB reserved for the level's other "
        "arrays). Free device memory held elsewhere in this process, or align a smaller "
        "detector region or fewer views."
    )
