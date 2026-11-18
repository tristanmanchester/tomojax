"""A level's storage: its stacks, each on the device or in a host buffer, and its volumes.

A level holds up to six arrays with one image per view: the data ``y`` and its
``precision`` (when not uniform), the linearisation's ``residual``, IRLS weights
``sw`` and pose columns ``j`` (FP16), and ``r``, the weighted projection that
the Schur matvec and then the step's line search use in turn (they never
coexist, so one buffer serves both). Together they are what a large scan cannot
fit on the device; everything else in a level is volume-sized.

The level's volume-sized arrays (the volume, its gradient and diagonal, the step
and the rest, and the motion prior's factor) are its *volumes*: allocated on the
device first, once, and rewritten in place by donating calls for the level's
life, so nothing volume-sized is allocated again during a step. What the device
reports in use above them then (the allocator's rounding of their chunks) is
measured, and counted with them.

Then a plan is made from the device's total room under its limit, less the
volumes and a reserve kept free for the level's largest transient, keeping
resident the stacks that most deserve the device (read in every matvec first)
and putting the rest on the host. The plan is a prediction: the total room is an
upper bound, blind to fragmentation, and the allocator can hand a stack a whole
chunk of up to twice its size. So the resident stacks are then really allocated,
in the same order, each waited for; the first that fails, or leaves less than the
reserve free, goes to the host with every stack after it, and the level records
the misprediction. Little is in use at setup, which keeps the plan close to
exact; releasing every buffer at the level's end keeps that true for the next level.

Compiled functions take :class:`Stack` handles and read and write a batch at a
time. A handle's placement is static, so a function compiles once per placement:
the resident path is a slice of a device array and costs nothing extra; only a
stack held on the host pays the transfer (``read_block`` / ``write_block``).
Written stacks are donated, so a level rewrites its buffers in place instead of
allocating fresh stacks at every linearisation and matvec.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import math
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source, read_block, write_block
from tomojax.core.host_volume import host_array

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def _in_use() -> int | None:
    """Bytes the device's allocator has handed out; None off the GPU."""
    if jax.default_backend() != "gpu":
        return None
    return int(jax.local_devices()[0].memory_stats()["bytes_in_use"])


class LevelStacks:
    """A level's storage: its volumes, then its stacks, planned and allocated, released at its end.

    ``volumes`` names the level's volume-sized arrays (each a ``jax.ShapeDtypeStruct`` or a
    pytree of them), allocated first and always on the device. ``data`` holds the
    level's padded host stacks (``y``, and ``precision`` when not uniform); ``specs`` the
    stacks the level uses; ``room`` the device's total room before the volumes and
    ``reserve`` the bytes that must stay free beside the level's storage, by part. The
    volumes and what the device holds above them once allocated are the plan's ``state``.
    A stack the plan puts on the device goes to the host with every stack after it if its
    allocation fails or leaves less than the reserve free: like the volumes, a stack can
    take a whole chunk of up to twice its size, so what it leaves is measured, not assumed.
    ``placement`` forces named stacks to "device" or "host" (the tests' and, until the
    public setting, the only override); a forced stack is still really allocated.
    """

    def __init__(
        self,
        specs: Mapping[str, tuple[tuple[int, ...], Any]],
        batch: int,
        *,
        data: Mapping[str, np.ndarray],
        room: int | None,
        reserve: int | Mapping[str, int],
        volumes: Mapping[str, Any] | None = None,
        placement: Mapping[str, Place] | None = None,
    ) -> None:
        self.names = [name for name in ORDER if name in specs]
        self.batch = batch
        self.specs = {
            name: (tuple(specs[name][0]), np.dtype(specs[name][1])) for name in self.names
        }
        self.sizes = {n: math.prod(s) * d.itemsize for n, (s, d) in self.specs.items()}
        self.reserve_parts = dict(reserve) if isinstance(reserve, Mapping) else {"total": reserve}
        self.free = sum(self.reserve_parts.values())
        self.forced = dict(placement or {})
        self.volumes: dict[str, Any] = {}
        self.volume_bytes: dict[str, int] = {}
        self.stacks: dict[str, Stack] = {}
        self.placed: dict[str, Place] = {}
        self._host: dict[str, np.ndarray] = {}
        self._keys = ExitStack()
        self.room, self.setup_waste = room, 0
        try:
            if volumes:
                self._own(volumes)
            self.reserve = sum(self.reserve_parts.values())
            self.planned = self._plan(room, self.reserve, self.forced)
            self._allocate(data)
            self.in_use = _in_use()
        except BaseException:
            self.close()
            raise

    def _own(self, volumes: Mapping[str, Any]) -> None:
        """Allocate the volumes, and count them with what the device holds above them."""
        before = _in_use()
        for name, spec in volumes.items():
            self.volumes[name] = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), spec)
            leaves = jax.tree.leaves(spec)
            self.volume_bytes[name] = sum(math.prod(a.shape) * a.dtype.itemsize for a in leaves)
        jax.block_until_ready(self.volumes)
        after = _in_use()
        held = sum(self.volume_bytes.values())
        # The allocator hands out a free chunk whole when it is under twice the request:
        # the volumes' chunks can hold more than the volumes. Measured here, once.
        self.setup_waste = 0 if before is None or after is None else after - before - held
        self.reserve_parts = {"state": held + self.setup_waste, **self.reserve_parts}

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
                array = self._fitting(name, shape, dtype, data.get(name))
                if array is None:
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

    def _fitting(self, name, shape, dtype, value) -> jax.Array | None:
        """The stack on the device, or None if it fails or leaves less than the reserve free."""
        try:
            array = self._device(shape, dtype, value)
        except jax.errors.JaxRuntimeError as err:
            if not _exhausted(err):
                raise
            return None
        room = device_room()
        if name in self.forced or room is None or room >= self.free:
            return array
        array.delete()
        return None

    @staticmethod
    def _device(shape: tuple[int, ...], dtype: Any, value: np.ndarray | None) -> jax.Array:
        """Allocate one stack on the device and wait for it: the allocator is the judge."""
        array = jnp.zeros(shape, dtype) if value is None else jax.device_put(value)
        return array.block_until_ready()

    def __getitem__(self, name: str) -> Any:
        return self.stacks[name] if name in self.stacks else self.volumes[name]

    def get(self, name: str) -> Stack | None:
        return self.stacks.get(name)

    def update(self, arrays: Iterable[tuple[str, Any]]) -> None:
        """Hold the stacks or volumes a donating call returned in place of the ones it consumed."""
        for name, array in arrays:
            (self.volumes if name in self.volumes else self.stacks)[name] = array

    def take(self, name: str) -> Any:
        """Hand a volume to the caller: it is no longer the level's, and outlives it."""
        return self.volumes.pop(name)

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
            "reserve_parts": dict(self.reserve_parts),
            "volume_bytes": dict(self.volume_bytes),
            "setup_waste_bytes": self.setup_waste,
            "in_use_bytes": self.in_use,
            "free_bytes": self.free,
        }

    def close(self) -> None:
        """Release every buffer now, so the next level starts with nothing of this one in use."""
        arrays = [stack.data for stack in self.stacks.values() if not stack.host]
        for array in [*arrays, *jax.tree.leaves(self.volumes)]:
            if not array.is_deleted():
                array.delete()
        self.stacks.clear()
        self.volumes.clear()
        self._keys.close()
        self._host.clear()


def temporaries(compiled: Any, *args: Any) -> int:
    """The temporary bytes of ``compiled`` (a jitted function) at ``args``, from its compile.

    The compile is the one its calls with arguments of the same shapes then reuse.
    """
    analysis = compiled.lower(*args).compile().memory_analysis()
    if analysis is None:
        raise RuntimeError(f"no memory analysis on {jax.default_backend()}: cannot plan a level")
    return int(analysis.temp_size_in_bytes)


def device_room() -> int | None:
    """The device's total room under its limit (an upper bound); None off the GPU."""
    from tomojax.backends import device_free_memory_bytes

    return device_free_memory_bytes() if jax.default_backend() == "gpu" else None


def level_reserve(ops: Any, prior: Any) -> dict[str, int]:
    """Device bytes the pose refinement reserves besides its stacks, in parts, from its shapes.

    These are the counts the solver's levels planned with before they owned their volumes
    (``_solver.level_volumes``); the refinement holds no level state of its own, so they
    over-reserve for it.

    Nothing a level compiles keeps a temporary that grows faster than the views but the
    motion prior (tests/test_alignment_memory.py), so the rest is volumes, batches and a
    fixed part. Measured on K11 54013's and 67208's fine levels (358e3ce2):

    - ``state``: what persists through a step at its worst (the volume and the previous one,
      the linearisation's gradient and diagonal, the damping, the stepped gradient, TV's
      conductance, the free-voxel mask, the two shared images, the clipped step; growing a
      little over the steps, with a retried proposal's), 9 volumes and 0.2 GiB that no array
      holds (the compiled programs and the libraries' workspaces): measured at the worst step
      as 0.2 GiB and 8.2, 8.1 and 8.8 volumes on K11 54013, K11 67208 and I13 159815.
    - ``prior_held``: the motion prior's matrices and factor (``MotionPrior.device_bytes``).
    - ``transient``: the larger of the Schur solve's temporaries (conjugate gradients'
      vectors and their FP64 copies, the matvec's product and TV term, per-batch
      projections and columns; beyond the batches 4.6, 5.7 and 7.9 volumes on the same
      three scans, planned as 9 volumes and 8 batches) and the prior factor's (which runs
      before the solve, not during it).
    """
    d, (nz, ny, nx) = ops.detector, ops.shape
    volume = 4 * nz * ny * nx
    batch = 4 * ops.batch * (1 + ops.dofs) * d.nv * d.nu
    prior_bytes = prior.device_bytes()
    schur = 9 * volume + 8 * batch
    return {
        "state": 9 * volume + 200 * 2**20,
        "prior_held": prior_bytes["held"],
        "transient": max(schur, prior_bytes["transient"]),
    }


def level_stacks(
    ops: Any,
    y: np.ndarray,
    precision: np.ndarray | None,
    *,
    reserve: Mapping[str, int],
    volumes: Mapping[str, Any] | None = None,
    names: Iterable[str] | None = None,
    placement: Mapping[str, Place] | None = None,
) -> LevelStacks:
    """Allocate a level's ``volumes``, then place and allocate its stacks for its data.

    ``names`` are the stacks (all of them by default); ``reserve`` is kept free beside both.
    """
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
        reserve=reserve,
        volumes=volumes,
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
