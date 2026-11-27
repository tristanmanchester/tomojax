"""The image Krylov solve across devices, partitioned by whole projector batches."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

from tomojax import progress
from tomojax.core.devices import VIEWS, ViewSplit, announce, pad_views, shared_devices, view_split
from tomojax.errors import device_out_of_memory

from . import _tv as tv
from ._krylov import solve as krylov_solve
from ._projection import Coefficients

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import _common
    from ._krylov import KrylovResult
    from ._operators import Operators
    from ._pose import MotionPrior, PoseFactor
    from ._stacks import ArrayStack


type SchurOperands = tuple[
    Coefficients,
    ArrayStack,
    ArrayStack,
    PoseFactor,
    jax.Array,
    jax.Array | None,
    tuple[jax.Array, jax.Array],
    float,
    jax.Array,
    jax.Array,
    ArrayStack | None,
    _common.Common,
    _common.PoseSolve,
]


def batch_split(ops: Operators) -> ViewSplit | None:
    """Share complete view batches on the allocated GPUs, respecting an explicit default device."""
    devices = shared_devices()
    if devices is None and jax.config.jax_default_device is None and jax.default_backend() == "gpu":
        devices = tuple(jax.local_devices())
    if devices is None or len(devices) < 2 or ops.nb < 2:
        return None
    return view_split(devices[: ops.nb], ops.nb)


class SharedViewProducts:
    """Each device keeps its weighted projections through all products of the Krylov loop."""

    def __init__(self, split: ViewSplit, ops: Operators) -> None:
        self.split, self.ops = split, ops
        self.sharding = NamedSharding(split.mesh, PartitionSpec(VIEWS))

    def spec(self, stack: ArrayStack | None) -> PartitionSpec:
        """A host key is replicated; a resident stack is split on its view axis."""
        return PartitionSpec() if stack is None or stack.host else PartitionSpec(VIEWS)

    def weighted_forward(
        self,
        q: Coefficients,
        x: jax.Array,
        sw: ArrayStack,
        j: ArrayStack,
        precision: ArrayStack | None,
        r: ArrayStack,
    ) -> tuple[ArrayStack, jax.Array]:
        ops = self.ops

        def forward(
            q: Coefficients,
            x: jax.Array,
            sw: ArrayStack,
            j: ArrayStack,
            precision: ArrayStack | None,
            r: ArrayStack,
        ) -> tuple[ArrayStack, jax.Array]:
            first = jax.lax.axis_index(VIEWS) * len(q.valid)
            r, rows = ops.forward_scores(q, x, sw, j, precision, r, first)
            rows = jax.lax.all_gather(rows, VIEWS, tiled=True)
            return r, ops.parameters.reduce_score(rows[: ops.n])

        share, whole = PartitionSpec(VIEWS), PartitionSpec()
        return jax.shard_map(
            forward,
            mesh=self.split.mesh,
            in_specs=(share, whole, self.spec(sw), self.spec(j), self.spec(precision), share),
            out_specs=(share, whole),
            check_vma=False,
        )(q, x, sw, j, precision, r)

    def weighted_back(
        self,
        q: Coefficients,
        r: ArrayStack,
        sw: ArrayStack,
        j: ArrayStack,
        p: jax.Array,
        active: tuple[jax.Array, ...] | None,
        precision: ArrayStack | None,
    ) -> jax.Array:
        ops, split = self.ops, self.split

        def back(
            q: Coefficients,
            r: ArrayStack,
            sw: ArrayStack,
            j: ArrayStack,
            p: jax.Array,
            active: tuple[jax.Array, ...] | None,
            precision: ArrayStack | None,
        ) -> jax.Array:
            rows = pad_views(ops.pad_poses(p), split.total * ops.batch)
            count = len(q.valid)
            share_rows = jax.lax.dynamic_slice_in_dim(
                rows, jax.lax.axis_index(VIEWS) * count, count
            )
            # Each batch adds into an FP32 image. Starting another partial sum on each
            # device changes the image Krylov path. Pass the running sum to the next share.
            rank = jax.lax.axis_index(VIEWS)
            size = split.mesh.size
            ring = [(i, (i + 1) % size) for i in range(size)]

            def contribute(source: jax.Array, accumulator: jax.Array) -> jax.Array:
                accumulator = jax.lax.cond(
                    rank == source,
                    lambda: ops.back_rows(
                        q, r, sw, j, share_rows, active, precision, accumulator, rank * count
                    ),
                    lambda: accumulator,
                )
                return jax.lax.ppermute(accumulator, VIEWS, ring)

            accumulator = jax.lax.fori_loop(0, size, contribute, jnp.zeros(ops.shape, np.float32))
            return jax.lax.psum(accumulator, VIEWS)

        share, whole = PartitionSpec(VIEWS), PartitionSpec()
        return jax.shard_map(
            back,
            mesh=split.mesh,
            in_specs=(
                share,
                share,
                self.spec(sw),
                self.spec(j),
                whole,
                whole,
                self.spec(precision),
            ),
            out_specs=whole,
            check_vma=False,
        )(q, r, sw, j, p, active, precision)

    def partition(self, array: jax.Array, *, repeat: bool = False) -> jax.Array:
        """Share an already batch-padded table and fill any last *batch* needed by the mesh."""
        blocks = self.ops.batches(array)
        placed = self.split.place(blocks, repeat=repeat, dtype=array.dtype)
        return placed.reshape((-1, *array.shape[1:]))

    def solve(
        self,
        program: Callable[[*SchurOperands], KrylovResult],
        *args: *SchurOperands,
    ) -> KrylovResult:
        """Keep volume and pose state whole, sharing ray data and Krylov scratch for this step."""
        q, sw, j, factor, free, d, damping, lm, rhs, invdiag, precision, common, poses = args

        owner = rhs.sharding if rhs.committed else None

        def cut(stack: ArrayStack) -> ArrayStack:
            if stack.host:
                return self.split.everywhere(stack)
            return self.ops.stack(self.partition(stack.data))

        if any(stack.host for stack in (sw, j, precision) if stack is not None):
            jax.effects_barrier()  # finish the linearization's host writes before parallel reads
        packet = Coefficients(self.partition(q.rays, repeat=True), self.partition(q.valid))
        shared = self.split.everywhere((factor, free, d, damping, lm, rhs, invdiag, common, poses))
        factor, free, d, damping, lm, rhs, invdiag, common, poses = shared
        result = program(
            packet,
            cut(sw),
            cut(j),
            factor,
            free,
            d,
            damping,
            lm,
            rhs,
            invdiag,
            None if precision is None else cut(precision),
            common,
            poses,
        )
        # Preserve the caller's placement as well as its values. A committed
        # volume makes XLA recompile the next single-device RHS with sharding:
        # on CPU that changes its rounding and the subsequent Krylov trajectory.
        # Only the completed volume step and four scalars leave the mesh.
        result = jax.device_put(result if owner is not None else jax.device_get(result), owner)
        _ = result.solution.block_until_ready()
        announce(self.split.mesh.devices.flat)
        return result


class ImageSolve:
    """The damped image Schur system, with the same CG stopping rule for every placement."""

    def __init__(
        self,
        ops: Operators,
        prior: MotionPrior,
        lam: float,
        scales: tuple[float, float, float],
        inner: int,
        image_tolerance: float,
    ) -> None:
        split = batch_split(ops)
        shared = None if split is None else SharedViewProducts(split, ops)

        def schur(
            q: Coefficients,
            sw: ArrayStack,
            j: ArrayStack,
            factor: PoseFactor,
            free: jax.Array,
            d: jax.Array | None,
            damping: tuple[jax.Array, jax.Array],
            lm: float,
            rhs: jax.Array,
            invdiag: jax.Array,
            precision: ArrayStack | None,
            common: _common.Common,
            poses: _common.PoseSolve,
            r: ArrayStack,
            *,
            distributed: bool = False,
        ) -> tuple[KrylovResult, ArrayStack]:
            # Fixed voxels' transpose is discarded: the gather may skip them.
            active = ops.projection.active(free)

            def matvec(v: jax.Array, r: ArrayStack) -> tuple[jax.Array, ArrayStack]:
                v = v * free
                products = shared if distributed else ops
                assert products is not None
                r, jr = products.weighted_forward(q, v, sw, j, precision, r)
                bv = common.of_image(v)
                p = poses(prior.solve, factor, common, jr + common.poses.T @ (common.inverse @ bv))
                out = products.weighted_back(q, r, sw, j, p, active, precision)
                out = out + common.image(common.inverse @ (bv - common.poses @ p))
                if d is not None:
                    out = out + jnp.float32(lam) * tv.diffusion(v, d, scales)
                diagonal, floor = damping
                out = out + (jnp.float32(lm) * jnp.maximum(diagonal, floor)) * v
                return out * free, r

            return krylov_solve(matvec, rhs, invdiag, inner, rtol=image_tolerance, work=r)

        def shared_solve(*args: *SchurOperands) -> KrylovResult:
            # This product's rays are overwritten before their first read and
            # discarded when CG finishes. The proposal computes a new projection.
            assert shared is not None
            shape = (len(args[0].valid), ops.detector.nv, ops.detector.nu)
            work = ops.stack(
                jax.lax.with_sharding_constraint(jnp.zeros(shape, np.float32), shared.sharding)
            )
            return schur(*args, work, distributed=True)[0]

        self.single = jax.jit(schur, donate_argnums=(8, 13))
        self.shared = None if shared is None else partial(shared.solve, jax.jit(shared_solve))

    def __call__(
        self,
        work: ArrayStack,
        *args: *SchurOperands,
    ) -> tuple[KrylovResult, ArrayStack]:
        if self.shared is not None:
            try:
                result = self.shared(*args)
            except (RuntimeError, MemoryError) as exc:
                if not device_out_of_memory(exc):
                    raise
                progress.retried("single-device image solve")
                self.shared = None
            else:
                return result, work
        return self.single(*args, work)
