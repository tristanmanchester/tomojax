"""Joint alignment whose measured data and Krylov vectors stay on the host.

The physical products are compiled view-batch passes, not a monolithic CG
program. Each pass owns one device accumulator. Active pose columns can stay
in a host buffer for one fixed linearization; every product reads only its
current batch. Without spare host memory, the columns are recomputed instead.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.backends import device_free_memory_bytes
from tomojax.backends.host_stream import (
    host_buffer,
    host_source,
    read_block,
    read_views,
    write_block,
)

from ._coupled_linear import CoupledLinearResult, solve_coupled_normal, solve_schur_normal
from ._coupled_program import (
    CoupledArrays,
    CoupledSpec,
    constrained_pose_inverse,
    filtered_operators,
    free_volume_support,
    physical_columns,
    physical_projectors,
    pose_metric_inverse,
    pose_smoothness,
    positive_sensitivity,
    volume_metric_inverse,
    volume_regularisation,
)
from ._coupled_vectors import HOST, BlockVector
from ._pose_block import pose_block_solver


def _view_batch(arrays, start, size, spec, weighted):
    shape = (size, spec.detector.nv, spec.detector.nu)
    return arrays._replace(
        poses=jax.lax.dynamic_slice_in_dim(arrays.poses, start, size),
        projections=read_views(arrays.projections, start, shape),
        weights=read_views(arrays.weights, start, shape) if weighted else arrays.weights,
        frames=None
        if arrays.frames is None
        else jax.lax.dynamic_slice_in_dim(arrays.frames, start, size),
    )


def _linearization(predict, batch, params, volume, indices, key, start, gradient):
    """Weighted residual and columns from each view's single physical ray traversal."""
    if key is not None and not gradient:
        shape = (params.shape[0], indices.size, batch.projections[0].size)
        return jnp.float32(0), read_block(key, start, shape)
    residual, columns = jax.lax.map(
        lambda k: predict(
            params[k],
            batch.poses[k],
            None if batch.frames is None else batch.frames[k],
            volume,
            batch.projections[k],
            batch.weights if batch.weights.ndim == 0 else batch.weights[k],
        ),
        jnp.arange(params.shape[0]),
    )
    columns *= batch.active[indices][None, :, None]
    if key is not None:
        write_block(key, start, columns)
    return residual.reshape(batch.projections.shape), columns


def _data_increment(mode, t, dx, dp, gram, batch, forward, pose_forward, pose_adjoint):
    """Data-space image of a block increment, eliminating only independent poses locally."""
    if mode == "cross":
        return pose_forward(dp)
    y = batch.weights * forward(t, dx, batch.frames)
    if mode == "schur":
        solve = pose_block_solver(gram, batch.smoothness, has_smoothness=False)
        return y - pose_forward(solve(pose_adjoint(y)))
    if mode == "normal":
        y += pose_forward(dp)
    return y


@partial(jax.jit, static_argnames=("spec", "mode", "weighted", "batch_size"))
def _pass(
    arrays: CoupledArrays,
    p: jax.Array,
    x: jax.Array,
    dx: jax.Array,
    dp: jax.Array,
    gram: jax.Array,
    columns_key: jax.Array | None = None,
    *,
    spec: CoupledSpec,
    mode: str,
    weighted: bool,
    batch_size: int = 32,
):
    """One data-space pass, with all large arrays supplied as dynamic arguments."""
    n, width = p.shape
    size = min(batch_size, n)
    indices = jnp.asarray(spec.active_columns, jnp.int32)
    shape = (spec.grid.nx, spec.grid.ny, spec.grid.nz)
    counts = arrays.counts if arrays.counts is not None else jnp.ones(n, jnp.float32)
    poses = apply_pose_updates(arrays.poses, p, translation_frame=spec.jacobian.translation_frame)

    def take(a, start):
        return jax.lax.dynamic_slice_in_dim(a, start, size)

    def step(i, carry):
        start = jnp.minimum(i * size, n - size)
        valid = (start + jnp.arange(size) >= i * size) * take(counts, start)
        batch = _view_batch(arrays, start, size, spec, weighted)
        measured, weights = batch.projections, batch.weights

        def masked(v):
            return v if arrays.mask.ndim == 0 else arrays.mask * v

        raw_forward, raw_adjoint = physical_projectors(spec, batch, masked)
        predict = physical_columns(spec, batch)
        batch, forward, adjoint, predict = filtered_operators(
            spec, batch, raw_forward, raw_adjoint, predict
        )
        t, q = take(poses, start), take(p, start)
        ax, rows, blocks, loss = carry
        if mode == "bound":
            ax = positive_sensitivity(spec, batch, t, masked, shape, valid, ax)
            return ax, rows, blocks, loss
        if mode == "loss":
            residual = weights * (forward(t, x, batch.frames) - batch.projections)
            loss += 0.5 * jnp.sum(residual**2 * valid[:, None, None])
            return ax, rows, blocks, loss
        residual, cols = _linearization(
            predict, batch, q, masked(x), indices, columns_key, start, mode == "gradient"
        )
        highest = jax.lax.Precision.HIGHEST

        def pose_forward(value):
            return jnp.einsum("nk,nkp->np", value[:, indices], cols, precision=highest).reshape(
                measured.shape
            )

        def pose_adjoint(value):
            small = jnp.einsum("nkp,np->nk", cols, value.reshape(size, -1), precision=highest)
            return jnp.zeros((size, width), jnp.float32).at[:, indices].set(small)

        if mode == "gradient":
            y = residual
            small = jnp.einsum("nkp,nlp->nkl", cols, cols, precision=highest)
            block = (
                jnp.zeros((size, width, width), jnp.float32)
                .at[:, indices[:, None], indices[None, :]]
                .set(small)
            )
            blocks = jax.lax.dynamic_update_slice_in_dim(blocks, block, start, axis=0)
        else:
            y = _data_increment(
                mode,
                t,
                dx,
                take(dp, start),
                take(gram, start),
                batch,
                forward,
                pose_forward,
                pose_adjoint,
            )
        rows = jax.lax.dynamic_update_slice_in_dim(rows, pose_adjoint(y), start, axis=0)
        if mode != "rows":
            ax = adjoint(t, weights * y * valid[:, None, None], batch.frames, ax)
        return ax, rows, blocks, loss

    initial = (
        jnp.zeros(shape, jnp.float32) if mode not in {"loss", "rows"} else jnp.float32(0),
        jnp.zeros_like(p),
        jnp.zeros((n, width, width), jnp.float32),
        jnp.float32(0),
    )
    return jax.lax.fori_loop(0, -(-n // size), step, initial)


@partial(jax.jit, donate_argnums=(0,))
def _mask_direction(dx, free):
    return dx * free


@partial(jax.jit, donate_argnums=(0,))
def _finish_product(ax, dx, free, damping, regularisation):
    return free * (ax + damping * dx + regularisation)


@partial(jax.jit, static_argnames=("spec",))
def _regularisation_derivative(x, direction, *, spec):
    """Gradient or Hessian product, without retained volume-sized factors."""
    gradient = partial(jax.grad(volume_regularisation), spec=spec)
    if direction is None:
        return gradient(x)
    return jax.jvp(gradient, (x,), (direction,))[1]


_REGULARISATION_TILE_BYTES = 64 << 20


@partial(jax.jit, donate_argnums=(0,), static_argnames=("spec",))
def _finish_regularised_product(ax, direction, x, free, damping, *, spec):
    """Add a local regulariser derivative in slabs, reusing the data product.

    Each output depends on at most one neighbour along any coordinate, even
    for the mixed gradient components. One-cell halos therefore preserve the
    full-volume Hessian while bounding its gradient-field scratch. ``None``
    for the direction adds the gradient instead, without retaining HVP factors.
    """
    axis = max(range(3), key=x.shape.__getitem__)
    length = x.shape[axis]
    plane_bytes = 4 * math.prod(x.shape) // length
    core = min(length, max(1, _REGULARISATION_TILE_BYTES // (8 * plane_bytes)))
    width = min(length, core + 2)

    def take(value, start, size):
        return jax.lax.dynamic_slice_in_dim(value, start, size, axis=axis)

    def body(i, out):
        first = jnp.minimum(i * core, length - core)
        start = jnp.clip(first - 1, 0, length - width)
        curvature = _regularisation_derivative(
            take(x, start, width),
            None if direction is None else take(direction, start, width),
            spec=spec,
        )
        previous = take(out, first, core)
        local = previous if direction is None else previous + damping * take(direction, first, core)
        local += take(curvature, first - start, core)
        local *= free if free.ndim == 0 else take(free, first, core)
        valid_shape = [1, 1, 1]
        valid_shape[axis] = core
        valid = (first + jnp.arange(core) >= i * core).reshape(valid_shape)
        return jax.lax.dynamic_update_slice_in_dim(
            out, jnp.where(valid, local, previous), first, axis=axis
        )

    return jax.lax.fori_loop(0, (length + core - 1) // core, body, ax)


_regularisation_value = jax.jit(volume_regularisation, static_argnames=("spec",))


class _HostSystem:
    """Own the resident image and tiny pose factorisation of one linear update."""

    def __init__(self, arrays, spec, p, x, weighted, columns_key=None) -> None:
        self.arrays, self.spec, self.p, self.x, self.weighted = arrays, spec, p, x, weighted
        self.columns_key = columns_key
        self.empty = np.empty((0,), np.float32)
        self.gram = jnp.zeros((p.shape[0], p.shape[1], p.shape[1]), jnp.float32)
        gx, gp, gram, _ = self.run("gradient")
        if spec.tv_weight:
            gx += np.asarray(
                _finish_regularised_product(
                    jnp.zeros_like(x), None, x, jnp.float32(1), 0.0, spec=spec
                )
            )
        smooth = partial(pose_smoothness, weights=arrays.smoothness)
        sg, self.smooth_hessian = jax.linearize(jax.grad(smooth), p)
        active = np.asarray(arrays.active)
        counts = 1 if arrays.counts is None else np.asarray(arrays.counts)[:, None]
        self.movable = counts * active
        device_gradient = jax.device_put(gx) if spec.nonnegative else jnp.float32(0)
        self.free = np.array(
            free_volume_support(x, device_gradient, arrays.mask, nonnegative=spec.nonnegative),
            order="C",
        )
        np.multiply(gx, self.free, out=gx)
        np.negative(gx, out=gx)
        project_pose = constrained_pose_inverse(
            lambda value: value, spec, arrays, lambda value: value
        )
        self.project_pose = jax.jit(project_pose) if spec.refinement is not None else project_pose
        self.rhs = (gx, -np.asarray(self.project_pose((gp + np.asarray(sg)) * self.movable)))
        self.pose_diagonal = np.diagonal(gram, axis1=1, axis2=2)
        self.gram = jnp.asarray(gram) + spec.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype)
        solve_pose = pose_block_solver(
            self.gram,
            arrays.smoothness * arrays.active,
            has_smoothness=spec.has_smoothness,
        )
        self.solve_pose = jax.jit(
            constrained_pose_inverse(solve_pose, spec, arrays, lambda value: value)
        )
        bound, _, _, _ = self.run("bound")
        self.inverse = HOST.map_owned(bound, partial(volume_metric_inverse, spec=spec, array=np))

    def run(self, mode, dx=None, dp=None):
        # Explicit transfer avoids a staged identity and its second full-volume
        # buffer. The donated device direction must not alias host CG state.
        direction = (
            jnp.float32(0)
            if dx is None
            else _mask_direction(jax.device_put(dx, may_alias=False), jax.device_put(self.free))
        )
        results = _pass(
            self.arrays,
            self.p,
            self.x,
            direction,
            jnp.zeros_like(self.p) if dp is None else jnp.asarray(dp),
            self.gram,
            self.columns_key,
            spec=self.spec,
            mode=mode,
            weighted=self.weighted,
        )
        if mode in {"normal", "schur"}:
            if self.spec.tv_weight:
                product = _finish_regularised_product(
                    results[0],
                    direction,
                    self.x,
                    jax.device_put(self.free),
                    self.spec.gn_volume_damping,
                    spec=self.spec,
                )
            else:
                product = _finish_product(
                    results[0],
                    direction,
                    jax.device_put(self.free),
                    self.spec.gn_volume_damping,
                    jnp.float32(0),
                )
            ax = np.asarray(product, order="C")
        else:
            # Initialisation products are subsequently edited on the host;
            # Krylov products stay read-only and need no second host volume.
            ax = np.array(results[0], order="C")
        return ax, np.asarray(results[1]), np.asarray(results[2]), float(results[3])

    def normal(self, pair: BlockVector) -> BlockVector:
        dx = pair[0]
        dp = np.asarray(self.project_pose(pair[1] * self.movable))
        ax, ap, _, _ = self.run("normal", dx, dp)
        ap = (
            ap + np.asarray(self.smooth_hessian(jnp.asarray(dp))) + self.spec.gn_damping * dp
        ) * self.movable
        return ax, np.asarray(self.project_pose(ap))

    def schur(self, pair: BlockVector) -> BlockVector:
        dx = pair[0]
        if self.spec.has_smoothness or self.spec.refinement is not None:
            _, rows, _, _ = self.run("rows", dx)
            dp = -np.asarray(self.solve_pose(jnp.asarray(rows)))
            ax, _, _, _ = self.run("normal", dx, dp)
        else:
            ax, _, _, _ = self.run("schur", dx)
        return ax, self.empty

    def solve(self):
        spec = self.spec
        if spec.gn_joint_solver == "stacked":
            pose_inverse = pose_metric_inverse(
                self.pose_diagonal, np.asarray(self.arrays.smoothness), spec.gn_damping
            )
            result = solve_coupled_normal(
                self.normal,
                self.rhs,
                (self.inverse, pose_inverse),
                max_iters=spec.gn_joint_iterations,
                placement=HOST,
                rtol=spec.gn_joint_rtol,
                volume_ramp=None if spec.refinement is None else spec.refinement.ramp_damping,
            )
        else:
            result = solve_schur_normal(
                self.normal,
                self.schur,
                self.rhs,
                self.inverse,
                lambda dp: self.run("cross", dp=dp)[0],
                lambda dx: self.run("rows", dx)[1],
                lambda dp: np.asarray(self.solve_pose(jnp.asarray(dp))),
                free=self.free,
                movable=self.movable,
                placement=HOST,
                max_iters=spec.gn_joint_iterations,
                rtol=spec.gn_joint_rtol,
                volume_ramp=None if spec.refinement is None else spec.refinement.ramp_damping,
            )
        # Project before transfer, without allocating another host image.
        dx = np.asarray(result.increment[0])
        dx *= self.free
        return CoupledLinearResult(
            (
                jax.device_put(dx),
                jnp.asarray(self.project_pose(result.increment[1])),
            ),
            jnp.int32(result.iterations),
            jnp.float32(result.relative_residual),
            jnp.bool_(result.finite),
        )


def host_columns_fit(spec, n_views):
    """Reserve Krylov/FFT workspace and half the remaining RAM at planning time."""
    try:
        host = next(iter(jax.local_devices(backend="cpu")))
    except RuntimeError:
        return False
    available = device_free_memory_bytes(device=host)
    volume_bytes = 4 * spec.grid.nx * spec.grid.ny * spec.grid.nz
    columns_bytes = 4 * n_views * len(spec.active_columns) * spec.detector.nv * spec.detector.nu
    vectors = 12 if spec.refinement is not None else 8
    return (
        available is not None and columns_bytes <= max(0, available - vectors * volume_bytes) // 2
    )


def build_host_objective(arrays: CoupledArrays, spec: CoupledSpec) -> tuple[Callable, Callable]:
    """Bind scan arrays on the host; register callback sources only during calls."""
    projections = np.asarray(arrays.projections)
    weights = np.asarray(arrays.weights)
    weighted = weights.ndim > 0

    def inputs(stack):
        return arrays._replace(
            projections=stack.enter_context(host_source(projections)),
            weights=stack.enter_context(host_source(weights)) if weighted else arrays.weights,
        )

    def update(p, x):
        with ExitStack() as stack:
            shape = (p.shape[0], len(spec.active_columns), spec.detector.nv * spec.detector.nu)
            columns = stack.enter_context(host_buffer(shape)) if spec.cache_columns else None
            return _HostSystem(inputs(stack), spec, p, x, weighted, columns).solve()

    def loss(p, x):
        with ExitStack() as stack:
            result = _pass(
                inputs(stack),
                p,
                x,
                jnp.float32(0),
                jnp.zeros_like(p),
                jnp.zeros((p.shape[0], p.shape[1], p.shape[1]), jnp.float32),
                spec=spec,
                mode="loss",
                weighted=weighted,
            )[3]
            return (
                float(result)
                + float(_regularisation_value(x, spec=spec))
                + float(pose_smoothness(p, weights=arrays.smoothness))
            )

    return update, loss
