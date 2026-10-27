"""Reusable compiled joint objectives with scan arrays kept out of cache keys.

With a :class:`~tomojax.core.devices.ViewSplit` in the spec, the update and
loss run in one ``shard_map``: each device holds its share of the views (their
projections, poses, cached pose columns and pose increments) and the whole
volume. Volume and scalar sums over views are summed across the devices, and
so are the pose parts of the solver's inner products; pose smoothness, which
couples neighbouring views, gathers the small ``(views, 6)`` pose arrays.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.typing import ArrayLike

from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.core.cone import cone_backproject, cone_project, frame_coefficients
from tomojax.core.devices import VIEWS, ViewSplit
from tomojax.core.joseph import (
    forward_project_planes,
    plane_coefficients,
    project_ones_native,
    sum_backproject_planes,
)
from tomojax.core.projector import forward_project_view_T, sum_backproject_views_T
from tomojax.core.trilinear import exact_adjoint, exact_forward
from tomojax.recon.fista_tv_core import FistaCoreConfig, regulariser_value_arrays

from . import _matching_model
from ._coupled_linear import CoupledLinearResult, solve_coupled_normal, solve_schur_normal
from ._coupled_vectors import BlockVector
from ._matching_model import JointRefinement
from ._pose_block import pose_block_solver
from ._pose_jacobian import PoseJacobianOptions, pose_prediction_and_columns

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid
    from tomojax.recon.types import Regulariser


# Views projected together in the update's data-space passes.
_VIEWS_PER_BATCH = 32


class _BatchOps(NamedTuple):
    """One pose stack's operators over a batch of views starting at ``start``."""

    cached: jax.Array | None  # every view's pose columns, when they fit
    residual: jax.Array | None  # the weighted residual computed with cached columns
    view_columns: Callable  # view i's weighted pose columns
    batch_columns: Callable  # the batch's pose columns
    project: Callable  # (start, dx) -> W A dx over the batch
    transpose: Callable  # (start, valid, y, out) -> out + A^T W y over its new views
    pose_forward: Callable  # (start, cols, dp) -> C dp over the batch
    pose_adjoint: Callable  # (start, cols, y, out) -> out with the batch's rows C^T y


@dataclass(frozen=True)
class CoupledSpec:
    """Only choices that change the compiled program, never measurement arrays."""

    grid: Grid
    detector: Detector
    backend: Literal["jax", "cuda"]
    jacobian: PoseJacobianOptions
    active_columns: tuple[int, ...]
    cache_columns: bool
    regulariser: Regulariser
    huber_delta: float
    tv_weight: float
    nonnegative: bool
    gn_volume_damping: float
    gn_damping: float
    gn_joint_solver: Literal["stacked", "pose_eliminated"]
    gn_joint_rtol: float
    gn_joint_iterations: int
    has_smoothness: bool
    # The views shared among devices; the arrays then hold its padded views.
    split: ViewSplit | None = None
    refinement: JointRefinement | None = None


class CoupledArrays(NamedTuple):
    """Per-scan numerical inputs; changing values can reuse a compiled program."""

    poses: jax.Array
    projections: jax.Array
    # Per-pixel least-squares weights, or a scalar when every pixel weighs the same.
    weights: jax.Array
    mask: jax.Array
    active: jax.Array
    smoothness: jax.Array
    det_grid: tuple[jax.Array, jax.Array]
    frames: jax.Array | None = None
    # Which views count, with a split: its padding views do not.
    counts: jax.Array | None = None


def volume_regularisation(x: jax.Array, *, spec: CoupledSpec) -> jax.Array:
    if spec.tv_weight == 0:
        return jnp.float32(0)
    config = FistaCoreConfig(regulariser=spec.regulariser, huber_delta=spec.huber_delta)
    return spec.tv_weight * regulariser_value_arrays(x, config)


def pose_smoothness(p: jax.Array, *, weights: jax.Array) -> jax.Array:
    """The weighted squared second differences of the poses along the views."""
    d2 = p[:-2] - 2 * p[1:-1] + p[2:]
    return jnp.sum(jnp.square(d2 * weights))


def _dot(a: ArrayLike, b: ArrayLike) -> jax.Array:
    return jnp.vdot(a, b, precision=jax.lax.Precision.HIGHEST).real


def _view_specs(arrays: CoupledArrays) -> CoupledArrays:
    """How a split cuts ``arrays``: per-view arrays by views, the rest whole."""
    cut, whole = PartitionSpec(VIEWS), PartitionSpec()
    specs = {
        "poses": cut,
        "projections": cut,
        "weights": whole if arrays.weights.ndim == 0 else cut,
        "mask": whole,
        "active": whole,
        "smoothness": whole,
        "det_grid": (whole, whole),
        "frames": None if arrays.frames is None else cut,
        "counts": cut,
    }
    return CoupledArrays._make(specs[name] for name in CoupledArrays._fields)


@dataclass(frozen=True)
class _Shares:
    """One device's ``n`` views of a split, and sums over all of them; identities for None."""

    split: ViewSplit | None
    n: int

    def across(self, value: jax.Array) -> jax.Array:
        """``value``, a sum over this device's views, summed over every device's."""
        return value if self.split is None else jax.lax.psum(value, VIEWS)

    def whole(self, rows: jax.Array) -> jax.Array:
        """Every view's rows, from each device's ``(n, ...)`` share of them."""
        if self.split is None:
            return rows
        return jax.lax.all_gather(rows, VIEWS, tiled=True)[: self.split.n]

    def mine(self, rows: jax.Array) -> jax.Array:
        """Return this device's share of every view's rows."""
        if self.split is None:
            return rows
        padded = jnp.pad(rows, ((0, self.split.total - self.split.n), (0, 0)))
        return jax.lax.dynamic_slice_in_dim(padded, jax.lax.axis_index(VIEWS) * self.n, self.n)

    def dot(self, a: BlockVector, b: BlockVector) -> jax.Array:
        """The inner product of (volume, pose rows) pairs, each device holding its views' rows."""
        return _dot(a[0], b[0]) + self.across(_dot(a[1], b[1]))

    def linearized(self, gradient: Callable, rows: jax.Array) -> tuple[jax.Array, Callable]:
        """``gradient`` of every view's rows, and its derivative, at ``rows``, as shares."""
        value, derivative = jax.linearize(gradient, self.whole(rows))
        return self.mine(value), lambda change: self.mine(derivative(self.whole(change)))

    def pose_solver(
        self, gram: jax.Array, weights: jax.Array, *, has_smoothness: bool
    ) -> Callable[[jax.Array], jax.Array]:
        """:func:`pose_block_solver` of the shares' ``gram``; smoothness couples all views."""
        if self.split is None or not has_smoothness:
            return pose_block_solver(gram, weights, has_smoothness=has_smoothness)
        every = self.whole(gram.reshape(self.n, -1)).reshape(-1, *gram.shape[1:])
        solve = pose_block_solver(every, weights, has_smoothness=True)
        return lambda rows: self.mine(solve(self.whole(rows)))


def _pose_gram(ops: _BatchOps, n_views: int, width: int, indices: jax.Array) -> jax.Array:
    """Embed the active-column Gram matrices in the full small pose blocks."""
    if ops.cached is not None:
        gram = jnp.einsum(
            "nkp,nlp->nkl", ops.cached, ops.cached, precision=jax.lax.Precision.HIGHEST
        )
    else:

        def view_gram(i):
            columns = ops.view_columns(i)
            return jnp.matmul(columns, columns.T, precision=jax.lax.Precision.HIGHEST)

        gram = jax.lax.map(view_gram, jnp.arange(n_views))
    return (
        jnp.zeros((n_views, width, width), gram.dtype)
        .at[:, indices[:, None], indices[None, :]]
        .set(gram)
    )


def _finish_adjoint(value, masked, initial, accumulate):
    """Mask only the new transpose contribution, never the existing sum."""
    value = masked(value)
    return value if accumulate is None or initial is not None else accumulate + value


def _map_columns[T](fn: Callable[..., T], views: jax.Array, spec: CoupledSpec) -> T:
    """Bound vmapped cone columns while sharing one volume and CUDA launch."""
    batch = None
    if spec.backend == "cuda" and spec.jacobian.cone_beam and spec.jacobian.jacobian == "autodiff":
        # Only column images occupy additional batch scratch; ray work stays in
        # registers. Limit each batch of column images to 16 MiB, even for large detectors.
        per_view = 4 * (1 + len(spec.active_columns)) * spec.detector.nu * spec.detector.nv
        batch = max(1, min(32, 16 * 1024**2 // per_view))
    return jax.lax.map(fn, views, batch_size=batch)


def physical_projectors(
    spec: CoupledSpec, arrays: CoupledArrays, masked: Callable, *, absolute_weights: bool = False
) -> tuple[Callable, Callable]:
    """Physical forward/transpose, or their absolute weights for sensitivity bounds."""
    if spec.refinement is not None:
        assert not spec.jacobian.cone_beam
        assert spec.jacobian.translation_frame == "detector"
        assert set(spec.active_columns) <= {3, 5}
        model = _matching_model.projectors(
            spec.grid, spec.detector, backend=spec.backend, masked=masked, absolute=absolute_weights
        )
        return model.forward, model.adjoint
    return _geometric_projectors(spec, arrays, masked, absolute_weights=absolute_weights)


def _geometric_projectors(spec, arrays, masked, *, absolute_weights):
    """The exact/sample/Joseph physical models in the scan's current geometry."""
    backend = spec.backend
    det_grid = None if backend == "cuda" else arrays.det_grid
    joseph = {"joseph": "linear", "joseph_cubic": "cubic"}.get(spec.jacobian.integrator)
    # Nonnegative stencils already equal their absolute values; keep their fast paths.
    absolute_weights = absolute_weights and joseph == "cubic"

    cone_backend = backend

    def forward(t, x, frames=None):
        if spec.jacobian.cone_beam:  # each view in its lab frame
            assert frames is not None
            coeff = frame_coefficients(t, frames, spec.grid, spec.detector)
            return cone_project(masked(x), coeff, spec.grid, spec.detector, backend=cone_backend)
        if joseph is not None:
            return forward_project_planes(
                plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                masked(x),
                spec.grid,
                spec.detector,
                backend=backend,
                interpolation=joseph,
                absolute_weights=absolute_weights,
            )
        if spec.jacobian.integrator == "exact":
            return exact_forward(
                t, spec.grid, spec.detector, masked(x), backend=backend, det_grid=det_grid
            )
        return jax.lax.map(
            lambda ti: forward_project_view_T(
                ti,
                spec.grid,
                spec.detector,
                masked(x),
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=spec.jacobian.unroll,
                use_checkpoint=spec.jacobian.checkpoint,
            ),
            t,
        )

    def adjoint(t, y, frames=None, accumulate=None):
        # With no support multiplication the physical kernel can own the
        # accumulator, avoiding a second full volume for each view batch.
        initial = accumulate if arrays.mask.ndim == 0 else None

        finish = partial(_finish_adjoint, masked=masked, initial=initial, accumulate=accumulate)

        if spec.jacobian.cone_beam:
            assert frames is not None
            coeff = frame_coefficients(t, frames, spec.grid, spec.detector)
            return finish(
                cone_backproject(
                    y, coeff, spec.grid, spec.detector, backend=cone_backend, accumulate=initial
                )
            )
        if joseph is not None:
            return finish(
                sum_backproject_planes(
                    plane_coefficients(t, spec.grid, spec.detector, arrays.det_grid),
                    y,
                    spec.grid,
                    spec.detector,
                    backend=backend,
                    interpolation=joseph,
                    accumulate=initial,
                    absolute_weights=absolute_weights,
                )
            )
        if spec.jacobian.integrator == "exact":
            value = exact_adjoint(
                t, spec.grid, spec.detector, y, backend=backend, det_grid=det_grid
            )
            return _finish_adjoint(value, masked, None, accumulate)

        # Accumulate one volume, never a volume stack per view.
        def add(x, pair):
            ti, yi = pair
            return x + sum_backproject_views_T(
                ti[None],
                spec.grid,
                spec.detector,
                yi[None],
                gather_dtype="fp32",
                det_grid=det_grid,
                unroll=spec.jacobian.unroll,
            ), None

        value = jax.lax.scan(
            add, jnp.zeros((spec.grid.nx, spec.grid.ny, spec.grid.nz), jnp.float32), (t, y)
        )[0]
        return _finish_adjoint(value, masked, None, accumulate)

    return forward, adjoint


def _unit_prediction(spec, arrays, poses, forward, shape):
    """Use implicit unit support when its exact native operator is available."""
    if (
        spec.refinement is None
        and spec.backend == "cuda"
        and spec.jacobian.integrator == "joseph"
        and not spec.jacobian.cone_beam
        and arrays.mask.ndim == 0
    ):
        coefficients = plane_coefficients(poses, spec.grid, spec.detector, arrays.det_grid)
        prediction = project_ones_native(coefficients, spec.grid, spec.detector)
        if prediction is not None:
            return prediction
    return forward(poses, jnp.ones(shape, jnp.float32), arrays.frames)


def positive_sensitivity(spec, arrays, poses, masked, shape, valid, out):
    """Accumulate |A|^T W^2 |A|1, without filtering the positive metric."""
    forward, adjoint = physical_projectors(spec, arrays, masked, absolute_weights=True)
    y = _unit_prediction(spec, arrays, poses, forward, shape)
    y = y * arrays.weights**2 * valid[:, None, None]
    return adjoint(poses, y, arrays.frames, out)


@partial(jax.jit, static_argnames=("nonnegative",))
def free_volume_support(x, gradient, mask, *, nonnegative):
    """KKT support: a zero voxel can move only if its gradient points inward."""
    return ((x > 0) | (gradient < 0)) & (mask != 0) if nonnegative else mask != 0


def volume_metric_inverse(bound, *, spec, array=jnp):
    regularisation = 12 * spec.tv_weight / spec.huber_delta if spec.tv_weight else 0
    return 1 / array.maximum(
        bound + regularisation + spec.gn_volume_damping, spec.gn_volume_damping
    )


def pose_metric_inverse(diagonal, smoothness, damping):
    return 1 / (diagonal + 12 * smoothness**2 + damping)


def data_residual_filter(spec, columns):
    """Profile nuisance terms after whitening, including in pose derivatives."""
    if spec.refinement is None:
        return lambda value, _weights: value, columns
    nu, nv = spec.detector.nu, spec.detector.nv
    filtered = _matching_model.nuisance_filter(spec.refinement.native_width, spec.detector)

    def filtered_columns(params, nominal, frame, volume, target, weight):
        residual, derivative = columns(params, nominal, frame, volume, target, weight)
        return (
            filtered(residual.reshape(nv, nu), weight).ravel(),
            filtered(derivative.reshape(-1, nv, nu), weight).reshape(derivative.shape),
        )

    return filtered, filtered_columns


def physical_columns(spec: CoupledSpec, arrays: CoupledArrays) -> Callable:
    """Bind columns from the same physical model used by every volume product."""
    if spec.refinement is not None:
        return _matching_model.pose_columns(
            spec.grid, spec.detector, backend=spec.backend, indices=spec.active_columns
        )
    return partial(
        pose_prediction_and_columns,
        grid=spec.grid,
        detector=spec.detector,
        det_grid=arrays.det_grid,
        options=spec.jacobian,
        column_indices=spec.active_columns,
    )


def constrained_pose_inverse(solve, spec, arrays, across):
    """Apply the matching model's global gauge, not a per-batch constraint."""
    if spec.refinement is None:
        return solve
    counts = arrays.counts
    if counts is None:
        counts = jnp.ones(arrays.poses.shape[0], jnp.float32)
    return _matching_model.constrain_pose_inverse(solve, arrays.poses, counts, across)


def _pose_column_cache(p, x, *, spec: CoupledSpec, arrays: CoupledArrays, predict_columns):
    """Cache the primal residual and active columns together, or stream columns."""
    indices = jnp.asarray(spec.active_columns, jnp.int32)

    def residual_columns(i):
        residual, derivative = predict_columns(
            p[i],
            arrays.poses[i],
            None if arrays.frames is None else arrays.frames[i],
            x,
            arrays.projections[i],
            arrays.weights if arrays.weights.ndim == 0 else arrays.weights[i],
        )
        return residual, derivative * arrays.active[indices, None]

    def columns(i):
        return residual_columns(i)[1]

    # The primal comes for free with analytic columns. Reuse it for the
    # gradient instead of projecting x a second time; both die after this update.
    residual, cached = (
        _map_columns(residual_columns, jnp.arange(arrays.poses.shape[0]), spec)
        if spec.cache_columns
        else (None, None)
    )

    return residual, cached, columns


def _batch_residual(ops, start, volume, measured):
    """Reuse cached weighted residuals or evaluate the same batch from its image."""
    if ops.residual is None:
        return ops.project(start, volume) - measured
    cached = jax.lax.dynamic_slice_in_dim(ops.residual, start, measured.shape[0])
    return cached.reshape(measured.shape)


def _build_program(
    spec: CoupledSpec, arrays: CoupledArrays
) -> tuple[
    Callable[[jax.Array, jax.Array], CoupledLinearResult],
    Callable[[jax.Array, jax.Array], jax.Array],
]:
    """Build the loss and updates in the free image subspace.

    Nonlocal preconditioners may fill inactive null directions of the normal
    system; returned image increments are projected onto the free coordinates.
    """
    # This factory executes during tracing, behind stable module-level jit
    # functions. Its closures contain tracers rather than embedded scan arrays.
    cfg = spec
    mask, weights, active = arrays.mask, arrays.weights, arrays.active
    # Only active derivative images are cached. Keep the small pose vectors
    # and normal matrices full-width, so frozen nonzero poses and smoothness
    # retain exactly the same coordinates throughout the linear solve.
    indices = jnp.asarray(spec.active_columns, jnp.int32)
    # A scalar mask means none: skip multiplying volumes by it, which would copy them.
    unmasked = mask.ndim == 0

    def masked(v):
        return v if unmasked else mask * v

    n_views = arrays.poses.shape[0]  # a device's share, with a split
    shares = _Shares(spec.split, n_views)
    across = shares.across
    counts = arrays.counts if arrays.counts is not None else jnp.ones(n_views, jnp.float32)
    movable = counts[:, None] * active  # pose rows that may change
    predict_columns = physical_columns(spec, arrays)
    project_pose = constrained_pose_inverse(lambda value: value, spec, arrays, across)
    regularisation = partial(volume_regularisation, spec=spec)
    smoothness = partial(pose_smoothness, weights=arrays.smoothness)

    def poses(p):
        return apply_pose_updates(
            arrays.poses, p, translation_frame=spec.jacobian.translation_frame
        )

    forward, adjoint = physical_projectors(spec, arrays, masked)
    filtered, predict_columns = data_residual_filter(spec, predict_columns)

    # Data-space work runs over batches of views: projecting, comparing and
    # transposing a batch before the next, so no projection-sized array is
    # stored but the cached pose columns (POSE_WIDTH of them, when they fit)
    # and one in the pose-eliminated solve. The last batch is shifted back to
    # end at the last view; ``valid`` marks its views not seen before.
    size = min(n_views, _VIEWS_PER_BATCH)
    count = -(-n_views // size)
    volume_zeros = jnp.zeros((spec.grid.nx, spec.grid.ny, spec.grid.nz), jnp.float32)

    def take(a, start):
        return jax.lax.dynamic_slice_in_dim(a, start, size)

    def over_batches(body, init):
        def step(i, carry):
            start = jnp.minimum(i * size, n_views - size)
            valid = (start + jnp.arange(size) >= i * size).astype(jnp.float32)
            return body(start, valid * take(counts, start), carry)

        return jax.lax.fori_loop(0, count, step, init)

    def batch_weights(start):
        return weights if weights.ndim == 0 else take(weights, start)

    def batch_frames(start):
        return None if arrays.frames is None else take(arrays.frames, start)

    def loss(p, x):
        t = poses(p)

        def body(start, valid, total):
            residual = batch_weights(start) * (
                forward(take(t, start), x, batch_frames(start)) - take(arrays.projections, start)
            )
            residual = filtered(residual, batch_weights(start))
            squares = jnp.sum(jnp.square(residual), axis=(1, 2))
            return total + 0.5 * jnp.sum(valid * squares)

        data = across(over_batches(body, jnp.float32(0)))
        return data + regularisation(x) + smoothness(shares.whole(p))

    highest = jax.lax.Precision.HIGHEST

    def batch_ops(t, p, x):  # the batch operators of one pose stack; see _BatchOps
        residual, cached, columns = _pose_column_cache(
            p, masked(x), spec=spec, arrays=arrays, predict_columns=predict_columns
        )

        def view_columns(i):
            return cached[i] if cached is not None else columns(i)

        def batch_columns(start):
            if cached is not None:
                return take(cached, start)
            return _map_columns(columns, start + jnp.arange(size), spec)

        def project(start, dx):
            weight = batch_weights(start)
            return filtered(weight * forward(take(t, start), dx, batch_frames(start)), weight)

        def transpose(start, valid, y, out):
            weight = batch_weights(start)
            y = weight * filtered(y, weight) * valid[:, None, None]
            return adjoint(take(t, start), y, batch_frames(start), accumulate=out)

        def pose_forward(start, cols, dp):
            return jnp.einsum(
                "nk,nkp->np", take(dp, start)[:, indices], cols, precision=highest
            ).reshape((size, spec.detector.nv, spec.detector.nu))

        def pose_adjoint(start, cols, y, out):
            rows = jnp.einsum("nkp,np->nk", cols, y.reshape(size, -1), precision=highest)
            rows = jnp.zeros((size, active.size), rows.dtype).at[:, indices].set(rows)
            return jax.lax.dynamic_update_slice_in_dim(out, rows, start, axis=0)

        return _BatchOps(
            cached,
            residual,
            view_columns,
            batch_columns,
            project,
            transpose,
            pose_forward,
            pose_adjoint,
        )

    block_dot = shares.dot

    def solve_pose_eliminated(ops, p, free, rhs, normal, reg_hessian, volume_inverse):
        """Eliminate the per-view pose blocks exactly and solve for the volume by CG."""
        gram = _pose_gram(ops, n_views, p.shape[1], indices)
        solve_pose = shares.pose_solver(
            gram + cfg.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype),
            arrays.smoothness * active,
            has_smoothness=spec.has_smoothness,
        )

        solve_pose = constrained_pose_inverse(solve_pose, spec, arrays, across)

        def cross(dp):
            def body(start, valid, out):
                return ops.transpose(
                    start, valid, ops.pose_forward(start, ops.batch_columns(start), dp), out
                )

            return across(over_batches(body, volume_zeros))

        def pose_rows(dx):  # C^T W A dx, and the batches' W A dx
            def body(start, _valid, carry):
                rows, ys = carry
                y = ops.project(start, dx)
                rows = ops.pose_adjoint(start, ops.batch_columns(start), y, rows)
                return rows, jax.lax.dynamic_update_slice_in_dim(ys, y, start, axis=0)

            stored = jnp.zeros((n_views, spec.detector.nv, spec.detector.nu), jnp.float32)
            return over_batches(body, (jnp.zeros_like(p), stored))

        def reduced_normal(increment):
            dx = free * increment[0]
            rows, ys = pose_rows(dx)
            dp = solve_pose(rows)

            def body(start, valid, out):
                y = take(ys, start) - ops.pose_forward(start, ops.batch_columns(start), dp)
                return ops.transpose(start, valid, y, out)

            ax = across(over_batches(body, volume_zeros))
            return (
                free * (ax + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                increment[1],
            )

        return solve_schur_normal(
            normal,
            reduced_normal,
            rhs,
            volume_inverse,
            cross,
            lambda dx: pose_rows(dx)[0],
            solve_pose,
            free=free,
            movable=movable,
            max_iters=cfg.gn_joint_iterations,
            rtol=cfg.gn_joint_rtol,
            dot=block_dot,
            volume_ramp=None if spec.refinement is None else spec.refinement.ramp_damping,
        )

    def update(p, x):
        ops = batch_ops(poses(p), p, x)
        if cfg.tv_weight:
            reg_grad, reg_hessian = jax.linearize(jax.grad(regularisation), x)
        else:  # no regulariser: no zero volumes to carry
            reg_grad, reg_hessian = jnp.float32(0), lambda _dx: jnp.float32(0)
        smooth_grad, smooth_hessian = shares.linearized(jax.grad(smoothness), p)

        def gradient_body(start, valid, carry):
            gx, gp = carry
            weight = batch_weights(start)
            measured = filtered(weight * take(arrays.projections, start), weight)
            residual = _batch_residual(ops, start, x, measured)
            cols = ops.batch_columns(start)
            return (
                ops.transpose(start, valid, residual, gx),
                ops.pose_adjoint(start, cols, residual, gp),
            )

        gx, gp = over_batches(gradient_body, (volume_zeros, jnp.zeros_like(p)))
        gx = across(gx) + reg_grad
        gp = project_pose((gp + smooth_grad) * movable)
        free = free_volume_support(x, gx, mask, nonnegative=cfg.nonnegative)

        def normal(increment):  # y = W A dx + C dp, then (A^T W y, C^T y), batch by batch
            dx, dp = free * increment[0], project_pose(movable * increment[1])

            def body(start, valid, carry):
                ax, ap = carry
                cols = ops.batch_columns(start)
                y = ops.project(start, dx) + ops.pose_forward(start, cols, dp)
                return ops.transpose(start, valid, y, ax), ops.pose_adjoint(start, cols, y, ap)

            ax, ap = over_batches(body, (volume_zeros, jnp.zeros_like(p)))
            return (
                free * (across(ax) + reg_hessian(dx) + cfg.gn_volume_damping * dx),
                project_pose(movable * (ap + smooth_hessian(dp) + cfg.gn_damping * dp)),
            )

        def ones_body(start, valid, out):
            batch = arrays._replace(weights=batch_weights(start), frames=batch_frames(start))
            return positive_sensitivity(
                spec, batch, take(poses(p), start), masked, x.shape, valid, out
            )

        row_bound = across(over_batches(ones_body, volume_zeros))
        volume_inverse = volume_metric_inverse(row_bound, spec=spec)
        rhs = (-(free * gx), -gp)
        if cfg.gn_joint_solver == "pose_eliminated":
            return solve_pose_eliminated(ops, p, free, rhs, normal, reg_hessian, volume_inverse)
        pose_diagonal = (
            jnp.sum(ops.cached**2, axis=2)
            if ops.cached is not None
            else jax.lax.map(
                lambda i: jnp.sum(ops.view_columns(i) ** 2, axis=1), jnp.arange(n_views)
            )
        )
        pose_diagonal = jnp.zeros_like(p).at[:, indices].set(pose_diagonal)
        pose_inverse = pose_metric_inverse(pose_diagonal, arrays.smoothness, cfg.gn_damping)
        step = solve_coupled_normal(
            normal,
            rhs,
            (volume_inverse, pose_inverse),
            max_iters=cfg.gn_joint_iterations,
            rtol=cfg.gn_joint_rtol,
            dot=block_dot,
            volume_ramp=None if spec.refinement is None else spec.refinement.ramp_damping,
        )
        return step._replace(increment=(free * step.increment[0], project_pose(step.increment[1])))

    return update, loss


@partial(jax.jit, static_argnames=("spec",))
def run_update(
    arrays: CoupledArrays, p: jax.Array, x: jax.Array, *, spec: CoupledSpec
) -> CoupledLinearResult:
    """The joint step from ``(p, x)``; with a split, ``p`` and the step's ``dp`` have its views."""

    def update(arrays: CoupledArrays, p: jax.Array, x: jax.Array) -> CoupledLinearResult:
        return _build_program(spec, arrays)[0](p, x)

    if spec.split is None:
        return update(arrays, p, x)
    cut, whole = PartitionSpec(VIEWS), PartitionSpec()
    # Unchecked: the loops start from zeros that each device's views then change.
    return jax.shard_map(
        update,
        mesh=spec.split.mesh,
        in_specs=(_view_specs(arrays), cut, whole),
        out_specs=CoupledLinearResult._make(((whole, cut), whole, whole, whole)),
        check_vma=False,
    )(arrays, p, x)


@partial(jax.jit, static_argnames=("spec",))
def run_loss(arrays: CoupledArrays, p: jax.Array, x: jax.Array, *, spec: CoupledSpec) -> jax.Array:
    """The joint objective at ``(p, x)``; with a split, ``p`` has its views."""

    def loss(arrays: CoupledArrays, p: jax.Array, x: jax.Array) -> jax.Array:
        return _build_program(spec, arrays)[1](p, x)

    if spec.split is None:
        return loss(arrays, p, x)
    cut, whole = PartitionSpec(VIEWS), PartitionSpec()
    return jax.shard_map(
        loss,
        mesh=spec.split.mesh,
        in_specs=(_view_specs(arrays), cut, whole),
        out_specs=whole,
        check_vma=False,
    )(arrays, p, x)
