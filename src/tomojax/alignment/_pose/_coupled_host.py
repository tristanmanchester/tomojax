"""Joint alignment whose measured data and Krylov vectors stay on the host.

The physical products are compiled view-batch passes, not a monolithic CG
program. Each pass owns one device accumulator. In the independent-pose
Schur product, columns are computed once per batch and immediately eliminated;
no full projected direction or Jacobian is kept on either host or device.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.alignment._geometry.parametrizations import apply_pose_updates
from tomojax.core.host_stream import host_source, read_views
from tomojax.core.joseph import plane_coefficients, project_ones_native

from ._coupled_host_linear import HostBlock, host_dot, solve_host_normal
from ._coupled_linear import CoupledLinearResult
from ._coupled_program import (
    CoupledArrays,
    CoupledSpec,
    filtered_operators,
    physical_projectors,
    pose_smoothness,
    volume_regularisation,
)
from ._pose_block import pose_block_solver
from ._pose_jacobian import pose_prediction_and_columns


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


def _linearization(predict, batch, params, volume, indices):
    """Weighted residual and columns from each view's single physical ray traversal."""
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
    return residual.reshape(batch.projections.shape), columns * batch.active[indices][None, :, None]


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


def _unit_prediction(spec, arrays, poses, forward, shape):
    """Use implicit unit support when its exact native operator is available."""
    if (
        spec.backend == "pallas"
        and spec.jacobian.integrator == "joseph"
        and not spec.jacobian.cone_beam
        and arrays.mask.ndim == 0
    ):
        coefficients = plane_coefficients(poses, spec.grid, spec.detector, arrays.det_grid)
        prediction = project_ones_native(coefficients, spec.grid, spec.detector)
        if prediction is not None:
            return prediction
    return forward(poses, jnp.ones(shape, jnp.float32), arrays.frames)


@partial(jax.jit, static_argnames=("spec", "mode", "weighted", "batch_size"))
def _pass(
    arrays: CoupledArrays,
    p: jax.Array,
    x: jax.Array,
    dx: jax.Array,
    dp: jax.Array,
    gram: jax.Array,
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
        predict = partial(
            pose_prediction_and_columns,
            grid=spec.grid,
            detector=spec.detector,
            det_grid=arrays.det_grid,
            options=spec.jacobian,
            column_indices=spec.active_columns,
        )
        batch, forward, adjoint, predict = filtered_operators(
            spec, batch, raw_forward, raw_adjoint, predict
        )
        t, q = take(poses, start), take(p, start)
        ax, rows, blocks, loss = carry
        if mode == "bound":
            y = weights**2 * _unit_prediction(spec, batch, t, raw_forward, shape)
            return raw_adjoint(t, y * valid[:, None, None], batch.frames, ax), rows, blocks, loss
        if mode == "loss":
            residual = weights * (forward(t, x, batch.frames) - batch.projections)
            loss += 0.5 * jnp.sum(residual**2 * valid[:, None, None])
            return ax, rows, blocks, loss
        residual, cols = _linearization(predict, batch, q, masked(x), indices)
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


@jax.jit
def _free_voxels(x, gradient):
    return (x > 0) | (gradient < 0)


class _HostSystem:
    """Own the resident image and tiny pose factorisation of one linear update."""

    def __init__(self, arrays, spec, p, x, weighted) -> None:
        self.arrays, self.spec, self.p, self.x, self.weighted = arrays, spec, p, x, weighted
        self.empty = np.empty((0,), np.float32)
        self.gram = jnp.zeros((p.shape[0], p.shape[1], p.shape[1]), jnp.float32)
        gx, gp, gram, _ = self.run("gradient")
        self.regularisation = partial(volume_regularisation, spec=spec)
        if spec.tv_weight:
            reg_grad, self.reg_hessian = jax.linearize(jax.grad(self.regularisation), x)
            gx += np.asarray(reg_grad)
        else:
            self.reg_hessian = None
        smooth = partial(pose_smoothness, weights=arrays.smoothness)
        sg, self.smooth_hessian = jax.linearize(jax.grad(smooth), p)
        active = np.asarray(arrays.active)
        counts = 1 if arrays.counts is None else np.asarray(arrays.counts)[:, None]
        self.movable = counts * active
        self.free = (
            np.array(_free_voxels(x, jax.device_put(gx)), order="C")
            if spec.nonnegative
            else np.bool_(True)
        )
        self.free = self.free & (np.asarray(arrays.mask) != 0)
        np.multiply(gx, self.free, out=gx)
        np.negative(gx, out=gx)
        self.rhs = (gx, -(gp + np.asarray(sg)) * self.movable)
        self.gram = jnp.asarray(gram) + spec.gn_damping * jnp.eye(p.shape[1], dtype=p.dtype)
        self.solve_pose = jax.jit(
            pose_block_solver(
                self.gram,
                arrays.smoothness * arrays.active,
                has_smoothness=spec.has_smoothness,
            )
        )
        bound, _, _, _ = self.run("bound")
        reg_bound = 12 * spec.tv_weight / spec.huber_delta if spec.tv_weight else 0
        for start in range(0, bound.shape[0], 8):
            chunk = bound[start : start + 8]
            np.maximum(
                chunk + reg_bound + spec.gn_volume_damping, spec.gn_volume_damping, out=chunk
            )
            np.reciprocal(chunk, out=chunk)
        self.inverse = bound

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
            spec=self.spec,
            mode=mode,
            weighted=self.weighted,
        )
        if mode in {"normal", "schur"}:
            regularisation = (
                jnp.float32(0) if self.reg_hessian is None else self.reg_hessian(direction)
            )
            product = _finish_product(
                results[0],
                direction,
                jax.device_put(self.free),
                self.spec.gn_volume_damping,
                regularisation,
            )
            ax = np.asarray(product, order="C")
        else:
            # Initialisation products are subsequently edited on the host;
            # Krylov products stay read-only and need no second host volume.
            ax = np.array(results[0], order="C")
        return ax, np.asarray(results[1]), np.asarray(results[2]), float(results[3])

    def normal(self, pair: HostBlock) -> HostBlock:
        dx = pair[0]
        dp = pair[1] * self.movable
        ax, ap, _, _ = self.run("normal", dx, dp)
        ap = (
            ap + np.asarray(self.smooth_hessian(jnp.asarray(dp))) + self.spec.gn_damping * dp
        ) * self.movable
        return ax, ap

    def schur(self, pair: HostBlock) -> HostBlock:
        dx = pair[0]
        if self.spec.has_smoothness:
            _, rows, _, _ = self.run("rows", dx)
            dp = -np.asarray(self.solve_pose(jnp.asarray(rows)))
            ax, _, _, _ = self.run("normal", dx, dp)
        else:
            ax, _, _, _ = self.run("schur", dx)
        return ax, self.empty

    def solve(self):
        spec = self.spec
        if spec.gn_joint_solver == "stacked":
            diagonal = np.asarray(jnp.diagonal(self.gram, axis1=1, axis2=2))
            pose_inverse = 1 / (diagonal + 12 * np.asarray(self.arrays.smoothness) ** 2)
            result = solve_host_normal(
                self.normal,
                self.rhs,
                (self.inverse, pose_inverse),
                max_iters=spec.gn_joint_iterations,
                rtol=spec.gn_joint_rtol,
            )
            return CoupledLinearResult(
                (jax.device_put(result.increment[0]), jnp.asarray(result.increment[1])),
                jnp.int32(result.iterations),
                jnp.float32(result.relative_residual),
                jnp.bool_(result.finite),
            )
        dp_rhs = np.asarray(self.solve_pose(jnp.asarray(self.rhs[1])))
        cross, _, _, _ = self.run("cross", dp=dp_rhs)
        # Initialisation products are owned host arrays: reuse this one rather
        # than retaining another full-volume reduced RHS and expression scratch.
        np.multiply(cross, self.free, out=cross)
        np.subtract(self.rhs[0], cross, out=cross)
        reduced_rhs = cross
        squared = host_dot(self.rhs, self.rhs)
        reduced_squared = host_dot((reduced_rhs, self.empty), (reduced_rhs, self.empty))
        result = solve_host_normal(
            self.schur,
            (reduced_rhs, self.empty),
            (self.inverse, self.empty),
            max_iters=spec.gn_joint_iterations,
            rtol=spec.gn_joint_rtol * np.sqrt(squared / max(reduced_squared, 1e-30)),
        )
        dx = result.increment[0]
        np.multiply(dx, self.free, out=dx)
        _, rows, _, _ = self.run("rows", dx)
        dp = self.movable * np.asarray(self.solve_pose(jnp.asarray(self.rhs[1] - rows)))
        actual = self.normal((dx, dp))
        residual_squared = 0.0
        for original, projected in zip(self.rhs, actual, strict=True):
            for start in range(0, original.shape[0], 8):
                difference = (original[start : start + 8] - projected[start : start + 8]).ravel()
                residual_squared += float(
                    np.einsum("i,i->", difference, difference, dtype=np.float64)
                )
        relative = np.sqrt(residual_squared / max(squared, 1e-30))
        return CoupledLinearResult(
            (jax.device_put(dx), jnp.asarray(dp)),
            jnp.int32(result.iterations),
            jnp.float32(relative),
            jnp.bool_(result.finite and np.isfinite(relative)),
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
            return _HostSystem(inputs(stack), spec, p, x, weighted).solve()

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
                + float(volume_regularisation(x, spec=spec))
                + float(pose_smoothness(p, weights=arrays.smoothness))
            )

    return update, loss
