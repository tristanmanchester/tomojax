"""Plane-sampled parallel-ray model and a JAX reference implementation.

Choose the fastest ray direction in voxel coordinates. Sample its voxel-centre
planes, interpolating the other two coordinates with zero padding. Bilinear
interpolation is the default; cubic convolution is an explicit alternative.
Multiply by the physical distance between planes. The resulting linear operator
differs from the ray-entry-phased trilinear sampler in ``projector.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ._plane_interpolation import cubic_weights_and_derivatives, validate_interpolation
from .geometry.base import grid_volume_origin

if TYPE_CHECKING:
    from .geometry.base import Detector, Grid


def validate_plane_geometry(poses: jax.Array, grid: Grid, detector: Detector) -> None:
    """Reject geometry that would make the plane map or gather bounds undefined."""
    if not np.isfinite((*grid_volume_origin(grid), *detector.det_center)).all():
        raise ValueError("cgls: Joseph grid origin and detector centre must be finite")
    host_poses = np.asarray(poses)
    rotations = host_poses[:, :3, :3]
    if not (
        np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-5)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5)
        and np.allclose(host_poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=2e-5)
    ):
        raise ValueError("cgls: Joseph sampling requires rigid homogeneous poses")


def select_axis(
    axis: jax.Array, x: jax.Array | int, y: jax.Array | int, z: jax.Array | int
) -> jax.Array:
    """Select an integer coordinate; explicit dtypes also support Triton lowering."""
    return jnp.where(
        axis == 0,
        jnp.asarray(x, jnp.int32),
        jnp.where(axis == 1, jnp.asarray(y, jnp.int32), jnp.asarray(z, jnp.int32)),
    )


def plane_coefficients(poses: jax.Array, grid: Grid, detector: Detector) -> jax.Array:
    """Prepare a dynamic per-view affine mapping from detector pixels to planes.

    Columns are ``axis, ub, vb, kb, cb, uc, vc, kc, cc, ds, inv00, inv01,
    inv10, inv11``. At plane k, q_b = ub*u + vb*v + kb*k + cb, and similarly
    q_c. The inverse 2x2 map bounds the rays that contribute to each voxel.
    Inputs are rigid world-from-object transforms and canonical detector pixels.
    """
    voxel = jnp.asarray((grid.vx, grid.vy, grid.vz), jnp.float32)
    origin = jnp.asarray(grid_volume_origin(grid), jnp.float32)
    u0 = detector.det_center[0] - (detector.nu - 1) * detector.du / 2
    v0 = detector.det_center[1] - (detector.nv - 1) * detector.dv / 2
    rotation = poses[:, :3, :3]
    base = jnp.sum(
        rotation * (jnp.asarray([u0, 0.0, v0], jnp.float32)[None, :] - poses[:, :3, 3])[:, :, None],
        axis=1,
    )
    base = (base - origin) / voxel
    direction = poses[:, 1, :3] / voxel
    axis = jnp.argmax(jnp.abs(direction), axis=1)
    b, c = (axis + 1) % 3, (axis + 2) % 3

    def take(values: jax.Array, index: jax.Array) -> jax.Array:
        return jnp.take_along_axis(values, index[:, None], axis=1)[:, 0]

    da = take(direction, axis)
    kb, kc = take(direction, b) / da, take(direction, c) / da
    u, v = rotation[:, 0, :] * detector.du / voxel, rotation[:, 2, :] * detector.dv / voxel
    ub, vb = take(u, b) - kb * take(u, axis), take(v, b) - kb * take(v, axis)
    uc, vc = take(u, c) - kc * take(u, axis), take(v, c) - kc * take(v, axis)
    cb, cc = take(base, b) - kb * take(base, axis), take(base, c) - kc * take(base, axis)
    determinant = ub * vc - vb * uc
    return jnp.stack(
        [
            axis.astype(jnp.float32),
            ub,
            vb,
            kb,
            cb,
            uc,
            vc,
            kc,
            cc,
            1 / jnp.abs(da),
            vc / determinant,
            -vb / determinant,
            -uc / determinant,
            ub / determinant,
        ],
        axis=1,
    )


def forward_jax(
    coefficients: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Reference model with ordinary AD; absolute weights are for roundoff bounds."""
    validate_interpolation(interpolation)
    shape = (grid.nx, grid.ny, grid.nz)
    u, v = jnp.meshgrid(jnp.arange(detector.nu), jnp.arange(detector.nv))

    def one_view(cf: jax.Array) -> jax.Array:
        axis = cf[0].astype(jnp.int32)
        ub, vb, kb, cb, uc, vc, kc, cc, weight = cf[1:10]
        na = select_axis(axis, *shape)
        nb, nc = (
            select_axis(axis, shape[1], shape[2], shape[0]),
            select_axis(axis, shape[2], shape[0], shape[1]),
        )

        def step(k: jax.Array, accumulator: jax.Array) -> jax.Array:
            qb, qc = (ub * u + vb * v) + (kb * k + cb), (uc * u + vc * v) + (kc * k + cc)
            ib, ic = jnp.floor(qb).astype(jnp.int32), jnp.floor(qc).astype(jnp.int32)
            wb, wc = qb - ib, qc - ic

            def sample(b: jax.Array, c: jax.Array) -> jax.Array:
                ix, iy, iz = (
                    select_axis(axis, k, c, b),
                    select_axis(axis, b, k, c),
                    select_axis(axis, c, b, k),
                )
                valid = (k < na) & (b >= 0) & (b < nb) & (c >= 0) & (c < nc)
                value = volume[
                    jnp.clip(ix, 0, shape[0] - 1),
                    jnp.clip(iy, 0, shape[1] - 1),
                    jnp.clip(iz, 0, shape[2] - 1),
                ]
                return jnp.where(valid, value, 0.0)

            if interpolation == "linear":
                value = (sample(ib, ic) * (1 - wc) + sample(ib, ic + 1) * wc) * (1 - wb) + (
                    sample(ib + 1, ic) * (1 - wc) + sample(ib + 1, ic + 1) * wc
                ) * wb
            else:
                weights_b, _ = cubic_weights_and_derivatives(wb)
                weights_c, _ = cubic_weights_and_derivatives(wc)
                if absolute_weights:
                    weights_b = tuple(jnp.abs(w) for w in weights_b)
                    weights_c = tuple(jnp.abs(w) for w in weights_c)
                value = jnp.zeros_like(accumulator)
                # Match the CUDA recurrence to keep near-zero residuals from
                # amplifying differences in summation order. The derivative
                # reference still uses ordinary JAX autodiff.
                for row in range(4):
                    row_value = jnp.zeros_like(accumulator)
                    for col in range(4):
                        row_value = row_value + sample(ib + row - 1, ic + col - 1) * weights_c[col]
                    value = value + row_value * weights_b[row]
            return accumulator + value * weight

        return jax.lax.fori_loop(
            0, max(shape), step, jnp.zeros((detector.nv, detector.nu), jnp.float32)
        )

    return jax.vmap(one_view)(coefficients)


def adjoint_jax(
    coefficients: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Reference transpose; ``absolute_weights`` selects abs(A).T for bounds."""
    zero = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)
    return jax.linear_transpose(
        lambda x: forward_jax(
            coefficients,
            x,
            grid,
            detector,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        ),
        zero,
    )(images)[0]


def forward_project_planes(
    coefficients: jax.Array,
    volume: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> jax.Array:
    """Apply a prepared plane model using the selected JAX or CUDA implementation."""
    if backend == "jax":
        return forward_jax(coefficients, volume, grid, detector, interpolation=interpolation)
    if backend == "pallas":
        from .pallas._pallas_joseph_autodiff import differentiable_forward

        return differentiable_forward(
            coefficients, volume, grid, detector, interpolation=interpolation
        )
    raise ValueError("Plane projection backend must be 'jax' or 'pallas'")


def sum_backproject_planes(
    coefficients: jax.Array,
    images: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
    absolute_weights: bool = False,
) -> jax.Array:
    """Apply the matched transpose, or abs(A).T when estimating cancellation."""
    if backend == "jax":
        return adjoint_jax(
            coefficients,
            images,
            grid,
            detector,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        )
    if backend == "pallas":
        from .pallas._pallas_joseph import adjoint_pallas

        return adjoint_pallas(
            coefficients,
            images,
            grid,
            detector,
            interpolation=interpolation,
            absolute_weights=absolute_weights,
        )
    raise ValueError("Plane projection backend must be 'jax' or 'pallas'")


def plane_l2_value_and_grad(
    coefficients: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return half squared error, coefficient gradient and volume gradient."""
    if backend == "pallas":
        from .pallas._pallas_joseph_derivatives import l2_loss_and_grads

        return l2_loss_and_grads(
            coefficients, volume, target, grid, detector, interpolation=interpolation
        )
    if backend == "jax":

        def loss(cf: jax.Array, x: jax.Array) -> jax.Array:
            return 0.5 * jnp.sum(
                (forward_jax(cf, x, grid, detector, interpolation=interpolation) - target) ** 2
            )

        value, (coefficient_grad, volume_grad) = jax.value_and_grad(loss, argnums=(0, 1))(
            coefficients, volume
        )
        return value, coefficient_grad, volume_grad
    raise ValueError("Plane projection backend must be 'jax' or 'pallas'")


def plane_l2_normal_equations(
    coefficients: jax.Array,
    volume: jax.Array,
    target: jax.Array,
    directions: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: str,
    interpolation: str = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-view loss, directional gradient, Gauss-Newton matrix and residuals."""
    if backend == "pallas":
        from .pallas._pallas_joseph_normals import pose_normal_equations

        return pose_normal_equations(
            coefficients, volume, target, directions, grid, detector, interpolation=interpolation
        )
    if backend != "jax":
        raise ValueError("Plane projection backend must be 'jax' or 'pallas'")

    def project(cf: jax.Array) -> jax.Array:
        return forward_jax(cf, volume, grid, detector, interpolation=interpolation)

    prediction = project(coefficients)
    # Direct JVPs keep ordinary forward mode; linearize would retain a large
    # per-plane residual tape before applying the tangent directions.
    jacobian = jax.vmap(
        lambda dc: jax.jvp(project, (coefficients,), (dc,))[1], in_axes=-1, out_axes=-1
    )(directions)
    jacobian = jacobian.reshape(coefficients.shape[0], -1, directions.shape[-1])
    residual = prediction - target
    flat = residual.reshape(coefficients.shape[0], -1)
    gradient = jnp.einsum("nmp,nm->np", jacobian, flat, precision=jax.lax.Precision.HIGHEST)
    matrix = jnp.einsum("nmp,nmq->npq", jacobian, jacobian, precision=jax.lax.Precision.HIGHEST)
    return 0.5 * jnp.sum(flat**2, axis=1), gradient, matrix, residual
