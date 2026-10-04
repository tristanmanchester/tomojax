"""Exact line integration of the zero-extended trilinear voxel basis.

The JAX reference supports ordinary differentiation. CUDA exposes an explicit
matched adjoint and fused five-parameter pose normal equations; its raw calls
are not automatically differentiable. All distances use the grid's physical
units, and poses map object coordinates to world coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.core._trilinear_reference import forward_project_view_exact_T
from tomojax.core.validation import (
    validate_detector,
    validate_detector_grid,
    validate_grid,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)

if TYPE_CHECKING:
    from tomojax.core.geometry.base import Detector, Grid


def _real_fp32(value: jax.Array) -> jax.Array:
    if jnp.iscomplexobj(value):
        raise ValueError("exact trilinear integration requires real arrays")
    return jnp.asarray(value, jnp.float32)


def _validate_backend(backend: str, detector: Detector, det_grid: object) -> None:
    if backend not in {"jax", "pallas"}:
        raise ValueError("exact trilinear integration: backend must be 'jax' or 'pallas'")
    if backend == "pallas":
        if jax.default_backend() != "gpu" or "NVIDIA" not in jax.devices()[0].device_kind:
            raise ValueError("exact trilinear integration: Pallas requires an NVIDIA CUDA device")
        from tomojax.core.pallas._pallas_config import _ensure_canonical_detector_grid

        _ensure_canonical_detector_grid(detector, det_grid)


def exact_forward(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    volume: jax.Array,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Integrate each parallel ray exactly within each trilinear interpolation cell.

    Rigid ``poses`` have shape ``(views, 4, 4)``. Output is FP32 with shape
    ``(views, detector.nv, detector.nu)``. Exact refers to polynomial quadrature,
    not infinite precision or detector-area integration. CUDA uses canonical
    detector pixels; the JAX reference also accepts explicit detector grids.
    """
    validate_grid(grid, "exact_forward")
    validate_detector(detector, "exact_forward")
    validate_volume(volume, grid, context="exact_forward")
    if poses.ndim != 3 or poses.shape[0] < 1:
        raise ValueError("exact_forward: poses must have nonempty shape (views, 4, 4)")
    validate_pose_stack(poses, poses.shape[0], context="exact_forward")
    validate_detector_grid(det_grid, detector, context="exact_forward")
    _validate_backend(backend, detector, det_grid)
    poses, volume = _real_fp32(poses), _real_fp32(volume)
    if backend == "pallas":
        from tomojax.core.pallas._pallas_trilinear import coefficients, run

        return run(coefficients(poses, grid, detector), volume, grid, detector)
    return jax.lax.map(
        lambda t: forward_project_view_exact_T(t, grid, detector, volume, det_grid=det_grid), poses
    )


def exact_adjoint(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    images: jax.Array,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Apply the matched FP32 transpose, accumulating views into one volume."""
    n, _, _ = validate_projection_stack(images, detector, context="exact_adjoint")
    validate_pose_stack(poses, n, context="exact_adjoint")
    validate_grid(grid, "exact_adjoint")
    validate_detector_grid(det_grid, detector, context="exact_adjoint")
    _validate_backend(backend, detector, det_grid)
    poses, images = _real_fp32(poses), _real_fp32(images)
    if backend == "pallas":
        from tomojax.core.pallas._pallas_trilinear import coefficients, run

        return run(coefficients(poses, grid, detector), images, grid, detector, mode="adjoint")
    initial = jnp.zeros((grid.nx, grid.ny, grid.nz), jnp.float32)

    def add_view(
        accumulator: jax.Array, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, None]:
        pose, image = inputs
        _, pullback = jax.vjp(
            lambda x: forward_project_view_exact_T(pose, grid, detector, x, det_grid=det_grid),
            initial,
        )
        return accumulator + pullback(image)[0], None

    result, _ = jax.lax.scan(add_view, initial, (poses, images))
    return result


def exact_pose_normal_equations(
    poses: jax.Array,
    pose_directions: jax.Array,
    grid: Grid,
    detector: Detector,
    volume: jax.Array,
    targets: jax.Array,
    weights: jax.Array,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return per-view weighted loss, gradient, Hessian approximation and residual.

    ``pose_directions`` has shape ``(views, 5, 4, 4)`` and supplies derivatives
    of the rigid matrices with respect to the five chosen pose coordinates.
    ``weights`` multiplies the residual, so its square weights normal equations.
    The returned data residual is also multiplied by squared weights. CUDA
    integrates pose derivatives with constant per-ray state and reduces small
    matrices, without retaining a full projection Jacobian.
    """
    n, _, _ = validate_projection_stack(targets, detector, context="exact pose normals")
    validate_pose_stack(poses, n, context="exact pose normals")
    validate_volume(volume, grid, context="exact pose normals")
    validate_detector_grid(det_grid, detector, context="exact pose normals")
    if pose_directions.shape != (n, 5, 4, 4):
        raise ValueError("exact pose normals: pose_directions must have shape (views, 5, 4, 4)")
    if weights.shape != targets.shape:
        raise ValueError("exact pose normals: weights and targets must have the same shape")
    _validate_backend(backend, detector, det_grid)
    poses, volume, targets, weights, pose_directions = (
        _real_fp32(x) for x in (poses, volume, targets, weights, pose_directions)
    )
    if backend == "pallas":
        from tomojax.core.pallas._pallas_trilinear import coefficients, normal_equations

        def coefficients_jvp(direction: jax.Array) -> jax.Array:
            return jax.jvp(lambda t: coefficients(t, grid, detector), (poses,), (direction,))[1]

        directions = jax.vmap(coefficients_jvp, in_axes=1)(pose_directions).swapaxes(0, 1)
        return normal_equations(
            coefficients(poses, grid, detector),
            volume,
            directions,
            targets,
            weights,
            grid,
            detector,
        )

    def one_view(
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pose, directions, target, weight = inputs

        def project(t: jax.Array) -> jax.Array:
            return forward_project_view_exact_T(t, grid, detector, volume, det_grid=det_grid)

        prediction = project(pose)
        residual = weight * (prediction - target)
        jacobian = jax.vmap(lambda dt: jax.jvp(project, (pose,), (dt,))[1])(directions)
        columns = (jacobian * weight).reshape((5, -1))
        gradient = jnp.matmul(columns, residual.ravel(), precision=jax.lax.Precision.HIGHEST)
        hessian = jnp.matmul(columns, columns.T, precision=jax.lax.Precision.HIGHEST)
        return 0.5 * jnp.vdot(residual, residual).real, gradient, hessian, weight * residual

    return jax.lax.map(one_view, (poses, pose_directions, targets, weights))
