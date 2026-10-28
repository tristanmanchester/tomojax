"""Exact line integration of the zero-extended trilinear voxel basis.

The JAX reference supports ordinary differentiation. CUDA exposes an explicit
matched adjoint; its raw calls are not automatically differentiable. All
distances use the grid's physical units, and poses map object coordinates to
world coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.core._trilinear_reference import forward_project_view_exact_T
from tomojax.core.backend_policy import cuda_backend
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
    if backend not in {"jax", "cuda"}:
        raise ValueError("exact trilinear integration: backend must be 'jax' or 'cuda'")
    if backend == "cuda":
        if not cuda_backend():
            raise ValueError("exact trilinear integration: CUDA requires an NVIDIA CUDA device")
        from tomojax.core.cuda_projector._config import _ensure_canonical_detector_grid

        _ensure_canonical_detector_grid(detector, det_grid)


def exact_forward(
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    volume: jax.Array,
    *,
    backend: Literal["jax", "cuda"] = "jax",
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Integrate each parallel ray exactly within each trilinear interpolation cell.

    Rigid ``poses`` have shape ``(views, 4, 4)``. Output is FP32 with shape
    ``(views, detector.nv, detector.nu)``. Coordinates, including explicit detector
    grids, are evaluated in FP32. Exact refers to polynomial quadrature,
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
    if backend == "cuda":
        from tomojax.core._trilinear_cuda import coefficients, run

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
    backend: Literal["jax", "cuda"] = "jax",
    det_grid: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Apply the matched FP32 transpose, accumulating views into one volume."""
    n, _, _ = validate_projection_stack(images, detector, context="exact_adjoint")
    validate_pose_stack(poses, n, context="exact_adjoint")
    validate_grid(grid, "exact_adjoint")
    validate_detector_grid(det_grid, detector, context="exact_adjoint")
    _validate_backend(backend, detector, det_grid)
    poses, images = _real_fp32(poses), _real_fp32(images)
    if backend == "cuda":
        from tomojax.core._trilinear_cuda import coefficients, run

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
