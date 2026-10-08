"""Differentiable plane-sampled parallel projection with explicit backend choice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.joseph import (
    forward_project_planes,
    plane_coefficients,
    plane_l2_normal_equations,
    plane_l2_value_and_grad,
    validate_interpolation,
)
from tomojax.core.validation import validate_detector, validate_pose_stack, validate_volume

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid


def _inputs(volume, poses, grid, detector, backend):
    context = "Joseph projection"
    validate_volume(volume, grid, context=context)
    validate_detector(detector, context)
    if not np.isfinite((*grid_volume_origin(grid), *detector.center)).all():
        raise ValueError(f"{context}: grid origin and detector centre must be finite")
    if jnp.iscomplexobj(volume) or jnp.iscomplexobj(poses):
        raise ValueError(f"{context}: volume and poses must be real")
    # A dtype keeps jnp.asarray from staging a second device copy of a host array.
    volume = jnp.asarray(volume, dtype=jnp.float32)
    poses = jnp.asarray(poses, dtype=jnp.float32)
    if poses.ndim != 3 or poses.shape[0] < 1:
        raise ValueError(f"{context}: poses must have nonempty shape (views, 4, 4)")
    validate_pose_stack(poses, poses.shape[0], context=context)
    if backend not in {"jax", "pallas"}:
        raise ValueError(f"{context}: backend must be 'jax' or 'pallas'")
    if backend == "pallas" and (
        jax.default_backend() != "gpu" or "NVIDIA" not in jax.devices()[0].device_kind
    ):
        raise ValueError(f"{context}: backend='pallas' requires an NVIDIA CUDA device")
    return volume.astype(jnp.float32), poses.astype(jnp.float32)


def project_joseph(
    volume: jax.Array,
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> jax.Array:
    """Project physical voxel-centre planes with zero-padded linear or cubic samples.

    Args:
        volume: Finite real volume in ``(nx, ny, nz)`` order, converted to FP32.
        poses: Finite rigid world-from-object matrices with shape ``(views, 4, 4)``.
            Keep poses rigid when optimizing (for example, parameterize rotations
            and translations). Shapes and static metadata are validated; array
            values must satisfy this contract, including under JIT.
        grid: Physical voxel grid; its origin locates voxel (0, 0, 0)'s centre.
        detector: Canonical physical detector grid, including detector offsets.
        backend: Explicit JAX reference or NVIDIA CUDA implementation; no fallback.
        interpolation: Bilinear (default) or Keys cubic convolution (a=-1/2).
            Cubic uses a 4-by-4 stencil and can produce negative interpolated
            values even for nonnegative inputs; it does not enforce positivity.

    Returns:
        FP32 projection stack with shape ``(views, detector.nv, detector.nu)``.

    The default linear discretization matches ``cgls(projector_model='joseph')``. It is distinct
    from the default trilinear ray model. CUDA supports JIT, vmap, JVP, VJP,
    jacfwd, jacrev and transposing a linearization, with first-order derivatives
    only. Use the JAX reference for higher derivatives. Derivatives follow the
    selected dominant-axis branch and interpolation cell, as in the reference.
    Axis switches are not smooth; cubic interpolation has continuous first
    coordinate derivatives within a fixed dominant-axis branch.
    """
    validate_interpolation(interpolation)
    volume, poses = _inputs(volume, poses, grid, detector, backend)
    return forward_project_planes(
        plane_coefficients(poses, grid, detector),
        volume,
        grid,
        detector,
        backend=backend,
        interpolation=interpolation,
    )


def joseph_l2_value_and_grad(
    volume: jax.Array,
    poses: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Return half squared error and ``(volume_gradient, pose_gradient)``.

    Compute half the sum of squared projection residuals using the selected
    interpolation, without normalization, masking or fitted scale. Inputs follow
    ``project_joseph``'s contract; target is a finite real ``(views, nv, nu)`` stack. Pose gradients
    are with respect to the matrix entries: use a VJP through a rigid pose
    parameterization before taking an optimization step.

    The CUDA path fuses projection and pose derivatives, retaining a residual
    sinogram and tile reductions, then applies the matched volume transpose.
    This CUDA operation provides first derivatives only; the JAX reference can
    be differentiated further. For other losses, compose ``project_joseph``
    with ordinary JAX functions and differentiate it.
    """
    validate_interpolation(interpolation)
    volume, poses = _inputs(volume, poses, grid, detector, backend)
    target = jnp.asarray(target)
    expected = (poses.shape[0], detector.nv, detector.nu)
    if target.shape != expected or jnp.iscomplexobj(target):
        raise ValueError(f"Joseph loss: target must be real with shape {expected}")
    target = target.astype(jnp.float32)
    coeff, pullback = jax.vjp(lambda t: plane_coefficients(t, grid, detector), poses)
    loss, coefficient_grad, volume_grad = plane_l2_value_and_grad(
        coeff, volume, target, grid, detector, backend=backend, interpolation=interpolation
    )
    return loss, (volume_grad, pullback(coefficient_grad)[0])


def joseph_pose_normal_equations(
    volume: jax.Array,
    poses: jax.Array,
    pose_directions: jax.Array,
    target: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: Literal["jax", "pallas"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return per-view least-squares loss, gradient, Gauss-Newton matrix, residual.

    ``pose_directions`` has shape ``(views, 4, 4, P)``, with 1 <= P <= 16.
    Its last axis gives matrix-pose tangents for each view's parameters, in the
    caller's chosen units. For example, use ``vmap(jacfwd(make_pose))`` for a
    function that maps one view's rigid parameters to its world-from-object
    matrix. Inputs otherwise follow ``project_joseph``; target and directions
    must be finite and real, including under JIT. Directions should correspond
    to a rigid parameterization when optimizing physical poses.

    For residual ``r = prediction - target`` and directional projection Jacobian
    ``J``, return ``(0.5*r.T@r, J.T@r, J.T@J, r)`` independently per view. Shapes
    are ``(views,)``, ``(views, P)``, ``(views, P, P)``, and ``(views, nv, nu)``.
    The matrix is the Gauss-Newton approximation, not the full loss Hessian.
    Shared global parameters can sum these contributions; per-view parameters
    use separate blocks. Priors and cross-view coupling are not included.

    CUDA recomputes plane samples once, reduces tile-local columns and retains
    the residual for a subsequent line search. It stores neither the full
    detector-by-parameter Jacobian nor a ray-by-plane tape. This explicit CUDA
    operation provides first-order information only; use the JAX reference for
    further differentiation. No damping, parameter step or gauge is chosen.
    """
    validate_interpolation(interpolation)
    volume, poses = _inputs(volume, poses, grid, detector, backend)
    target = jnp.asarray(target)
    expected = (poses.shape[0], detector.nv, detector.nu)
    if target.shape != expected or jnp.iscomplexobj(target):
        raise ValueError(f"Joseph normal equations: target must be real with shape {expected}")
    directions = jnp.asarray(pose_directions)
    if (
        directions.ndim != 4
        or directions.shape[:3] != poses.shape
        or not 1 <= directions.shape[-1] <= 16
        or jnp.iscomplexobj(directions)
    ):
        raise ValueError(
            "Joseph normal equations: pose_directions must be real with shape (views, 4, 4, P), "
            "with 1 <= P <= 16"
        )
    coeff, linear = jax.linearize(lambda t: plane_coefficients(t, grid, detector), poses)
    coefficient_directions = jax.vmap(linear, in_axes=-1, out_axes=-1)(
        directions.astype(jnp.float32)
    )
    return plane_l2_normal_equations(
        coeff,
        volume,
        target.astype(jnp.float32),
        coefficient_directions,
        grid,
        detector,
        backend=backend,
        interpolation=interpolation,
    )
