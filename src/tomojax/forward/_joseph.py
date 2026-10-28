"""Differentiable plane-sampled parallel projection with explicit backend choice."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import cuda_backend
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
    if backend not in {"jax", "cuda"}:
        raise ValueError(f"{context}: backend must be 'jax' or 'cuda'")
    if backend == "cuda" and not cuda_backend():
        raise ValueError(f"{context}: backend='cuda' requires an NVIDIA CUDA device")
    return volume.astype(jnp.float32), poses.astype(jnp.float32)


def project_joseph(
    volume: jax.Array,
    poses: jax.Array,
    grid: Grid,
    detector: Detector,
    *,
    backend: Literal["jax", "cuda"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> jax.Array:
    """Project physical voxel-centre planes with zero-padded linear or cubic samples.

    Parameters
    ----------
    volume : jax.Array
        Finite real volume in (nz, ny, nx) order, converted to float32.
    poses : jax.Array
        Finite rigid world-from-object matrices, shape (views, 4, 4).
        Keep poses rigid when optimizing, for example by parameterizing rotations
        and translations. Shapes and static metadata are validated; array values
        must satisfy this contract, including under JIT.
    grid : Grid
        Physical voxel grid; its origin locates voxel (0, 0, 0)'s centre.
    detector : Detector
        Canonical physical detector grid, including detector offsets.
    backend : {'jax', 'cuda'}
        Explicit JAX reference or NVIDIA CUDA implementation; no fallback.
    interpolation : {'linear', 'cubic'}
        Bilinear or Keys cubic convolution (a=-1/2). Cubic uses a 4-by-4 stencil
        and can produce negative samples even for nonnegative inputs; it does
        not enforce positivity.

    Returns
    -------
    jax.Array
        Float32 projection stack, shape (views, detector.nv, detector.nu).

    Raises
    ------
    ValueError
        If shapes, interpolation or backend are unsupported, including CUDA without a GPU.

    Notes
    -----
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
    backend: Literal["jax", "cuda"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Return the Joseph projection loss and volume and matrix-pose gradients.

    Compute half the sum of squared residuals, without normalization, masking or
    fitted scale. Pose gradients are with respect to matrix entries: use a VJP
    through a rigid pose parameterization before taking an optimization step.
    The CUDA path provides first derivatives only; use JAX for higher derivatives.

    Parameters
    ----------
    volume : jax.Array
        Finite real volume of shape (grid.nz, grid.ny, grid.nx), converted to float32.
    poses : jax.Array
        Rigid world-from-object matrices of shape (views, 4, 4).
    target : jax.Array
        Finite real target projections of shape (views, detector.nv, detector.nu).
    grid : Grid
        Physical voxel grid; its origin locates voxel (0, 0, 0)'s center.
    detector : Detector
        Physical detector grid, including offsets, in the same length unit as grid.
    backend : {'jax', 'cuda'}, optional
        Explicit reference or NVIDIA CUDA implementation; default is 'jax', with no fallback.
    interpolation : {'linear', 'cubic'}, optional
        Bilinear or Keys cubic convolution (a=-1/2); default is 'linear'.

    Returns
    -------
    loss : jax.Array
        Scalar half-squared projection error.
    gradients : tuple of jax.Array
        Volume and matrix-pose gradients, with the same shapes as volume and poses.

    Raises
    ------
    ValueError
        If shapes, interpolation or backend are unsupported, including CUDA without a GPU.
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
    backend: Literal["jax", "cuda"] = "jax",
    interpolation: Literal["linear", "cubic"] = "linear",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return per-view Joseph loss, gradient, Gauss-Newton matrix and residual.

    For directional projection Jacobian J and residual r=prediction-target, return
    (0.5*r.T@r, J.T@r, J.T@J, r) per view. Shared parameters can sum these blocks.
    Priors, cross-view coupling, damping and gauge constraints are not included.
    CUDA provides first-order information only; use JAX for higher derivatives.

    Parameters
    ----------
    volume : jax.Array
        Finite real volume of shape (grid.nz, grid.ny, grid.nx), converted to float32.
    poses : jax.Array
        Rigid world-from-object matrices of shape (views, 4, 4).
    pose_directions : jax.Array
        Finite real matrix-pose tangents of shape (views, 4, 4, P), with
        1 <= P <= 16. Tangents use the caller's parameter units and should come
        from a rigid parameterization when optimizing physical poses.
    target : jax.Array
        Finite real target projections of shape (views, detector.nv, detector.nu).
    grid : Grid
        Physical voxel grid; its origin locates voxel (0, 0, 0)'s center.
    detector : Detector
        Physical detector grid, including offsets, in the same length unit as grid.
    backend : {'jax', 'cuda'}, optional
        Explicit reference or NVIDIA CUDA implementation; default is 'jax', with no fallback.
    interpolation : {'linear', 'cubic'}, optional
        Bilinear or Keys cubic convolution (a=-1/2); default is 'linear'.

    Returns
    -------
    loss : jax.Array
        Half-squared residual sums, shape (views,).
    gradient : jax.Array
        Directional gradients, shape (views, P).
    normal : jax.Array
        Gauss-Newton matrices, shape (views, P, P), not full loss Hessians.
    residual : jax.Array
        Prediction minus target, shape (views, detector.nv, detector.nu).

    Raises
    ------
    ValueError
        If shapes, interpolation or backend are unsupported, including CUDA without a GPU.
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
