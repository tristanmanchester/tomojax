"""Host-array CGLS using the same linear Joseph pair without JAX startup/tracing."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys

import numpy as np

from tomojax import progress
from tomojax.core import backend_policy
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import host_view_poses
from tomojax.core.joseph_cuda import JosephCUDA
from tomojax.core.registration import planes
from tomojax.core.validation import (
    validate_physical_poses,
    validate_pose_stack,
    validate_projection_stack,
)
from tomojax.wire.metrics import Metric

from ._cgls_stopping import cancellation_magnitude, precision_stalled, termination
from ._data_term import project_residual
from .cgls import CGLSConfig, resident_bytes


def available(
    geometry, projections, cfg: CGLSConfig, *, host_initial: bool, canonical: bool
) -> bool:
    """Whether the native host path supports this solve and the selected device."""
    if (
        not isinstance(projections, np.ndarray)
        or not host_initial
        or cfg.backend == "jax"
        or cfg.interpolation != "linear"
        or cfg.gradient_damping
        or cfg.devices is not None
        or cfg.stream
        or not canonical
        or type(geometry) not in {ParallelGeometry, LaminographyGeometry}
        or os.environ.get("TOMOJAX_CUDA_KERNELS") == "0"
    ):
        return False
    if "jax" not in sys.modules and os.environ.get("JAX_DEFAULT_DEVICE"):
        return False
    platforms = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")
    if "jax" in sys.modules:
        import jax

        platforms = jax.config.jax_platforms or jax.config.read("jax_platform_name")
        device = jax.config.jax_default_device
        if device is not None and (
            isinstance(device, str)
            or device.platform != "gpu"
            or "cuda" not in device.client.platform_version.lower()
        ):
            return False
    if platforms and platforms.split(",")[0] not in {"cuda", "gpu"}:
        return False
    return importlib.util.find_spec("cupy") is not None


def _device(cp):
    if "jax" in sys.modules:
        import jax

        device = jax.config.jax_default_device
        if device is not None and not isinstance(device, str):
            return cp.cuda.Device(device.local_hardware_id)
    return cp.cuda.Device(cp.cuda.runtime.getDevice())


def native_fits(cfg: CGLSConfig, grid, shape, free: int | None) -> bool:
    """Whether the host-input CUDA path would run the entire resident solve."""
    return cfg.stream is False or (free is not None and resident_bytes(grid, shape) <= free)


def solve(geometry, grid, detector, projections, initial, cfg: CGLSConfig):
    """Return None only when CUDA is unavailable or the resident data would not fit."""
    import cupy as cp

    if not cp.cuda.is_available():
        return None
    with _device(cp):
        if not native_fits(
            cfg,
            grid,
            projections.shape,
            backend_policy.cupy_available_bytes() if cfg.stream is None else None,
        ):
            return None
        if progress.tracking_placement():
            progress.placed((f"gpu:{cp.cuda.runtime.getDevice()}",))
        shape = grid.shape
        n, _, _ = validate_projection_stack(
            projections, detector, geometry=geometry, context="cgls"
        )
        poses = host_view_poses(geometry, n)
        assert poses is not None
        validate_pose_stack(poses, n, context="cgls")
        validate_physical_poses(poses, "cgls", rigid=True)
        if not np.isfinite(projections).all() or (
            initial is not None and not np.isfinite(initial).all()
        ):
            raise ValueError("cgls: projections, initial volume and poses must be finite")
        if initial is not None and initial.shape != shape:
            raise ValueError("cgls: init_x must have the volume shape")
        # The iterate stays in private z-fast storage, read contiguously by every view.
        ops = JosephCUDA(poses, grid, detector, views_per_batch=cfg.views_per_batch, z_fast=True)
        data = cp.asarray(projections, dtype=cp.float32)
        x = (
            cp.zeros(shape[::-1], cp.float32)
            if initial is None
            else cp.array(initial.transpose(2, 1, 0), dtype=cp.float32, order="C")
        )
        result, info = _iterate(cp, ops, data, x, grid, cfg, zero_start=initial is None)
        volume = np.ascontiguousarray(cp.asnumpy(result).transpose(2, 1, 0))
        del ops, data, x, result
    # The solve returns host arrays, so none of its device memory outlives it:
    # give the blocks back to the driver rather than keep them cached beside
    # JAX's allocator, where a later JAX call could not use them.
    cp.get_default_memory_pool().free_all_blocks()
    return volume, info


@functools.cache
def _kernels(cp):
    """Fused updates and tests: no volume-sized temporaries beside the iterate.

    The solve's device memory is then the iterate, the direction and two
    gradients (the old one until the new one replaces it), within
    ``resident_bytes``, whatever memory another allocator holds. Inner
    products reduce flat views; an all-elements test clears a flag.
    """
    return {
        "axpy": cp.ElementwiseKernel("T a, T d", "T x", "x += a * d", "cgls_axpy"),
        "xpay": cp.ElementwiseKernel("T g, T b", "T d", "d = g + b * d", "cgls_xpay"),
        "negaxpy": cp.ElementwiseKernel("T c, T x", "T g", "g -= c * x", "cgls_negaxpy"),
        "absaxpy": cp.ElementwiseKernel("T c, T x", "T g", "g += c * abs(x)", "cgls_absaxpy"),
        "difference": cp.ElementwiseKernel("T u, T v", "T w", "w = u - v", "cgls_difference"),
        "tiny_step": cp.ElementwiseKernel(
            "T step, T d, T x, T e, T t",
            "raw int32 flag",
            "if (!(abs(step * d) <= e * max(abs(x), t))) flag[0] = 0",
            "cgls_tiny_step",
        ),
        "bounded": cp.ElementwiseKernel(
            "T g, T c, T e",
            "raw int32 flag",
            "if (!(abs(g) <= e * c)) flag[0] = 0",
            "cgls_bounded",
        ),
    }


def _dot(cp, u, v):
    """The inner product of two arrays as a device scalar, without a product array."""
    return cp.dot(u.ravel(), v.ravel())


def _all(cp, kernel, *args: object):
    """Whether ``kernel``'s condition holds at every element, as a device boolean."""
    flag = cp.ones(1, cp.int32)
    kernel(*args, flag)
    return cp.equal(flag[0], np.int32(1))


def _iterate(cp, ops, data, x, grid, cfg: CGLSConfig, *, zero_start: bool):
    with progress.preparing("initialise cgls"):
        k = _kernels(cp)
        residual = data.copy() if zero_start else data - ops.forward(x)
        residual = project_residual(residual, cfg.data_term, xp=cp)
        damp2 = np.float32(cfg.damping) ** 2
        gradient = ops.adjoint(project_residual(residual, cfg.data_term, xp=cp))
        k["negaxpy"](damp2, x, gradient)
        gamma = _dot(cp, gradient, gradient)
        initial_norm = float(cp.sqrt(gamma))
    tolerance = np.float32(cfg.absolute_tolerance + cfg.tolerance * initial_norm)
    count, recomputations = 0, 0
    failed = not np.isfinite(initial_norm)
    roundoff, verified = False, True
    norm2 = float(gamma)
    eps = np.finfo(np.float32).eps
    direction = cp.empty_like(gradient)
    restart, previous_gamma = True, gamma
    while count < cfg.iterations and norm2 > tolerance**2 and not failed and not roundoff:
        progress.check_cancelled()
        if restart:
            cp.copyto(direction, gradient)
        else:
            k["xpay"](gradient, gamma / previous_gamma, direction)
        previous = norm2, verified
        gradient = None  # release the previous gradient before the transpose allocates the next
        with progress.preparing("first cgls update", count):
            step = _step(cp, ops, x, direction, residual, gamma, damp2, cfg.data_term)
        if step is None:
            failed = True
            break
        gradient, gamma_new, roundoff, small_update = step
        del step  # hold the gradient only through ``gradient``
        count += 1
        stats = cp.asnumpy(cp.stack((gamma_new, roundoff, small_update)))
        norm2, roundoff, small_update = float(stats[0]), bool(stats[1]), bool(stats[2])
        failed = not np.isfinite(norm2)
        candidate = bool(norm2 <= tolerance**2)
        # Keep verifying after a near-precision check, not sixteen steps later.
        periodic = (count % 16 == 0 or previous[1]) and norm2 <= eps * initial_norm**2
        verified = roundoff or small_update or candidate or periodic
        restart = False
        if verified and not failed:
            residual, gradient, gamma_new, norm2, restart, roundoff = _verify(
                cp,
                ops,
                data,
                x,
                gradient,
                gamma_new,
                damp2,
                data_term=cfg.data_term,
                candidate=candidate,
                roundoff=roundoff,
                previous=previous,
            )
            failed = not np.isfinite(norm2)
            recomputations += 1
        if failed:
            norm2 = float(gamma)
        previous_gamma, gamma = gamma, gamma_new
        if (task := progress.for_unit("iterations")) is not None:
            # The stopping statistics above already wait for the updated iterate.
            task.advance(count, {Metric.normal_residual_norm.name: float(np.sqrt(norm2))})
            progress.preview(lambda: planes(x, grid, reversed_axes=True))
    converged = bool(not failed and norm2 <= tolerance**2)
    info = {
        "effective_iterations": count,
        "converged": converged,
        "termination": termination(failed=failed, converged=converged, roundoff=bool(roundoff)),
        "normal_residual_norm": float(np.sqrt(norm2)),
        "normal_residual_is_recomputed": bool(verified),
        "residual_recomputations": recomputations,
        "initial_normal_residual_norm": initial_norm,
        "normal_residual_tolerance": float(tolerance),
        "projector_backend": "cuda",
        "execution_backend": "cupy",
        "interpolation": "linear",
        "views_per_batch": ops.batch,
        "formulation": "cgls",
        "data_term": cfg.data_term,
        "background_unresolved": cfg.data_term == "per_view_contrast",
        "damping": float(cfg.damping),
        "gradient_damping": 0.0,
    }
    if count:
        progress.preview(lambda: planes(x, grid, reversed_axes=True), final=True)
    return x, info


def _step(cp, ops, x, direction, residual, gamma, damp2, data_term):
    """Advance ``x`` and ``residual`` along ``direction`` in place.

    Returns the new gradient, its squared norm and the two round-off tests as
    device values, or None when the step length breaks down.
    """
    k = _kernels(cp)
    eps, tiny = np.finfo(np.float32).eps, np.finfo(np.float32).tiny
    projected = project_residual(ops.forward(direction), data_term, xp=cp)
    direction_norm2 = _dot(cp, direction, direction)
    denominator = _dot(cp, projected, projected) + cp.multiply(damp2, direction_norm2)
    value = float(denominator)
    if not np.isfinite(value) or value <= 0:
        return None
    alpha = gamma / denominator
    k["axpy"](alpha, direction, x)
    k["axpy"](cp.negative(alpha), projected, residual)
    del projected
    gradient = ops.adjoint(project_residual(residual, data_term, xp=cp))
    k["negaxpy"](damp2, x, gradient)
    roundoff = _all(cp, k["tiny_step"], alpha, direction, x, eps, tiny)
    small_update = cp.square(alpha) * direction_norm2 <= cp.multiply(eps**2, _dot(cp, x, x))
    return gradient, _dot(cp, gradient, gradient), roundoff, small_update


def _verify(
    cp,
    ops,
    data,
    x,
    gradient,
    gamma,
    damp2,
    *,
    data_term,
    candidate: bool,
    roundoff: bool,
    previous,
):
    """Replace the recurrence with the measured residual before accepting a stop."""
    k = _kernels(cp)
    prediction = ops.forward(x)
    cancellation = ops.adjoint(cancellation_magnitude(data, prediction, data_term))
    k["absaxpy"](damp2, x, cancellation)
    residual = project_residual(data - prediction, data_term, xp=cp)
    del prediction
    exact = ops.adjoint(project_residual(residual, data_term, xp=cp))
    k["negaxpy"](damp2, x, exact)
    exact_gamma = _dot(cp, exact, exact)
    at_precision = _all(
        cp, k["bounded"], exact, cancellation, np.float32(8 * np.finfo(np.float32).eps)
    )
    k["difference"](exact, gradient, cancellation)  # the bound is used; reuse its storage
    gap = _dot(cp, cancellation, cancellation)
    del cancellation
    checks = cp.asnumpy(cp.stack((exact_gamma, at_precision, gap)))
    norm2, at_precision, gap = float(checks[0]), bool(checks[1]), float(checks[2])
    restart = candidate or roundoff or at_precision or gap > 0.01 * float(gamma)
    stalled = precision_stalled(norm2, previous[0], previous[1], at_precision)
    return residual, exact, exact_gamma, norm2, restart, roundoff or stalled
