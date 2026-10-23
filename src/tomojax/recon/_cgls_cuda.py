"""Host-array CGLS using the same linear Joseph pair without JAX startup/tracing."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import host_view_poses
from tomojax.core.joseph_cuda import JosephCUDA
from tomojax.core.validation import validate_grid, validate_pose_stack, validate_projection_stack

from ._cgls_stopping import precision_stalled, termination
from .cgls import CGLSConfig, resident_bytes


def available(geometry, projections, initial, cfg: CGLSConfig, det_grid) -> bool:
    """Whether the native host path supports this solve and the selected device."""
    if (
        not isinstance(projections, np.ndarray)
        or (initial is not None and not isinstance(initial, np.ndarray))
        or cfg.backend == "jax"
        or cfg.projector_model == "ray"
        or cfg.joseph_interpolation != "linear"
        or cfg.gradient_damping
        or cfg.devices is not None
        or cfg.stream_projections
        or det_grid is not None
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


def _poses(geometry, n):
    poses = host_view_poses(geometry, n)
    assert poses is not None
    validate_pose_stack(poses, n, context="cgls")
    rotations = poses[:, :3, :3]
    if not (
        np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-5)
        and np.allclose(np.linalg.det(rotations), 1.0, atol=2e-5)
        and np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=2e-5)
    ):
        raise ValueError("cgls: Joseph sampling requires rigid homogeneous poses")
    return poses


def solve(geometry, grid, detector, projections, initial, cfg: CGLSConfig):
    """Return None only when CUDA is unavailable or the resident data would not fit."""
    import cupy as cp

    if not cp.cuda.is_available():
        return None
    with _device(cp):
        if cfg.stream_projections is None:
            free, _total = cp.cuda.runtime.memGetInfo()
            # Freed arrays stay in CuPy's pool, not in the driver's free count.
            # A repeated solve can reuse them without falling back to JAX.
            pool = cp.get_default_memory_pool()
            if cp.cuda.get_allocator() == pool.malloc:
                free += pool.free_bytes()
            if resident_bytes(grid, projections.shape) > free:
                return None
        shape = validate_grid(grid, "cgls grid")[::-1]
        n, _, _ = validate_projection_stack(
            projections, detector, geometry=geometry, context="cgls"
        )
        if not np.isfinite((*grid_volume_origin(grid), *detector.center)).all():
            raise ValueError("cgls: Joseph grid origin and detector centre must be finite")
        poses = _poses(geometry, n)
        if not np.isfinite(projections).all() or (
            initial is not None and not np.isfinite(initial).all()
        ):
            raise ValueError("cgls: projections, initial volume and poses must be finite")
        if initial is not None and initial.shape != shape:
            raise ValueError("cgls: init_x must have the volume shape")
        ops = JosephCUDA(poses, grid, detector, views_per_batch=cfg.views_per_batch)
        data = cp.asarray(projections, dtype=cp.float32)
        x = (
            cp.zeros(shape, cp.float32)
            if initial is None
            else cp.array(initial, dtype=cp.float32, order="C")
        )
        result, info = _iterate(cp, ops, data, x, cfg, zero_start=initial is None)
        return cp.asnumpy(result), info


def _iterate(cp, ops, data, x, cfg: CGLSConfig, *, zero_start: bool):
    residual = data.copy() if zero_start else data - ops.forward(x)
    damp2 = np.float32(cfg.damping) ** 2
    # Explicit dense scalar ufuncs avoid importing CuPy sparse-operand dispatch.
    gradient = ops.adjoint(residual) - cp.multiply(damp2, x)
    gamma = cp.sum(gradient * gradient)
    initial_norm = float(cp.sqrt(gamma))
    tolerance = np.float32(cfg.atol + cfg.rtol * initial_norm)
    direction = gradient
    count, recomputations = 0, 0
    failed = not np.isfinite(initial_norm)
    roundoff, verified = False, True
    norm2 = float(gamma)
    eps = np.finfo(np.float32).eps
    while count < cfg.iterations and norm2 > tolerance**2 and not failed and not roundoff:
        previous = norm2, verified
        projected = ops.forward(direction)
        direction_norm2 = cp.sum(direction * direction)
        denominator = cp.sum(projected * projected) + cp.multiply(damp2, direction_norm2)
        value = float(denominator)
        if not np.isfinite(value) or value <= 0:
            failed = True
            break
        alpha = gamma / denominator
        x += alpha * direction
        residual -= alpha * projected
        gradient = ops.adjoint(residual) - cp.multiply(damp2, x)
        gamma_new = cp.sum(gradient * gradient)
        count += 1
        roundoff = cp.all(
            cp.abs(alpha * direction)
            <= cp.multiply(eps, cp.maximum(cp.abs(x), np.finfo(np.float32).tiny))
        )
        small_update = cp.square(alpha) * direction_norm2 <= cp.multiply(eps**2, cp.sum(x * x))
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
                candidate=candidate,
                roundoff=roundoff,
                previous=previous,
            )
            failed = not np.isfinite(norm2)
            recomputations += 1
        if failed:
            norm2 = float(gamma)
        else:
            direction = gradient if restart else gradient + (gamma_new / gamma) * direction
            gamma = gamma_new
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
        "projector_model": "joseph",
        "joseph_interpolation": "linear",
        "views_per_batch": ops.batch,
        "formulation": "cgls",
        "damping": float(cfg.damping),
        "gradient_damping": 0.0,
    }
    return x, info


def _verify(cp, ops, data, x, gradient, gamma, damp2, *, candidate: bool, roundoff: bool, previous):
    """Replace the recurrence with the measured residual before accepting a stop."""
    prediction = ops.forward(x)
    cancellation = ops.adjoint(cp.abs(data) + cp.abs(prediction)) + cp.multiply(damp2, cp.abs(x))
    residual = data - prediction
    exact = ops.adjoint(residual) - cp.multiply(damp2, x)
    exact_gamma = cp.sum(exact * exact)
    at_precision = cp.all(cp.abs(exact) <= cp.multiply(8 * np.finfo(np.float32).eps, cancellation))
    gap = cp.sum((exact - gradient) ** 2)
    checks = cp.asnumpy(cp.stack((exact_gamma, at_precision, gap)))
    norm2, at_precision, gap = float(checks[0]), bool(checks[1]), float(checks[2])
    restart = candidate or roundoff or at_precision or gap > 0.01 * float(gamma)
    stalled = precision_stalled(norm2, previous[0], previous[1], at_precision)
    return residual, exact, exact_gamma, norm2, restart, roundoff or stalled
