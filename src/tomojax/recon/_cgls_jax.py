"""Conjugate-gradient least-squares reconstruction with matched FP32 operators."""

from __future__ import annotations

from functools import partial
import math
import operator
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source
from tomojax.core.devices import ViewSplit, refuse_streaming, view_split
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_grid,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from tomojax.recon._cgls_host import solve_on_host
from tomojax.recon._cgls_stopping import precision_stalled, termination
from tomojax.recon._projection import (
    ConeModel,
    normal_equation_operators,
    projection_operators as _operators,
    resolve_geometry_projector,
)
from tomojax.recon._quadratic import gradient_energy, gradient_normal, regularization_normal
from tomojax.recon._volume_storage import (
    Model,
    initialize_volume,
    is_joseph,
    resident_model,
    restore_volume,
    voxel_spacing,
)
from tomojax.recon.cgls import CGLSConfig, resident_bytes, validate_model

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid


class _State(NamedTuple):
    iteration: jax.Array
    x: jax.Array
    residual: jax.Array
    direction: jax.Array
    gamma: jax.Array
    failed: jax.Array
    roundoff: jax.Array
    residual_verified: jax.Array
    residual_recomputations: jax.Array


def _running(state: _State, *, max_iters: jax.Array, threshold: jax.Array) -> jax.Array:
    """Whether the CGLS loop continues: budget left, tolerance unmet, no breakdown or roundoff."""
    return (
        (state.iteration < max_iters)
        & (state.gamma > threshold * threshold)
        & ~state.failed
        & ~state.roundoff
    )


def _checked_inputs(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    grid: Grid,
    model: Model,
) -> tuple[jax.Array, jax.Array]:
    # Check inputs inside the solve: separate eager checks would each compile.
    _, initial = initialize_volume(initial, model, grid)
    finite = (
        jnp.all(jnp.isfinite(data)) & jnp.all(jnp.isfinite(poses)) & jnp.all(jnp.isfinite(initial))
    )
    return initial, finite


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "zero_start",
        "model",
        "joseph_interpolation",
        "split",
    ),
)
def _solve(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    max_iters: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    zero_start: bool,
    model: str = "ray",
    joseph_interpolation: str = "linear",
    gradient_damping: jax.Array | None = None,
    split: ViewSplit | None = None,
) -> tuple[_State, jax.Array, jax.Array, jax.Array]:
    # Iteration budgets and changing data/poses are dynamic: budget sweeps and
    # repeated scans reuse the same compiled executable.
    forward, adjoint = _operators(
        poses, grid, detector, det_grid, backend, batch_size, model, joseph_interpolation,
        split=split,
    )  # fmt: skip
    magnitude_adjoint = adjoint
    if is_joseph(model) and joseph_interpolation == "cubic":
        # Cubic convolution has negative lobes. A.T @ abs(residual) is not
        # a cancellation bound: the roundoff estimate needs abs(A).T instead.
        _, magnitude_adjoint = _operators(
            poses,
            grid,
            detector,
            det_grid,
            backend,
            batch_size,
            model,
            joseph_interpolation,
            absolute_weights=True,
            split=split,
        )
    initial, inputs_finite = _checked_inputs(poses, data, initial, grid, model)
    residual = data if zero_start else data - forward(initial)
    damp2 = damping * damping
    spacing = voxel_spacing(model, grid)

    penalty_gradient = partial(
        regularization_normal,
        spacing=spacing,
        damping=damping,
        gradient_damping=gradient_damping,
    )

    gradient = adjoint(residual) - penalty_gradient(initial)
    gamma = jnp.sum(gradient * gradient)
    threshold = atol + rtol * jnp.sqrt(gamma)
    state = _State(
        jnp.int32(0),
        initial,
        residual,
        gradient,
        gamma,
        ~(inputs_finite & jnp.isfinite(gamma)),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.int32(0),
    )

    condition = partial(_running, max_iters=max_iters, threshold=threshold)

    def step(state: _State) -> _State:
        projected = forward(state.direction)
        direction_norm2 = jnp.sum(state.direction**2)
        denominator = jnp.sum(projected * projected) + damp2 * direction_norm2
        if gradient_damping is not None:
            denominator = denominator + gradient_damping**2 * gradient_energy(
                state.direction, spacing
            )
        valid = jnp.isfinite(denominator) & (denominator > 0)
        alpha = jnp.where(valid, state.gamma / jnp.where(valid, denominator, 1.0), 0.0)
        # A finite denominator implies a finite direction, so the update keeps a
        # finite iterate finite; never retaining the previous one saves a volume.
        stepped = valid
        x = jnp.where(valid, state.x + alpha * state.direction, state.x)
        residual = state.residual - alpha * projected
        gradient = adjoint(residual) - penalty_gradient(x)
        gamma_new = jnp.sum(gradient * gradient)
        valid &= jnp.isfinite(gamma_new)
        # A requested normal-residual tolerance can lie below attainable FP32
        # accuracy, especially for inconsistent damped systems. Continuing a
        # conjugate recurrence below roundoff can amplify arithmetic noise.
        # Check each voxel: a global volume norm can hide meaningful updates
        # to weak regions when another region contains much larger values.
        # Report this separately; do not silently relax the requested tolerance.
        roundoff = jnp.all(
            jnp.abs(alpha * state.direction)
            <= jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.abs(x), jnp.finfo(jnp.float32).tiny)
        )
        small_update = alpha * alpha * direction_norm2 <= jnp.finfo(jnp.float32).eps ** 2 * jnp.sum(
            x * x
        )
        convergence_candidate = gamma_new <= threshold * threshold
        # Atomic transpose reductions can keep a noisy conjugate recurrence
        # moving even after it reaches the FP32 cancellation floor. Check the
        # true residual periodically once its norm is small, before that noise
        # gets amplified. This does not relax the requested stopping tolerance.
        near_precision = gamma_new <= jnp.finfo(jnp.float32).eps * gamma
        check_due = ((state.iteration + 1) % 16 == 0) | state.residual_verified
        verify = roundoff | small_update | convergence_candidate | (check_due & near_precision)

        def recompute(
            _recurrence: jax.Array, gradient: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            # Replace the recurrence without retaining a second sinogram.
            prediction = forward(x)
            # A componentwise cancellation estimate keeps a bright, already
            # solved region from setting another region's precision limit.
            # It is an FP32 stagnation diagnostic, never a relaxed tolerance.
            cancellation = magnitude_adjoint(
                (data, prediction), lambda d, p: jnp.abs(d) + jnp.abs(p)
            ) + damp2 * jnp.abs(x)
            if gradient_damping is not None:
                cancellation = cancellation + gradient_damping**2 * gradient_normal(
                    jnp.abs(x), spacing, absolute_weights=True
                )
            exact_residual = data - prediction
            exact_gradient = adjoint(exact_residual) - penalty_gradient(x)
            exact_gamma = jnp.sum(exact_gradient * exact_gradient)
            bounded = jnp.all(
                jnp.abs(exact_gradient) <= 8 * jnp.finfo(jnp.float32).eps * cancellation
            )
            gap2 = jnp.sum((exact_gradient - gradient) ** 2)
            restart = convergence_candidate | roundoff | bounded | (gap2 > 0.01 * gamma_new)
            stalled = precision_stalled(exact_gamma, state.gamma, state.residual_verified, bounded)
            return exact_residual, exact_gradient, exact_gamma, restart, roundoff | stalled

        residual, gradient, gamma_new, restart, roundoff = jax.lax.cond(
            verify,
            recompute,
            lambda residual, gradient: (residual, gradient, gamma_new, jnp.bool_(False), roundoff),
            residual,
            gradient,
        )
        valid &= jnp.isfinite(gamma_new)
        # A large recurrence drift invalidates the conjugate directions. If the
        # recomputed norm fails the tolerance, continue with a fresh direction.
        direction = jnp.where(
            restart, gradient, gradient + (gamma_new / state.gamma) * state.direction
        )
        # At breakdown, report the last finite iterate and stop.
        return _State(
            state.iteration + stepped.astype(jnp.int32),
            x,
            residual,
            direction,
            jnp.where(valid, gamma_new, state.gamma),
            ~valid,
            jnp.where(valid, roundoff, state.roundoff),
            jnp.where(valid, verify, state.residual_verified),
            state.residual_recomputations + verify.astype(jnp.int32),
        )

    result = jax.lax.while_loop(condition, step, state)
    return result, jnp.sqrt(gamma), threshold, inputs_finite


@partial(
    jax.jit,
    static_argnames=("grid", "detector", "backend", "batch_size", "model", "joseph_interpolation"),
)
def _solve_streamed(
    poses: jax.Array,
    key: jax.Array,
    initial: jax.Array | None,
    max_iters: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str,
    joseph_interpolation: str,
    gradient_damping: jax.Array | None = None,
) -> tuple[_State, jax.Array, jax.Array, jax.Array]:
    """CG on the normal equations with data streamed from a host source.

    In exact arithmetic this is CGLS. The data-space residual is never stored:
    the normal residual follows the recurrence ``g -= alpha * (A^T A + R) d``,
    and every recomputation streams the measured views once to replace it with
    ``A^T (y - A x) - R x``, under the same verification rules as ``_solve``.
    Only volume-sized arrays and one view batch occupy the device.
    """
    normal, exact = normal_equation_operators(
        poses, grid, detector, backend, batch_size, model, joseph_interpolation
    )
    initial, x0 = initialize_volume(initial, model, grid)
    damp2 = damping * damping
    spacing = voxel_spacing(model, grid)
    penalty = partial(
        regularization_normal, spacing=spacing, damping=damping, gradient_damping=gradient_damping
    )
    data_gradient, _ = exact(initial, key)
    gradient = data_gradient - penalty(x0)
    gamma = jnp.sum(gradient * gradient)
    finite = jnp.all(jnp.isfinite(poses)) & jnp.all(jnp.isfinite(x0))
    threshold = atol + rtol * jnp.sqrt(gamma)
    state = _State(
        jnp.int32(0),
        x0,
        gradient,
        gradient,
        gamma,
        ~(finite & jnp.isfinite(gamma)),
        jnp.bool_(False),
        jnp.bool_(True),
        jnp.int32(0),
    )

    condition = partial(_running, max_iters=max_iters, threshold=threshold)

    def step(state: _State) -> _State:
        energy, applied = normal(state.direction)
        applied = applied + penalty(state.direction)
        direction_norm2 = jnp.sum(state.direction**2)
        denominator = energy + damp2 * direction_norm2
        if gradient_damping is not None:
            denominator = denominator + gradient_damping**2 * gradient_energy(
                state.direction, spacing
            )
        valid = jnp.isfinite(denominator) & (denominator > 0)
        alpha = jnp.where(valid, state.gamma / jnp.where(valid, denominator, 1.0), 0.0)
        x = jnp.where(valid, state.x + alpha * state.direction, state.x)
        gradient = state.residual - alpha * applied
        gamma_new = jnp.sum(gradient * gradient)
        roundoff = jnp.all(
            jnp.abs(alpha * state.direction)
            <= jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.abs(x), jnp.finfo(jnp.float32).tiny)
        )
        small_update = alpha * alpha * direction_norm2 <= jnp.finfo(jnp.float32).eps ** 2 * jnp.sum(
            x * x
        )
        convergence_candidate = gamma_new <= threshold * threshold
        near_precision = gamma_new <= jnp.finfo(jnp.float32).eps * gamma
        check_due = ((state.iteration + 1) % 16 == 0) | state.residual_verified
        verify = roundoff | small_update | convergence_candidate | (check_due & near_precision)

        def recompute(recurrence: jax.Array) -> tuple[jax.Array, ...]:
            data_gradient, cancellation = exact(x, key)
            exact_gradient = data_gradient - penalty(x)
            cancellation = cancellation + damp2 * jnp.abs(x)
            if gradient_damping is not None:
                cancellation = cancellation + gradient_damping**2 * gradient_normal(
                    jnp.abs(x), spacing, absolute_weights=True
                )
            exact_gamma = jnp.sum(exact_gradient * exact_gradient)
            bounded = jnp.all(
                jnp.abs(exact_gradient) <= 8 * jnp.finfo(jnp.float32).eps * cancellation
            )
            gap2 = jnp.sum((exact_gradient - recurrence) ** 2)
            restart = convergence_candidate | roundoff | bounded | (gap2 > 0.01 * gamma_new)
            stalled = precision_stalled(exact_gamma, state.gamma, state.residual_verified, bounded)
            return exact_gradient, exact_gamma, restart, roundoff | stalled

        gradient, gamma_new, restart, roundoff = jax.lax.cond(
            verify,
            recompute,
            lambda recurrence: (recurrence, gamma_new, jnp.bool_(False), roundoff),
            gradient,
        )
        valid &= jnp.isfinite(gamma_new)
        direction = jnp.where(
            restart, gradient, gradient + (gamma_new / state.gamma) * state.direction
        )
        return _State(
            state.iteration + valid.astype(jnp.int32),
            x,
            gradient,
            direction,
            jnp.where(valid, gamma_new, state.gamma),
            ~valid,
            jnp.where(valid, roundoff, state.roundoff),
            jnp.where(valid, verify, state.residual_verified),
            state.residual_recomputations + verify.astype(jnp.int32),
        )

    result = jax.lax.while_loop(condition, step, state)
    return result, jnp.sqrt(gamma), threshold, finite


def _as_float32(array: object) -> jax.Array:
    # Cast host arrays on the host; a device-side cast would compile a separate
    # program. device_put avoids the transient second device copy that
    # jnp.asarray makes of a host array.
    if isinstance(array, jax.Array):
        return array if array.dtype == jnp.float32 else array.astype(jnp.float32)
    return jax.device_put(np.asarray(array, dtype=np.float32))


def _placement(
    cfg: CGLSConfig,
    projections: jnp.ndarray | np.ndarray,
    n: int,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
    *,
    grid: Grid,
) -> tuple[ViewSplit | None, bool]:
    """How the projections are held: shared among devices, or streamed from the host."""
    split = view_split(cfg.devices, n)
    refuse_streaming(split, cfg.stream_projections, "cgls")
    if split is not None:
        return split, False  # each device holds its share
    if isinstance(projections, jax.Array) or det_grid is not None:
        return None, False
    if cfg.stream_projections is not None:
        return None, bool(cfg.stream_projections)
    if jax.default_backend() != "gpu":
        return None, False
    from tomojax.backends import device_free_memory_bytes

    free = device_free_memory_bytes()
    return None, free is not None and resident_bytes(grid, np.shape(projections)) > free


def _laid_out(
    split: ViewSplit | None,
    projections: jnp.ndarray | np.ndarray,
    poses: jax.Array,
    initial: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """The data, poses and start for :func:`_solve`: on one device, or on ``split``'s."""
    if split is None:
        return _as_float32(projections), poses, initial
    # Each device's views of the data; the poses and start on all, wherever they were.
    return split.place(projections), *split.everywhere((poses, initial))


def _validate_config(cfg: CGLSConfig) -> tuple[int, int]:
    """Validate iteration and regularization settings before selecting storage."""
    iterations, batch = operator.index(cfg.iterations), operator.index(cfg.views_per_batch)
    if iterations < 0 or batch < 1:
        raise ValueError("cgls: iterations must be nonnegative and views_per_batch positive")
    for name in ("rtol", "atol", "damping", "gradient_damping"):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"cgls: {name} must be finite and nonnegative")
    validate_model(cfg)
    return iterations, batch


def cgls(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | np.ndarray | None = None,
    config: CGLSConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    out: np.ndarray | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Reconstruct with CGLS and return volume plus convergence diagnostics.

    Inputs and the solve use FP32. The method supports arbitrary posed parallel
    rays, anisotropic voxels, detector offsets, and JAX explicit detector grids.
    It avoids operator-norm estimation and stores a constant number of volumes
    and sinograms, independent of iteration count. This public solver returns
    host diagnostics and is not a differentiable reconstruction layer.
    """
    cfg = CGLSConfig() if config is None else config
    iterations, batch = _validate_config(cfg)
    model, backend = resolve_geometry_projector(
        geometry, cfg.projector_model, cfg.backend,
        detector=detector, det_grid=det_grid, context="cgls",
    )  # fmt: skip
    if isinstance(model, ConeModel) and cfg.joseph_interpolation != "linear":
        raise ValueError("cgls: cone-beam geometry supports joseph_interpolation='linear' only")
    _ = validate_grid(grid, "cgls grid")
    n, _, _ = validate_projection_stack(projections, detector, geometry=geometry, context="cgls")
    validate_detector_grid(det_grid, detector, context="cgls")
    if init_x is not None:
        validate_volume(init_x, grid, context="cgls", name="init_x")
    split, stream = _placement(cfg, projections, n, det_grid, grid=grid)
    poses = stack_view_poses(geometry, n)
    validate_pose_stack(poses, n, context="cgls")
    hosted = solve_on_host(
        geometry, grid, detector, projections, init_x, cfg,
        model=model, backend=backend, poses=poses, shared=split is not None, out=out,
    )  # fmt: skip
    if hosted is not None:
        return hosted
    public_model = model
    model = resident_model(model, backend)
    initial = None if init_x is None else _as_float32(init_x)
    if is_joseph(model):
        from tomojax.core.joseph import validate_plane_geometry

        validate_plane_geometry(poses, grid, detector)
    budget = (jnp.int32(iterations), jnp.float32(cfg.rtol), jnp.float32(cfg.atol))
    options = {
        "grid": grid,
        "detector": detector,
        "backend": backend,
        "batch_size": batch,
        "model": model,
        "joseph_interpolation": cfg.joseph_interpolation,
        "gradient_damping": jnp.float32(cfg.gradient_damping) if cfg.gradient_damping else None,
    }
    if stream:
        with host_source(np.asarray(projections)) as key:
            result, initial_norm, threshold, inputs_finite = _solve_streamed(
                poses, key, initial, *budget, jnp.float32(cfg.damping), **options
            )
            jax.block_until_ready(result)
    else:
        data, poses, initial = _laid_out(split, projections, poses, initial)
        result, initial_norm, threshold, inputs_finite = _solve(
            poses, data, initial, det_grid, *budget, jnp.float32(cfg.damping),
            zero_start=initial is None, split=split, **options,
        )  # fmt: skip
    finite, count, gamma, failed, roundoff, first_norm, tolerance, verified, recomputations = (
        jax.device_get(
            (
                inputs_finite,
                result.iteration,
                result.gamma,
                result.failed,
                result.roundoff,
                initial_norm,
                threshold,
                result.residual_verified,
                result.residual_recomputations,
            )
        )
    )
    if not finite:
        raise ValueError("cgls: projections, initial volume and poses must be finite")
    converged = bool(not failed and gamma <= tolerance * tolerance)
    info = {
        "effective_iterations": int(count),
        "converged": converged,
        "termination": termination(
            failed=bool(failed), converged=converged, roundoff=bool(roundoff)
        ),
        "normal_residual_norm": float(np.sqrt(gamma)),
        "normal_residual_is_recomputed": bool(verified),
        "residual_recomputations": int(recomputations),
        "initial_normal_residual_norm": float(first_norm),
        "normal_residual_tolerance": float(tolerance),
        "projector_backend": backend,
        "projector_model": public_model,
        "joseph_interpolation": cfg.joseph_interpolation if is_joseph(model) else None,
        "views_per_batch": min(batch, n),
        "formulation": "streamed_normal_equations" if stream else "cgls",
        "damping": float(cfg.damping),
        "gradient_damping": float(cfg.gradient_damping),
    }
    volume = result.x if split is None else split.gather(result.x)
    return restore_volume(volume, model), info
