"""Conjugate-gradient least-squares reconstruction with matched FP32 operators."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.backends.host_stream import host_source
from tomojax.core.devices import ViewSplit, announce, view_split
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.validation import (
    validate_detector_grid,
    validate_pose_stack,
    validate_projection_stack,
    validate_volume,
)
from tomojax.recon._cgls_host import solve_on_host
from tomojax.recon._cgls_stopping import (
    CGLSState,
    cancellation_magnitude,
    precision_stalled,
    running,
    stopping_record,
)
from tomojax.recon._chunks import Chunked
from tomojax.recon._data_term import project_residual
from tomojax.recon._projection import (
    normal_equation_operators,
    projection_operators as _operators,
    resolve_geometry_projector,
)
from tomojax.recon._quadratic import gradient_energy, gradient_normal, regularization_normal
from tomojax.recon._volume_storage import (
    initialize_volume,
    is_joseph,
    preview_planes,
    resident_model,
    restore_volume,
    voxel_spacing,
)
from tomojax.recon.cgls import CGLSConfig, resident_bytes
from tomojax.wire.metrics import Metric

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid

    from ._chunks import Drive, Metrics
    from ._volume_storage import Model
    from .types import DataTerm


def _verification(
    state: CGLSState,
    x: jax.Array,
    alpha: jax.Array,
    direction_norm2: jax.Array,
    gamma_new: jax.Array,
    gamma: jax.Array,
    threshold: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Check small steps, requested convergence and periodically the FP32 floor."""
    roundoff = jnp.all(
        jnp.abs(alpha * state.direction)
        <= jnp.finfo(jnp.float32).eps * jnp.maximum(jnp.abs(x), jnp.finfo(jnp.float32).tiny)
    )
    small_update = alpha * alpha * direction_norm2 <= jnp.finfo(jnp.float32).eps ** 2 * jnp.sum(
        x * x
    )
    convergence_candidate = gamma_new <= threshold * threshold
    near_precision = gamma_new <= jnp.finfo(jnp.float32).eps * gamma
    check_due = ((state.iteration + 1) % 16 == 0) | state.restarted
    verify = roundoff | small_update | convergence_candidate | (check_due & near_precision)
    return roundoff, convergence_candidate, verify


def _verified_gradient(
    exact_gradient: jax.Array,
    gradient: jax.Array,
    cancellation: jax.Array,
    state: CGLSState,
    gamma_new: jax.Array,
    candidate: jax.Array,
    roundoff: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compare the recomputed gradient with its recurrence and cancellation bound."""
    exact_gamma = jnp.sum(exact_gradient * exact_gradient)
    bounded = jnp.all(jnp.abs(exact_gradient) <= 8 * jnp.finfo(jnp.float32).eps * cancellation)
    gap2 = jnp.sum((exact_gradient - gradient) ** 2)
    restart = candidate | roundoff | bounded | (gap2 > 0.01 * gamma_new)
    stalled = precision_stalled(exact_gamma, state.gamma, state.residual_verified, bounded)
    return exact_gamma, restart, stalled


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "zero_start",
        "model",
        "interpolation",
        "split",
        "data_term",
    ),
)
def _solve_initial(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    tolerance: jax.Array,
    absolute_tolerance: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    zero_start: bool,
    model: str = "joseph",
    interpolation: str = "linear",
    gradient_damping: jax.Array | None = None,
    split: ViewSplit | None = None,
    data_term: DataTerm = "absolute",
    target: jax.Array | None = None,
) -> tuple[CGLSState, jax.Array, jax.Array, jax.Array]:
    # Iteration budgets and changing data/poses are dynamic: budget sweeps and
    # repeated scans reuse the same compiled executable.
    forward, adjoint = _operators(
        poses, grid, detector, det_grid, backend, batch_size, model, interpolation,
        split=split,
    )  # fmt: skip
    # Check inputs inside the solve: separate eager checks would each compile.
    _, initial = initialize_volume(initial, model, grid)
    inputs_finite = (
        jnp.all(jnp.isfinite(data)) & jnp.all(jnp.isfinite(poses)) & jnp.all(jnp.isfinite(initial))
    )
    residual = project_residual(data if zero_start else data - forward(initial), data_term)
    spacing = voxel_spacing(model, grid)

    penalty_gradient = partial(
        regularization_normal,
        spacing=spacing,
        damping=damping,
        gradient_damping=gradient_damping,
    )

    gradient = adjoint(residual, lambda r: project_residual(r, data_term)) - penalty_gradient(
        initial
    )
    gamma = jnp.sum(gradient * gradient)
    threshold = absolute_tolerance + tolerance * jnp.sqrt(gamma)
    state = CGLSState(
        jnp.int32(0),
        initial,
        residual,
        gradient,
        gamma,
        ~(inputs_finite & jnp.isfinite(gamma)),
        jnp.bool_(False),
        jnp.bool_(True),
        None if target is None else jnp.sum(residual * residual) - target,
        jnp.bool_(True),
        jnp.int32(0),
    )
    return state, gamma, threshold, inputs_finite


@partial(
    jax.jit,
    donate_argnames=("state",),
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "model",
        "interpolation",
        "split",
        "data_term",
    ),
)
def _solve(
    state: CGLSState,
    gamma: jax.Array,
    threshold: jax.Array,
    poses: jax.Array,
    data: jax.Array,
    det_grid: tuple[jax.Array, jax.Array] | None,
    max_iters: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str = "joseph",
    interpolation: str = "linear",
    gradient_damping: jax.Array | None = None,
    split: ViewSplit | None = None,
    data_term: DataTerm = "absolute",
    target: jax.Array | None = None,
) -> CGLSState:
    forward, adjoint = _operators(
        poses, grid, detector, det_grid, backend, batch_size, model, interpolation,
        split=split,
    )  # fmt: skip
    magnitude_adjoint = adjoint
    if is_joseph(model) and interpolation == "cubic":
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
            interpolation,
            absolute_weights=True,
            split=split,
        )
    damp2 = damping * damping
    spacing = voxel_spacing(model, grid)

    penalty_gradient = partial(
        regularization_normal,
        spacing=spacing,
        damping=damping,
        gradient_damping=gradient_damping,
    )

    condition = partial(running, max_iters=max_iters, threshold=threshold)

    def step(state: CGLSState) -> CGLSState:
        projected = project_residual(forward(state.direction), data_term)
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
        gradient = adjoint(residual, lambda r: project_residual(r, data_term)) - penalty_gradient(x)
        gamma_new = jnp.sum(gradient * gradient)
        valid &= jnp.isfinite(gamma_new)
        # A requested normal-residual tolerance can lie below attainable FP32
        # accuracy, especially for inconsistent damped systems. Continuing a
        # conjugate recurrence below roundoff can amplify arithmetic noise.
        # Check each voxel: a global volume norm can hide meaningful updates
        # to weak regions when another region contains much larger values.
        # Report this separately, as a roundoff limit.
        roundoff, convergence_candidate, verify = _verification(
            state, x, alpha, direction_norm2, gamma_new, gamma, threshold
        )

        def recompute(
            _recurrence: jax.Array, gradient: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            # Replace the recurrence without retaining a second sinogram.
            prediction = forward(x)

            # Keep the nonnegative bound separate from the centered operator.
            cancellation = magnitude_adjoint(
                (data, prediction), partial(cancellation_magnitude, data_term=data_term)
            ) + damp2 * jnp.abs(x)
            if gradient_damping is not None:
                cancellation = cancellation + gradient_damping**2 * gradient_normal(
                    jnp.abs(x), spacing, absolute_weights=True
                )
            exact_residual = project_residual(data - prediction, data_term)
            exact_gradient = adjoint(
                exact_residual, lambda r: project_residual(r, data_term)
            ) - penalty_gradient(x)
            exact_gamma, restart, stalled = _verified_gradient(
                exact_gradient,
                gradient,
                cancellation,
                state,
                gamma_new,
                convergence_candidate,
                roundoff,
            )
            return exact_residual, exact_gradient, exact_gamma, restart, roundoff | stalled

        residual, gradient, gamma_new, restart, roundoff = jax.lax.cond(
            verify,
            recompute,
            lambda residual, gradient: (residual, gradient, gamma_new, jnp.bool_(False), roundoff),
            residual,
            gradient,
        )
        valid &= jnp.isfinite(gamma_new)
        margin = None if target is None else jnp.sum(residual * residual) - target
        return state.advanced(
            x, residual, gradient, gamma_new, margin, valid=valid, stepped=stepped,
            restart=restart, verified=verify, roundoff=roundoff,
        )  # fmt: skip

    return jax.lax.while_loop(condition, step, state)


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "model",
        "interpolation",
        "data_term",
    ),
)
def _solve_streamed_initial(
    poses: jax.Array,
    key: jax.Array,
    initial: jax.Array | None,
    tolerance: jax.Array,
    absolute_tolerance: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str,
    interpolation: str,
    gradient_damping: jax.Array | None = None,
    data_term: DataTerm = "absolute",
    target: jax.Array | None = None,
) -> tuple[CGLSState, jax.Array, jax.Array, jax.Array]:
    """CG on the normal equations with data streamed from a host source.

    In exact arithmetic this is CGLS. The data-space residual is never stored:
    the normal residual follows the recurrence ``g -= alpha * (A^T A + R) d``,
    and every recomputation streams the measured views once to replace it with
    ``A^T (y - A x) - R x``, under the same verification rules as ``_solve``.
    Only volume-sized arrays and one view batch occupy the device.
    """
    _, exact = normal_equation_operators(
        poses,
        grid,
        detector,
        backend,
        batch_size,
        model,
        interpolation,
        data_term=data_term,
    )
    initial, x0 = initialize_volume(initial, model, grid)
    spacing = voxel_spacing(model, grid)
    penalty = partial(
        regularization_normal, spacing=spacing, damping=damping, gradient_damping=gradient_damping
    )
    data_gradient, _, energy = exact(initial, key)
    gradient = data_gradient - penalty(x0)
    gamma = jnp.sum(gradient * gradient)
    finite = jnp.all(jnp.isfinite(poses)) & jnp.all(jnp.isfinite(x0))
    threshold = absolute_tolerance + tolerance * jnp.sqrt(gamma)
    state = CGLSState(
        jnp.int32(0),
        x0,
        gradient,
        gradient,
        gamma,
        ~(finite & jnp.isfinite(gamma)),
        jnp.bool_(False),
        jnp.bool_(True),
        None if target is None else energy - target,
        jnp.bool_(True),
        jnp.int32(0),
    )
    return state, gamma, threshold, finite


@partial(
    jax.jit,
    donate_argnames=("state",),
    static_argnames=(
        "grid",
        "detector",
        "backend",
        "batch_size",
        "model",
        "interpolation",
        "data_term",
    ),
)
def _solve_streamed(
    state: CGLSState,
    gamma: jax.Array,
    threshold: jax.Array,
    poses: jax.Array,
    key: jax.Array,
    max_iters: jax.Array,
    damping: jax.Array,
    *,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: str,
    interpolation: str,
    gradient_damping: jax.Array | None = None,
    data_term: DataTerm = "absolute",
    target: jax.Array | None = None,
) -> CGLSState:
    normal, exact = normal_equation_operators(
        poses,
        grid,
        detector,
        backend,
        batch_size,
        model,
        interpolation,
        data_term=data_term,
    )
    damp2 = damping * damping
    spacing = voxel_spacing(model, grid)
    penalty = partial(
        regularization_normal, spacing=spacing, damping=damping, gradient_damping=gradient_damping
    )
    condition = partial(running, max_iters=max_iters, threshold=threshold)

    def step(state: CGLSState) -> CGLSState:
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
        roundoff, convergence_candidate, verify = _verification(
            state, x, alpha, direction_norm2, gamma_new, gamma, threshold
        )

        margin = state.discrepancy_margin
        if margin is not None:
            margin = margin - alpha * state.gamma
            verify |= margin <= 0

        def recompute(
            recurrence: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None]:
            data_gradient, cancellation, energy = exact(x, key)
            exact_gradient = data_gradient - penalty(x)
            cancellation = cancellation + damp2 * jnp.abs(x)
            if gradient_damping is not None:
                cancellation = cancellation + gradient_damping**2 * gradient_normal(
                    jnp.abs(x), spacing, absolute_weights=True
                )
            exact_gamma, restart, stalled = _verified_gradient(
                exact_gradient,
                recurrence,
                cancellation,
                state,
                gamma_new,
                convergence_candidate,
                roundoff,
            )
            return (
                exact_gradient,
                exact_gamma,
                restart,
                roundoff | stalled,
                None if target is None else energy - target,
            )

        gradient, gamma_new, restart, roundoff, margin = jax.lax.cond(
            verify,
            recompute,
            lambda recurrence: (recurrence, gamma_new, jnp.bool_(False), roundoff, margin),
            gradient,
        )
        valid &= jnp.isfinite(gamma_new)
        return state.advanced(
            x, gradient, gradient, gamma_new, margin, valid=valid, stepped=valid,
            restart=restart, verified=verify, roundoff=roundoff,
        )  # fmt: skip

    return jax.lax.while_loop(condition, step, state)


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
    if split is not None:
        return split, False  # each device holds its share
    if isinstance(projections, jax.Array) or det_grid is not None:
        return None, False
    if cfg.stream is not None:
        return None, bool(cfg.stream)
    if jax.default_backend() != "gpu":
        return None, False
    from tomojax.backends import device_free_memory_bytes
    from tomojax.core.devices import block_available

    free = device_free_memory_bytes()
    if free is None:
        return None, False
    # The resident solve's arrays come from one block that holds them all, or the data streams.
    needed = resident_bytes(grid, np.shape(projections))
    return None, needed > free or not block_available(needed)


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


def _cgls_program(
    poses: jax.Array,
    data: jax.Array,
    initial: jax.Array | None,
    det_grid: tuple[jax.Array, jax.Array] | None,
    iterations: int,
    tolerances: tuple[jax.Array, jax.Array],
    damping: jax.Array,
    *,
    streamed: bool,
    grid: Grid,
    detector: Detector,
    backend: str,
    batch_size: int,
    model: Model,
    interpolation: str,
    gradient_damping: jax.Array | None,
    data_term: DataTerm,
    split: ViewSplit | None,
    drive: Drive[CGLSState],
    target: jax.Array | None = None,
) -> tuple[CGLSState, jax.Array, jax.Array, jax.Array]:
    options = {
        "grid": grid,
        "detector": detector,
        "backend": backend,
        "batch_size": batch_size,
        "model": model,
        "interpolation": interpolation,
        "gradient_damping": gradient_damping,
        "data_term": data_term,
        "target": target,
    }
    if streamed:
        state, first_gamma, threshold, finite = _solve_streamed_initial(
            poses,
            data,
            initial,
            *tolerances,
            damping,
            **options,
        )

        def advance(state: CGLSState, end: int) -> CGLSState:
            return _solve_streamed(
                state, first_gamma, threshold, poses, data, end, damping, **options
            )
    else:
        state, first_gamma, threshold, finite = _solve_initial(
            poses,
            data,
            initial,
            det_grid,
            *tolerances,
            damping,
            zero_start=initial is None,
            split=split,
            **options,
        )

        def advance(state: CGLSState, end: int) -> CGLSState:
            return _solve(
                state,
                first_gamma,
                threshold,
                poses,
                data,
                det_grid,
                end,
                damping,
                split=split,
                **options,
            )

    def completed(state: CGLSState) -> tuple[int, bool, Metrics]:
        count, gamma, threshold_host, failed, roundoff = jax.device_get(
            (state.iteration, state.gamma, threshold, state.failed, state.roundoff)
        )
        return (
            int(count),
            bool(
                failed
                or roundoff
                or gamma <= threshold_host * threshold_host
                or (state.discrepancy_margin is not None and state.discrepancy_margin <= 0)
            ),
            {
                Metric.normal_residual_norm.name: float(np.sqrt(gamma)),
            },
        )

    return (
        drive(state, iterations, advance, completed, lambda s: preview_planes(s.x, model, grid)),
        first_gamma,
        threshold,
        finite,
    )


_run_cgls = Chunked(
    _cgls_program,
    static_argnames=(
        "streamed",
        "grid",
        "detector",
        "backend",
        "batch_size",
        "model",
        "interpolation",
        "data_term",
        "split",
    ),
)


def _solver_info(
    result: CGLSState,
    initial_gamma: jax.Array,
    threshold: jax.Array,
    inputs_finite: jax.Array,
    iterations: int,
    target: float | None,
    samples: int,
) -> dict[str, object]:
    """Transfer the final scalar diagnostics together while the volume stays on device."""
    finite, count, gamma, failed, roundoff, first_norm, tolerance, verified, recomputations = (
        jax.device_get(
            (
                inputs_finite,
                result.iteration,
                result.gamma,
                result.failed,
                result.roundoff,
                initial_gamma,
                threshold,
                result.residual_verified,
                result.residual_recomputations,
            )
        )
    )
    if not finite:
        raise ValueError("cgls: projections, initial volume and poses must be finite")
    return stopping_record(
        count=int(count),
        budget=iterations,
        gamma=float(gamma),
        first_norm=float(np.sqrt(first_norm)),
        tolerance=float(tolerance),
        failed=bool(failed),
        roundoff=bool(roundoff),
        verified=bool(verified),
        recomputations=int(recomputations),
        target=target,
        margin=result.discrepancy_margin,
        samples=samples,
    )


def cgls(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | np.ndarray | None = None,
    config: CGLSConfig,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    out: np.ndarray | None = None,
    target: float | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Reconstruct with CGLS and return volume plus convergence diagnostics.

    Inputs and the solve use FP32. The method supports arbitrary posed parallel
    rays, anisotropic voxels, detector offsets, and JAX explicit detector grids.
    It avoids operator-norm estimation and stores a constant number of volumes
    and sinograms, independent of iteration count. This public solver returns
    host diagnostics and is not a differentiable reconstruction layer.
    """
    cfg = config
    iterations, batch = cfg.iterations, cfg.views_per_batch
    model, backend = resolve_geometry_projector(
        geometry, cfg.backend,
        detector=detector, det_grid=det_grid, context="cgls",
    )  # fmt: skip
    n, _, _ = validate_projection_stack(projections, detector, geometry=geometry, context="cgls")
    validate_detector_grid(det_grid, detector, context="cgls")
    if init_x is not None:
        _ = validate_volume(init_x, grid, context="cgls", name="init_x")
    split, stream = _placement(cfg, projections, n, det_grid, grid=grid)
    announce(None if split is None else split.mesh.devices.flat, array=projections)
    poses = stack_view_poses(geometry, n)
    _ = validate_pose_stack(poses, n, context="cgls")
    hosted = solve_on_host(
        geometry, grid, detector, projections, init_x, cfg,
        model=model, backend=backend, poses=poses, shared=split is not None, out=out,
        det_grid=det_grid, target=target,
    )  # fmt: skip
    if hosted is not None:
        return hosted
    model = resident_model(model, backend)
    initial = None if init_x is None else _as_float32(init_x)
    if is_joseph(model):
        from tomojax.core.validation import validate_physical_poses

        validate_physical_poses(poses, "cgls", rigid=True)
    solve = partial(
        _run_cgls,
        grid=grid,
        detector=detector,
        backend=backend,
        batch_size=batch,
        model=model,
        interpolation=cfg.interpolation,
        data_term=cfg.data_term,
        gradient_damping=jnp.float32(cfg.gradient_damping) if cfg.gradient_damping else None,
        iterations=iterations,
        tolerances=(jnp.float32(cfg.tolerance), jnp.float32(cfg.absolute_tolerance)),
        damping=jnp.float32(cfg.damping),
        streamed=stream,
        split=split,
        target=target,
    )
    if stream:
        with host_source(np.asarray(projections)) as key:
            result, initial_gamma, threshold, inputs_finite = solve(poses, key, initial, det_grid)
            jax.block_until_ready(result)
    else:
        data, poses, initial = _laid_out(split, projections, poses, initial)
        result, initial_gamma, threshold, inputs_finite = solve(poses, data, initial, det_grid)
    assert iterations is not None
    info = {
        **_solver_info(
            result,
            initial_gamma,
            threshold,
            inputs_finite,
            iterations,
            target,
            n * detector.nu * detector.nv,
        ),
        "projector_backend": backend,
        "interpolation": cfg.interpolation if is_joseph(model) else None,
        "views_per_batch": min(batch, n),
        "formulation": "streamed_normal_equations" if stream else "cgls",
        "data_term": cfg.data_term,
        "background_unresolved": cfg.data_term == "per_view_contrast",
        "damping": float(cfg.damping),
        "gradient_damping": float(cfg.gradient_damping),
    }
    volume = result.x if split is None else split.gather(result.x)
    return restore_volume(volume, model), info
