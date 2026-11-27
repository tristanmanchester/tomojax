"""Conjugate-gradient least squares using the exact matched projector pair."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import operator
from typing import ClassVar, Literal, get_args

import numpy as np

from tomojax import _parameters, _typed_arrays, progress
from tomojax._parameters import check_backend
from tomojax.errors import InvalidParameter
from tomojax.geometry import Detector, Geometry, Grid

from .types import DataTerm


@dataclass(frozen=True, kw_only=True)
class CGLSConfig:
    """Control an unconstrained least-squares solve.

    Minimize ``||C (A x - y)||² + damping² ||x||² + gradient_damping² ||D x||²``
    using the discrete matched adjoint. ``C`` is the identity for absolute data
    and subtracts each full detector view's mean for per-view contrast.
    With ``iterations=None``, ``tolerance=None``, and zero ``absolute_tolerance``,
    ``damping``, and ``gradient_damping``, CGLS stops when the projection
    residual RMS reaches an estimate of detector noise plus sampling error,
    capped at 50 updates (on a one-pixel detector, only the numerical checks and
    50-update cap apply). Numerical convergence, roundoff, and breakdown checks
    still apply. Passing an explicit budget, tolerance, or penalty disables
    discrepancy stopping and uses numerical stopping alone: stop when the
    normal-residual norm falls below
    ``absolute_tolerance + tolerance * initial_normal_residual_norm`` or after
    ``iterations`` updates. On real scans, unmodeled errors such as motion,
    scatter, or calibration bias often keep the residual above the local
    estimate at 50 updates (``termination="iteration_limit"``).
    ``roundoff_limit`` indicates voxelwise update stagnation or a normal gradient
    stagnating within an FP32 cancellation estimate, without claiming the
    requested tolerance. Before reporting convergence or roundoff, CGLS
    recomputes the data and normal residuals and restarts the conjugate
    direction if recursive drift caused the trigger. Near the FP32 precision
    floor, periodic checks run every 16 updates: an accurate check resumes that
    16-update cadence, while a direction restart schedules a follow-up check on
    the next small residual. Numerical stationarity does not establish
    reconstruction quality, and running unpenalized CGLS for hundreds of updates
    fits noise and discretization error. No positivity clipping or TV
    regularization is applied. ``auto`` uses CUDA on CUDA with a canonical
    detector grid, otherwise JAX. An explicit CUDA request must be supported and
    never silently falls back.

    When both penalties are zero, updates lie in the range of the matched
    adjoint, up to floating-point error. A converged zero-start solve therefore
    selects the minimum Euclidean-norm least-squares solution. A supplied
    initial volume retains its component in the projector's null space:
    measurements cannot determine that component. A small residual alone does
    not establish image accuracy, particularly in sparse or tilted scans.

    ``D`` takes adjacent voxel differences divided by physical voxel spacing,
    with no edges outside the volume. ``gradient_damping`` adds quadratic
    smoothness; it does not impose positivity or penalize a constant volume.
    Both penalties default to zero. Their weights are independent of view count;
    the data term is an unnormalized sum of squared residuals.

    The projector samples voxel-center planes along the dominant voxel
    direction, interpolates each plane and uses its matched transpose; a rolled
    (affine) ``det_grid`` is carried in the plane coefficients. On CUDA the
    transpose gathers into voxels without atomic writes. Bilinear interpolation
    is the default; ``interpolation="cubic"`` selects Keys cubic convolution
    with a=-1/2 and a 4-by-4 stencil, including negative weights.

    ``stream`` solves the equivalent normal equations with NumPy or
    memmap projections read from host memory one view batch at a time, so only
    volume-sized arrays occupy the device; ``None`` reserves space for the
    resident solver state and verification scratch, streaming when they would
    exceed free device memory. The normal residual then follows
    a recurrence between recomputations, each of which streams the data once.

    ``devices`` (two or more JAX devices) shares the views among them, each
    holding the whole volume, and sums their backprojections; the projections
    are then held in device memory, never streamed.

    Parameters
    ----------
    exterior : {"cylinder", "zero"}, optional
        In scan reconstruction, "cylinder" estimates the contribution from outside
        the grid using cylinder-continued detector rows. "zero" fits the measured
        data directly. The analytic warm start uses the same setting.
    data_term : {"absolute", "per_view_contrast"}, optional
        Absolute least squares fits line integrals. "per_view_contrast" removes
        one constant per detector view and leaves the absolute level undetermined.
    iterations : int, optional
        Maximum number of updates. None enables statistical stopping on an
        unpenalized solve with implicit tolerances and caps work at 50 updates.
    tolerance : float, optional
        Relative normal-residual tolerance. None uses 1e-6 and allows
        statistical stopping. An explicit zero disables the relative test.
    absolute_tolerance : float, optional
        Nonnegative absolute tolerance on the normal-residual norm. Default is 0.0.
    damping : float, optional
        Nonnegative weight of the volume-norm penalty; squared in the objective.
        Default is 0.0.
    views_per_batch : int, optional
        Number of views processed together to bound temporary storage. Default is
        64.
    backend : Literal['auto', 'jax', 'cuda'], optional
        Execution backend: "auto", "jax" or "cuda". An explicit accelerator
        request never silently falls back. Default is 'auto'.
    interpolation : Literal['linear', 'cubic'], optional
        Plane interpolation: "linear" or Keys "cubic" with a=-1/2. Default is
        'linear'.
    gradient_damping : float, optional
        Nonnegative weight of the physical finite-difference penalty; squared in the
        objective. Default is 0.0.
    stream : bool | None, optional
        Read host projections one view batch at a time. None selects streaming from
        available device memory; False keeps them resident.
    devices : jax.Device or sequence of jax.Device or None, optional
        JAX device or sequence of devices. None uses the current default; multiple
        devices divide the views and each hold the full volume.
    """

    method: ClassVar[Literal["cgls"]] = "cgls"

    exterior: _parameters.Exterior = _parameters.DEFAULT_EXTERIOR
    data_term: DataTerm = "absolute"
    iterations: int | None = None
    tolerance: float | None = None
    absolute_tolerance: float = 0.0
    damping: float = 0.0
    views_per_batch: int = 64
    backend: Literal["auto", "jax", "cuda"] = "auto"
    interpolation: Literal["linear", "cubic"] = "linear"
    gradient_damping: float = 0.0
    stream: bool | None = None
    devices: _typed_arrays.Device | Sequence[_typed_arrays.Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        _parameters.check_exterior(self.exterior)
        iterations, batch = (
            operator.index(50 if self.iterations is None else self.iterations),
            operator.index(self.views_per_batch),
        )
        if iterations < 0 or batch < 1:
            raise InvalidParameter(
                "iterations" if iterations < 0 else "views_per_batch",
                "cgls: iterations must be nonnegative and views_per_batch positive",
            )
        for name in ("tolerance", "absolute_tolerance", "damping", "gradient_damping"):
            if name == "tolerance" and self.tolerance is None:
                continue
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise InvalidParameter(name, f"cgls: {name} must be finite and nonnegative")
        if self.data_term not in get_args(DataTerm):
            raise InvalidParameter(
                "data_term", "cgls: data_term must be absolute or per_view_contrast"
            )
        if self.interpolation not in ("linear", "cubic"):
            raise InvalidParameter(
                "interpolation", "cgls: interpolation must be 'linear' or 'cubic'"
            )
        check_backend(self.backend, "cgls")
        if self.devices is not None:
            from tomojax.core.devices import as_devices, refuse_streaming

            refuse_streaming(self.devices, self.stream, "cgls")

            object.__setattr__(self, "devices", as_devices(self.devices))


def check_cgls_geometry(geometry: Geometry, interpolation: str) -> None:
    """Require linear interpolation for the cone projector and its matched transpose."""
    from tomojax.core.geometry.cone import is_cone_beam

    if interpolation != "linear" and is_cone_beam(geometry):
        raise InvalidParameter(
            "config.interpolation", "cgls: cone-beam geometry supports interpolation='linear' only"
        )


def resident_bytes(grid: Grid, projection_shape: Sequence[int]) -> int:
    """Conservative FP32 resident CGLS state, including residual verification.

    Reserve six volumes for the iterate, direction, gradients, cancellation
    bound and update scratch; four sinograms for data, residual, prediction
    and recomputation scratch. Counting just the input misses the largest
    allocations, even when that input occupies well below half the device.
    Both execution runtimes use this bound before transferring host data.
    """
    volume = grid.nx * grid.ny * grid.nz
    return 4 * (6 * volume + 4 * math.prod(projection_shape))


@progress.reports_iterations(50)
def cgls(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: "_typed_arrays.JaxArray | np.ndarray",
    *,
    init_x: "_typed_arrays.JaxArray | np.ndarray | None" = None,
    config: CGLSConfig | None = None,
    det_grid: "tuple[_typed_arrays.JaxArray, _typed_arrays.JaxArray] | None" = None,
    out: np.ndarray | None = None,
) -> "tuple[_typed_arrays.JaxArray | np.ndarray, dict[str, object]]":
    """Reconstruct with matched CGLS and report convergence diagnostics.

    This solver fits the supplied data as given, and the workflows ``tj.reconstruct``
    and ``tj.reconstruct_slabs`` handle scans extending beyond the detector.

    Inputs and the solve use FP32. Host arrays on CUDA can use the native
    linear Joseph kernels and return a NumPy volume, avoiding JAX startup and
    tracing; device arrays, streaming, and other models use the compiled JAX
    solver. On axial parallel scans, CUDA linear Joseph solves (single-GPU,
    multi-GPU, and host-slab) use the separable axial CUDA Joseph projector
    pair; tilted and nonseparable geometries use the matched 3-D pair. Both paths use the same
    stationarity and roundoff checks. This public solver returns host diagnostics
    and is not differentiable.

    Cone-beam volumes, and CUDA linear Joseph parallel/laminography volumes,
    whose working set exceeds the device (or given ``out``) use host z slabs.
    The result is a C-ordered NumPy volume, ``out`` itself if given. If the
    four solver volumes exceed half of available host memory, automatically
    allocated storage uses unlinked temporary files in the standard temporary
    directory (controlled by ``TMPDIR``). The result owns its file until
    released. A memory-mapped ``out`` places the three work files beside it.

    Parameters
    ----------
    geometry : Geometry
        Scan geometry, including view angles and any per-view poses.
    grid : Grid
        Reconstruction grid in ``(x, y, z)`` order, with physical voxel spacing.
    detector : Detector
        Detector dimensions and placement in the same length unit as the grid.
    projections : array-like
        Measured line integrals with shape ``(views, detector.nv, detector.nu)``.
    init_x : array-like or None, optional
        Initial volume of shape ``(nz, ny, nx)``; None starts from zero.
    config : CGLSConfig or None, optional
        Solver settings. None uses CGLSConfig defaults.
    det_grid : tuple of array-like or None, optional
        Explicit detector-coordinate arrays ``(u, v)``, each of shape ``(nv, nu)``.
        None uses the canonical detector grid.
    out : numpy.ndarray or None, optional
        Writable float32 ``(nz, ny, nx)`` host storage for a supported host-slab result,
        including a memmap; it must not overlap the projections.

    Returns
    -------
    volume : jax.Array or numpy.ndarray
        Float32 reconstruction of shape ``(nz, ny, nx)``.
    info : dict
        Solver diagnostics and effective work: ``effective_iterations``,
        ``termination`` (``"projection_discrepancy"``, ``"iteration_limit"``,
        ``"converged"``, ``"roundoff_limit"``, or ``"numerical_breakdown"``),
        ``early_stop``, ``measurement_noise_rms``, ``projection_error_rms``,
        ``residual_rms`` (for discrepancy-stopped runs), and normal-residual
        diagnostics (``converged``, ``normal_residual_norm``,
        ``normal_residual_is_recomputed``, ``residual_recomputations``). A
        small residual or iteration count does not establish reconstruction
        accuracy.

    Raises
    ------
    ValueError
        If shapes, geometry or solver settings are incompatible with this method.

    Examples
    --------
    >>> import numpy as np
    >>> import tomojax as tj
    >>> from tomojax.recon import CGLSConfig, cgls
    >>> grid = tj.Grid(4, 4, 1, 1.0, 1.0, 1.0)
    >>> detector = tj.Detector(4, 1, 1.0, 1.0)
    >>> geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 8, endpoint=False))
    >>> data = tj.project(geometry, np.ones((1, 4, 4), np.float32))
    >>> volume, info = cgls(
    ...     geometry,
    ...     geometry.grid,
    ...     geometry.detector,
    ...     data,
    ...     config=CGLSConfig(iterations=1),
    ... )
    >>> volume.shape
    (1, 4, 4)
    """
    cfg = CGLSConfig() if config is None else config
    check_cgls_geometry(geometry, cfg.interpolation)
    discrepancy = None
    if (
        cfg.iterations is None
        and cfg.tolerance is None
        and not (cfg.absolute_tolerance or cfg.damping or cfg.gradient_damping)
    ):
        from tomojax.core.validation import validate_projection_stack

        from ._discrepancy import estimate

        _ = validate_projection_stack(projections, detector, geometry=geometry, context="cgls")
        discrepancy = estimate(
            projections,
            geometry,
            grid,
            detector,
            linear=cfg.interpolation == "linear" and det_grid is None,
            centered=cfg.data_term == "per_view_contrast",
        )
    cfg = replace(
        cfg,
        iterations=50 if cfg.iterations is None else cfg.iterations,
        tolerance=1e-6 if cfg.tolerance is None else cfg.tolerance,
    )
    target = None if discrepancy is None else discrepancy.target(math.prod(projections.shape))
    from ._cgls_cuda import available, solve

    result = None
    if out is None and available(
        geometry,
        projections,
        cfg,
        host_initial=init_x is None or isinstance(init_x, np.ndarray),
        canonical=det_grid is None,
    ):
        result = solve(geometry, grid, detector, projections, init_x, cfg, target=target)
    if result is None:
        from ._cgls_jax import cgls as solve_jax

        result = solve_jax(
            geometry, grid, detector, projections, init_x=init_x, config=cfg, det_grid=det_grid,
            out=out, target=target,
        )  # fmt: skip
    result[1]["measurement_noise_rms"] = None if discrepancy is None else discrepancy.noise
    return result
