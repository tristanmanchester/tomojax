"""Conjugate-gradient least squares using the exact matched projector pair."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax.numpy as jnp
    import numpy as np

    from tomojax._typed_arrays import Device
    from tomojax.geometry import Detector, Geometry, Grid


@dataclass(frozen=True, kw_only=True)
class CGLSConfig:
    """Control an unconstrained least-squares solve.

    Minimize ``||A x - y||² + damping² ||x||² + gradient_damping² ||D x||²``
    using the discrete matched
    adjoint. Stop when the normal-residual norm falls below
    ``atol + rtol * initial_normal_residual_norm`` or after ``iterations`` updates.
    ``roundoff_limit`` indicates voxelwise update stagnation or a normal gradient
    within an FP32 cancellation estimate, without claiming the requested tolerance.
    Before reporting convergence or roundoff, recompute the data and normal
    residuals. A premature convergence indication from recursive residual drift
    restarts the conjugate direction. Periodic checks near the FP32 noise floor
    also catch recurrences kept moving by nondeterministic atomic reductions.
    This criterion measures stationarity, not reconstruction quality. No
    positivity clipping or TV regularization is applied. ``auto`` uses Pallas
    on CUDA with a canonical detector grid, otherwise JAX. An explicit Pallas
    request must be supported and never silently falls back.

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

    ``projector_model="joseph"`` samples voxel-centre planes along the dominant
    voxel direction, interpolates each plane and uses its matched transpose. On
    CUDA that transpose gathers into voxels without atomic writes. Bilinear
    interpolation is the default; ``joseph_interpolation="cubic"`` selects Keys
    cubic convolution with a=-1/2 and a 4-by-4 stencil, including negative
    weights. ``"ray"`` is the trilinear ray marcher, the only model supporting
    explicit detector grids. ``"auto"`` selects Joseph unless ``det_grid`` is given;
    both models match analytic line integrals equally well, and Joseph is faster.

    ``stream_projections`` solves the equivalent normal equations with NumPy or
    memmap projections read from host memory one view batch at a time, so only
    volume-sized arrays occupy the device; ``None`` streams when the stack would
    take more than 40% of free device memory. The normal residual then follows
    a recurrence between recomputations, each of which streams the data once.

    ``devices`` (two or more JAX devices) shares the views among them, each
    holding the whole volume, and sums their backprojections; the projections
    are then held in device memory, never streamed.
    """

    iterations: int = 50
    rtol: float = 1e-6
    atol: float = 0.0
    damping: float = 0.0
    views_per_batch: int = 64
    projector_backend: Literal["auto", "jax", "pallas"] = "auto"
    projector_model: Literal["auto", "ray", "joseph"] = "auto"
    joseph_interpolation: Literal["linear", "cubic"] = "linear"
    gradient_damping: float = 0.0
    stream_projections: bool | None = None
    devices: Device | Sequence[Device] | None = None  # kept as a tuple

    def __post_init__(self) -> None:
        if self.devices is not None:
            from tomojax.core.devices import as_devices

            object.__setattr__(self, "devices", as_devices(self.devices))


def validate_model(cfg: CGLSConfig) -> None:
    """Validate interpolation independently of the chosen execution runtime."""
    if cfg.joseph_interpolation not in {"linear", "cubic"}:
        raise ValueError("cgls: joseph_interpolation must be 'linear' or 'cubic'")
    if cfg.joseph_interpolation != "linear" and cfg.projector_model == "ray":
        raise ValueError("cgls: cubic interpolation requires projector_model='joseph'")


def cgls(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray | np.ndarray,
    *,
    init_x: jnp.ndarray | np.ndarray | None = None,
    config: CGLSConfig | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray | np.ndarray, dict[str, object]]:
    """Reconstruct with matched CGLS and return volume plus convergence diagnostics.

    Inputs and the solve use FP32. Host arrays on CUDA can use the native
    linear Joseph kernels and return a NumPy volume, avoiding JAX startup and
    tracing; device arrays, streaming, and other models use the compiled JAX
    solver. Both paths use the same stationarity and roundoff checks. This
    public solver returns host diagnostics and is not differentiable.
    """
    cfg = CGLSConfig() if config is None else config
    iterations, batch = operator.index(cfg.iterations), operator.index(cfg.views_per_batch)
    if iterations < 0 or batch < 1:
        raise ValueError("cgls: iterations must be nonnegative and views_per_batch positive")
    for name in ("rtol", "atol", "damping", "gradient_damping"):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"cgls: {name} must be finite and nonnegative")
    validate_model(cfg)
    if cfg.projector_backend not in {"auto", "jax", "pallas"}:
        raise ValueError("cgls: projector_backend must be 'auto', 'jax' or 'pallas'")
    if cfg.projector_model not in {"auto", "ray", "joseph"}:
        raise ValueError("cgls: projector_model must be 'auto', 'joseph' or 'ray'")
    from ._cgls_cuda import available, solve

    if available(geometry, projections, init_x, cfg, det_grid):
        result = solve(geometry, grid, detector, projections, init_x, cfg)
        if result is not None:
            return result
    from ._cgls_jax import cgls as solve_jax

    return solve_jax(
        geometry, grid, detector, projections, init_x=init_x, config=cfg, det_grid=det_grid
    )
