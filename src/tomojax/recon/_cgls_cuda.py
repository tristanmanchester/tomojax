"""Host-array CGLS using the same linear Joseph pair without JAX startup/tracing."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from tomojax import progress
from tomojax.core import backend_policy, checkpoint
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

from ._cgls_stopping import (
    cancellation_magnitude,
    precision_stalled,
    stopping_record,
)
from ._data_term import project_residual
from .cgls import CGLSConfig, resident_bytes

if TYPE_CHECKING:
    from types import ModuleType

    import jax

    from tomojax.core.cuda_types import CuPyValue
    from tomojax.geometry import Detector, Geometry, Grid

    from .types import DataTerm


class DeviceOperator(Protocol):
    """Matched native projector and transpose in the solver's private volume order."""

    batch: int

    def forward(self, volume: CuPyValue) -> CuPyValue: ...
    def adjoint(self, images: CuPyValue) -> CuPyValue: ...


def available(
    geometry: Geometry,
    projections: jax.Array | np.ndarray,
    cfg: CGLSConfig,
    *,
    host_initial: bool,
    canonical: bool,
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
    ):
        return False
    return native_platform()


def native_platform() -> bool:
    """Whether native kernels may run on the selected device without starting JAX."""
    if os.environ.get("TOMOJAX_CUDA_KERNELS") == "0":
        return False
    if "jax" not in sys.modules and os.environ.get("JAX_DEFAULT_DEVICE"):
        return False
    platforms = os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")
    if "jax" in sys.modules:
        import jax

        platforms = jax.config.jax_platforms or jax.config.read("jax_platform_name")
        device = jax.config.jax_default_device
        if device is not None and (isinstance(device, str) or not backend_policy.cuda_backend()):
            return False
    if platforms and platforms.split(",")[0] not in {"cuda", "gpu"}:
        return False
    return importlib.util.find_spec("cupy") is not None


def array_module() -> ModuleType:
    """The optional native array module, loaded when a CUDA solve is selected."""
    import cupy as cp

    return cp


def native_device(cp: ModuleType) -> CuPyValue:
    """CuPy device matching the active reconstruction placement."""
    if "jax" in sys.modules:
        import jax

        device = jax.config.jax_default_device
        if device is not None and not isinstance(device, str):
            return cp.cuda.Device(device.local_hardware_id)
    return cp.cuda.Device()


def native_fits(cfg: CGLSConfig, grid: Grid, shape: tuple[int, ...], free: int | None) -> bool:
    """Whether the host-input CUDA path would run the entire resident solve."""
    return cfg.stream is False or (free is not None and resident_bytes(grid, shape) <= free)


def solve(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    initial: jax.Array | np.ndarray | None,
    cfg: CGLSConfig,
    *,
    target: float | None = None,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Return None only when CUDA is unavailable or the resident data would not fit."""
    cp = array_module()
    if not cp.cuda.is_available():
        return None
    with native_device(cp):
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
        _ = validate_pose_stack(poses, n, context="cgls")
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
        with checkpoint.loop() as saves:
            result, info = iterate(
                cp, ops, data, x, grid, cfg, saves, zero_start=initial is None, target=target
            )
        volume = np.ascontiguousarray(cp.asnumpy(result).transpose(2, 1, 0))
        del ops, data, x, result
    # The solve returns host arrays, so none of its device memory outlives it:
    # give the blocks back to the driver rather than keep them cached beside
    # JAX's allocator, where a later JAX call could not use them.
    cp.get_default_memory_pool().free_all_blocks()
    return volume, info


@functools.cache
def update_kernels(cp: CuPyValue) -> dict[str, CuPyValue]:
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


def _dot(cp: CuPyValue, u: CuPyValue, v: CuPyValue) -> CuPyValue:
    """The inner product of two arrays as a device scalar, without a product array."""
    return cp.dot(u.ravel(), v.ravel())


def _all(cp: CuPyValue, kernel: CuPyValue, *args: object) -> CuPyValue:
    """Whether ``kernel``'s condition holds at every element, as a device boolean."""
    flag = cp.ones(1, cp.int32)
    kernel(*args, flag)
    return cp.equal(flag[0], np.int32(1))


def _initial(
    cp: CuPyValue,
    ops: DeviceOperator,
    data: CuPyValue,
    x: CuPyValue,
    cfg: CGLSConfig,
    *,
    zero_start: bool,
) -> tuple[CuPyValue, CuPyValue, CuPyValue]:
    """The residual, normal gradient and its squared norm at the starting volume ``x``."""
    with progress.preparing("initialize cgls"):
        residual = data.copy() if zero_start else data - ops.forward(x)
        residual = project_residual(residual, cfg.data_term, xp=cp)
        gradient = ops.adjoint(project_residual(residual, cfg.data_term, xp=cp))
        update_kernels(cp)["negaxpy"](np.float32(cfg.damping) ** 2, x, gradient)
        return residual, gradient, _dot(cp, gradient, gradient)


class _Scalars(NamedTuple):
    """What a checkpoint keeps of the loop besides its arrays."""

    initial_norm: float
    count: int
    recomputations: int
    roundoff: bool
    verified: bool
    restart: bool
    margin: float | None

    def report(
        self,
        budget: int,
        gamma: float,
        tolerance: float,
        failed: bool | np.bool_,
        target: float | None,
        samples: int,
    ) -> dict[str, object]:
        return stopping_record(
            count=self.count,
            budget=budget,
            gamma=gamma,
            first_norm=self.initial_norm,
            tolerance=float(tolerance),
            failed=bool(failed),
            roundoff=self.roundoff,
            verified=self.verified,
            recomputations=self.recomputations,
            target=target,
            margin=self.margin,
            samples=samples,
        )


def iterate(
    cp: CuPyValue,
    ops: DeviceOperator,
    data: CuPyValue,
    x: CuPyValue,
    grid: Grid,
    cfg: CGLSConfig,
    saves: checkpoint.Loop,
    *,
    zero_start: bool,
    target: float | None,
) -> tuple[CuPyValue, dict[str, object]]:
    k, damp2 = update_kernels(cp), np.float32(cfg.damping) ** 2
    if saves.saved is None:
        residual, gradient, gamma = _initial(cp, ops, data, x, cfg, zero_start=zero_start)
        direction, previous_gamma = cp.empty_like(gradient), gamma
        margin = None if target is None else float(_dot(cp, residual, residual)) - target
        scalars = _Scalars(float(cp.sqrt(gamma)), 0, 0, False, True, True, margin)
    else:
        arrays, scalars = saves.saved.arrays, saves.saved.scalars(_Scalars)
        cp.copyto(x, cp.asarray(arrays[0]))
        direction, gradient, residual, gamma, previous_gamma = map(cp.asarray, arrays[1:])
    initial_norm, count, recomputations, roundoff, verified, restart, margin = scalars
    assert cfg.tolerance is not None
    assert cfg.iterations is not None
    tolerance = np.float32(cfg.absolute_tolerance + cfg.tolerance * initial_norm)
    failed = not np.isfinite(initial_norm)
    norm2, eps = float(gamma), np.finfo(np.float32).eps
    while (
        (margin is None or margin > 0)
        and count < cfg.iterations
        and norm2 > tolerance**2
        and not failed
        and not roundoff
    ):
        progress.check_canceled()
        if restart:
            cp.copyto(direction, gradient)
        else:
            k["xpay"](gradient, gamma / previous_gamma, direction)
        previous = norm2, verified
        gradient = None  # release the previous gradient before the transpose allocates the next
        with progress.preparing("first cgls update", count):
            step = _step(cp, ops, x, direction, residual, gamma, damp2, cfg.data_term, target)
        if step is None:
            failed = True
            break
        gradient, gamma_new, norm2, roundoff, small_update, margin = step
        del step  # hold the gradient only through ``gradient``
        count += 1
        failed = not np.isfinite(norm2)
        candidate = bool(norm2 <= tolerance**2)
        # Confirm a restart promptly; an accurate recurrence waits for the next period.
        periodic = (count % 16 == 0 or restart) and norm2 <= eps * initial_norm**2
        verified = roundoff or small_update or candidate or periodic
        restart = False
        if verified and not failed:
            residual, gradient, gamma_new, norm2, restart, roundoff, margin = _verify(
                cp, ops, data, x, gradient, gamma_new, damp2,
                data_term=cfg.data_term, candidate=candidate, roundoff=roundoff,
                previous=previous, target=target,
            )  # fmt: skip
            failed = not np.isfinite(norm2)
            recomputations += 1
        if failed:
            norm2 = float(gamma)
        previous_gamma, gamma = gamma, gamma_new
        if (task := progress.for_unit("iterations")) is not None:
            # The stopping statistics above already wait for the updated iterate.
            task.advance(count, {Metric.normal_residual_norm.name: float(np.sqrt(norm2))})
            progress.preview(lambda: planes(x, grid, reversed_axes=True))
        if not failed and saves.due():
            flags = bool(roundoff), bool(verified), bool(restart)
            state = _Scalars(initial_norm, count, recomputations, *flags, margin)
            saves.save((x, direction, gradient, residual, gamma, previous_gamma), state)
    final = _Scalars(
        initial_norm, count, recomputations, bool(roundoff), bool(verified), restart, margin
    )
    info = {
        **final.report(cfg.iterations, norm2, float(tolerance), failed, target, data.size),
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


def _step(
    cp: CuPyValue,
    ops: DeviceOperator,
    x: CuPyValue,
    direction: CuPyValue,
    residual: CuPyValue,
    gamma: CuPyValue,
    damp2: np.float32,
    data_term: DataTerm,
    target: float | None,
) -> tuple[CuPyValue, CuPyValue, float, bool, bool, float | None] | None:
    """Advance ``x`` and ``residual`` along ``direction`` in place.

    Returns the device gradient and its squared norm, the host norm and roundoff
    tests, and the optional discrepancy margin. None signals a step breakdown.
    """
    k = update_kernels(cp)
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
    gamma_new = _dot(cp, gradient, gradient)
    metrics = (gamma_new, roundoff, small_update)
    if target is not None:
        metrics += (_dot(cp, residual, residual),)
    stats = cp.asnumpy(cp.stack(metrics))
    margin = None if target is None else float(stats[3]) - target
    return gradient, gamma_new, float(stats[0]), bool(stats[1]), bool(stats[2]), margin


def _verify(
    cp: CuPyValue,
    ops: DeviceOperator,
    data: CuPyValue,
    x: CuPyValue,
    gradient: CuPyValue,
    gamma: CuPyValue,
    damp2: np.float32,
    *,
    data_term: DataTerm,
    candidate: bool,
    roundoff: bool,
    previous: tuple[float, bool | np.bool_],
    target: float | None,
) -> tuple[CuPyValue, CuPyValue, CuPyValue, float, bool, bool | np.bool_, float | None]:
    """Replace the recurrence with the measured residual before accepting a stop."""
    k = update_kernels(cp)
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
    margin = None if target is None else float(_dot(cp, residual, residual)) - target
    return residual, exact, exact_gamma, norm2, restart, roundoff or stalled, margin
