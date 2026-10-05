"""Public FISTA/TV reconstruction adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import functools
import math
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.operator_norm import estimate_normal_norm
from tomojax.core.projector import (
    RAY_INTEGRATORS,
    backproject_view_T,
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.core.validation import (
    validate_grid,
    validate_optional_broadcastable_shape,
    validate_pose_stack,
    validate_projection_shape,
    validate_projection_stack,
    validate_volume,
)

from ._callbacks import LossCallback, emit_loss_callback_endpoints
from ._projection import (
    ProjectorBackend,
    ProjectorModel,
    least_squares_operators,
    normal_operator_norm,
    projection_operators,
    resolve_projector,
)
from ._tv_ops import (
    div3,
    grad3,
    huber_tv_grad,
    huber_tv_value,
    isotropic_tv_value,
    validate_regulariser,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tomojax.core.geometry.base import Detector, Geometry, Grid

    from .types import Regulariser

GradMode = Literal["auto", "batched", "stream"]


class FistaScanState(NamedTuple):
    """State carried through the FISTA scan loop."""

    x: jnp.ndarray
    z: jnp.ndarray
    t: jnp.ndarray
    loss: jnp.ndarray
    prev_obj: jnp.ndarray
    streak: jnp.ndarray
    done: jnp.ndarray
    has_prev: jnp.ndarray
    last_obj: jnp.ndarray
    iters_done: jnp.ndarray


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=["positivity", "lower_bound", "upper_bound"],
)
@dataclass(frozen=True)
class _FistaConstraints:
    positivity: bool
    lower_bound: float | None
    upper_bound: float | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _FistaRuntime:
    config: FistaConfig
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})
    constraints: _FistaConstraints
    # None starts from zeros created inside the compiled solve, so the start
    # is not a separate argument buffer alongside the iterates.
    x0: jnp.ndarray | None
    poses: jnp.ndarray
    lipschitz: float
    volume_mask: jnp.ndarray | None
    detector_grid: tuple[jnp.ndarray, jnp.ndarray] | None
    # (model, backend, views per batch) for the batched operators, or None for
    # the ray-model reference path.
    projector: tuple[str, str, int] | None = field(metadata={"static": True})


@dataclass(frozen=True)
class _FistaResult:
    volume: jnp.ndarray
    losses: jnp.ndarray
    lipschitz: float
    effective_iters: int
    early_stop: bool
    regulariser: Regulariser = field(metadata={"static": True})
    huber_delta: float = field(metadata={"static": True})

    def info(self) -> dict[str, object]:
        return {
            "loss": np.asarray(self.losses).tolist(),
            "L": self.lipschitz,
            "effective_iters": self.effective_iters,
            "early_stop": self.early_stop,
            "regulariser": self.regulariser,
            "huber_delta": self.huber_delta,
        }


def _effective_view_chunk_size(n_views: int, views_per_batch: int | None) -> int:
    requested = (
        int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else 1
    )
    return max(1, min(requested, int(n_views)))


def _view_chunk_schedule(
    i: jnp.ndarray,
    *,
    n_views: int,
    chunk_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    i = jnp.asarray(i, dtype=jnp.int32)
    b = jnp.int32(chunk_size)
    start = i * b
    remaining = jnp.maximum(0, jnp.int32(n_views) - start)
    valid = jnp.minimum(b, remaining)
    shift = b - valid
    start_shifted = jnp.maximum(0, start - shift)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    valid_mask = idx >= (b - valid)
    return start_shifted, valid_mask, start_shifted + idx


@dataclass
class FistaConfig:
    """Configuration for public FISTA/TV reconstruction.

    ``projector_model`` and ``projector_backend`` choose the projection operator as
    in :class:`CGLSConfig`; ``"auto"`` uses Joseph plane sampling with Pallas
    kernels on CUDA. An explicit detector grid or the exact ray integrator uses
    the ray-model reference path instead, which is the only path honouring
    ``gather_dtype``, ``projector_unroll``, ``checkpoint_projector`` and
    ``grad_mode``. ``views_per_batch=None`` uses 64 views per batch with batched
    operators and one view at a time on the reference path.
    """

    iters: int = 50
    lambda_tv: float = 0.005
    regulariser: Regulariser = "tv"
    huber_delta: float = 1e-2
    L: float | None = None
    views_per_batch: int | None = None
    projector_unroll: int = 1
    checkpoint_projector: bool = True
    gather_dtype: str = "fp32"
    grad_mode: GradMode = "auto"
    tv_prox_iters: int = 10
    recon_rel_tol: float | None = None
    recon_patience: int = 0
    power_iters: int = 5
    support: jnp.ndarray | None = None
    positivity: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    ray_integrator: str = "sampled"
    projector_model: ProjectorModel = "auto"
    projector_backend: ProjectorBackend = "auto"


jax.tree_util.register_dataclass(
    FistaConfig,
    data_fields=["support"],
    meta_fields=[field.name for field in fields(FistaConfig) if field.name != "support"],
)


def grad_data_term(  # noqa: PLR0915
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    x: jnp.ndarray,
    *,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: GradMode = "auto",
    ray_integrator: str = "sampled",
    T_all: jnp.ndarray | None = None,
    vol_mask: jnp.ndarray | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, float]:
    """Compute grad(1/2 sum_i ||A_i x - y_i||^2) and loss.

    Two execution modes:
    - batched: vmap over a chunk of views. Fast but higher peak memory.
    - stream: process one view at a time via lax.scan and explicit adjoint. Low peak memory.

    When grad_mode="auto", selects stream if the effective batch is 1, else batched.
    """
    validate_grid(grid, "grad_data_term grid")
    n_views, nv, nu = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="grad_data_term projections",
    )
    validate_volume(x, grid, context="grad_data_term", name="x")
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="grad_data_term support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="grad_data_term geometry")

    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid
    mask_arr = None if vol_mask is None else jnp.asarray(vol_mask, dtype=jnp.float32)

    def apply_mask(vol: jnp.ndarray) -> jnp.ndarray:
        return vol * mask_arr if mask_arr is not None else vol

    def adjoint(resid: jnp.ndarray, T_i: jnp.ndarray) -> jnp.ndarray:
        grad_i = backproject_view_T(
            T_i,
            grid,
            detector,
            resid,
            unroll=int(projector_unroll),
            gather_dtype=gather_dtype,
            det_grid=det_grid,
            ray_integrator=ray_integrator,
        )
        return grad_i if mask_arr is None else grad_i * mask_arr

    def batched_loss_and_grad(vol: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Loss/grad over views batched in chunks, using a scan to keep jaxpr compact.

        We pad the last chunk to size ``b`` and mask it out in the reduction so the
        compiled graph is a single-sized loop body regardless of the number of chunks.
        """
        masked_vol = vol * vol_mask if vol_mask is not None else vol
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
            ),
            in_axes=(0, None),
        )
        n = int(T_all.shape[0])
        nv = int(projections.shape[1])
        nu = int(projections.shape[2])
        b = _effective_view_chunk_size(n, views_per_batch)
        m = (n + b - 1) // b

        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray],
            i: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            loss_acc, grad_acc = carry
            start_shifted, valid_mask, _view_idx = _view_chunk_schedule(
                i,
                n_views=n,
                chunk_size=b,
            )
            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
            pred = vm_project(T_chunk, masked_vol)
            mask = valid_mask[:, None, None]
            resid = (pred - y_chunk).astype(jnp.float32) * mask
            loss_batch = 0.5 * jnp.vdot(resid, resid).real
            grad_batch = sum_backproject_views_T(
                T_chunk,
                grid,
                detector,
                resid,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
            )
            if mask_arr is not None:
                grad_batch = grad_batch * mask_arr
            return ((loss_acc + loss_batch, grad_acc + grad_batch), None)

        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, grad_tot), _ = jax.lax.scan(body, init, jnp.arange(m))
        return loss_tot, grad_tot

    def stream_loss_and_grad(vol: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(
            carry: tuple[jnp.ndarray, jnp.ndarray],
            i: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
            loss_acc, g_acc = carry
            T_i = jax.lax.dynamic_slice(T_all, (i, 0, 0), (1, 4, 4))[0]
            y_i = jax.lax.dynamic_slice(projections, (i, 0, 0), (1, nv, nu))[0]
            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
            )
            resid_i = (pred_i - y_i).astype(jnp.float32)
            loss_i = 0.5 * jnp.vdot(resid_i, resid_i).real
            g_i = adjoint(resid_i, T_i)
            return (loss_acc + loss_i, g_acc + g_i), None

        init = (jnp.float32(0.0), jnp.zeros_like(vol))
        (loss_tot, g_tot), _ = jax.lax.scan(one_view, init, jnp.arange(T_all.shape[0]))
        return loss_tot, g_tot

    # Select execution mode
    eff_b = (
        int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else 1
    )
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        loss_val, grad = stream_loss_and_grad(x)
        return grad, loss_val
    loss_val, grad = batched_loss_and_grad(x)
    return grad, loss_val


def data_term_value(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    x: jnp.ndarray,
    *,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: GradMode = "auto",
    ray_integrator: str = "sampled",
    T_all: jnp.ndarray | None = None,
    vol_mask: jnp.ndarray | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Compute the data term ``1/2 Σ_i ||A_i x - y_i||^2`` without its gradient."""
    validate_grid(grid, "data_term_value grid")
    n_views, nv, nu = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="data_term_value projections",
    )
    validate_volume(x, grid, context="data_term_value", name="x")
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="data_term_value support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="data_term_value geometry")

    det_grid = get_detector_grid_device(detector) if det_grid is None else det_grid

    def batched_loss(vol: jnp.ndarray) -> jnp.ndarray:
        masked_vol = vol * vol_mask if vol_mask is not None else vol
        vm_project = jax.vmap(
            lambda T, v: forward_project_view_T(
                T,
                grid,
                detector,
                v,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
            ),
            in_axes=(0, None),
        )
        n = int(T_all.shape[0])
        b = _effective_view_chunk_size(n, views_per_batch)
        m = (n + b - 1) // b

        def body(
            loss_acc: jnp.ndarray,
            i: jnp.ndarray,
        ) -> tuple[jnp.ndarray, None]:
            start_shifted, valid_mask, _view_idx = _view_chunk_schedule(
                i,
                n_views=n,
                chunk_size=b,
            )
            T_chunk = jax.lax.dynamic_slice(T_all, (start_shifted, 0, 0), (b, 4, 4))
            y_chunk = jax.lax.dynamic_slice(projections, (start_shifted, 0, 0), (b, nv, nu))
            pred = vm_project(T_chunk, masked_vol)
            mask = valid_mask[:, None, None]
            resid = (pred - y_chunk).astype(jnp.float32) * mask
            loss_batch = 0.5 * jnp.vdot(resid, resid).real
            return (loss_acc + loss_batch, None)

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(body, loss0, jnp.arange(m))
        return loss_tot

    def stream_loss(vol: jnp.ndarray) -> jnp.ndarray:
        masked_vol = vol * vol_mask if vol_mask is not None else vol

        def one_view(
            loss_acc: jnp.ndarray,
            i: jnp.ndarray,
        ) -> tuple[jnp.ndarray, None]:
            T_i = jax.lax.dynamic_slice(T_all, (i, 0, 0), (1, 4, 4))[0]
            y_i = jax.lax.dynamic_slice(projections, (i, 0, 0), (1, nv, nu))[0]

            pred_i = forward_project_view_T(
                T_i,
                grid,
                detector,
                masked_vol,
                use_checkpoint=checkpoint_projector,
                unroll=int(projector_unroll),
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                ray_integrator=ray_integrator,
            )
            resid_i = (pred_i - y_i).astype(jnp.float32)
            loss_i = 0.5 * jnp.vdot(resid_i, resid_i).real
            return loss_acc + loss_i, None

        loss0 = jnp.float32(0.0)
        loss_tot, _ = jax.lax.scan(one_view, loss0, jnp.arange(T_all.shape[0]))
        return loss_tot

    eff_b = (
        int(views_per_batch) if (views_per_batch is not None and int(views_per_batch) > 0) else 1
    )
    mode = grad_mode
    if grad_mode == "auto":
        mode = "stream" if eff_b <= 1 else "batched"

    if mode == "stream":
        return stream_loss(x)
    return batched_loss(x)


def power_method_L(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections_shape: tuple[int, int, int],
    *,
    iters: int = 10,
    views_per_batch: int | None = None,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    grad_mode: GradMode = "auto",
    ray_integrator: str = "sampled",
    T_all: jnp.ndarray | None = None,
    vol_mask: jnp.ndarray | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> float:
    """Estimate the Lipschitz constant of the data gradient by power iteration."""
    validate_grid(grid, "power_method_L grid")
    n_views, _, _ = validate_projection_shape(
        projections_shape,
        detector,
        geometry=geometry,
        context="power_method_L projections_shape",
    )
    validate_optional_broadcastable_shape(
        vol_mask,
        (grid.nx, grid.ny, grid.nz),
        context="power_method_L support",
        name="vol_mask",
        fix="use a volume mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if T_all is None:
        T_all = stack_view_poses(geometry, n_views)
    validate_pose_stack(T_all, n_views, context="power_method_L geometry")
    batch_size = (
        1 if grad_mode == "stream" else _effective_view_chunk_size(n_views, views_per_batch)
    )
    initial = jnp.ones((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    return float(
        estimate_normal_norm(
            T_all,
            initial,
            det_grid,
            vol_mask,
            grid=grid,
            detector=detector,
            batch_size=batch_size,
            iters=int(iters),
            unroll=int(projector_unroll),
            checkpoint=checkpoint_projector,
            gather_dtype=gather_dtype,
            ray_integrator=ray_integrator,
        )
    )


def tv_proximal(x: jnp.ndarray, lam_over_L: float, iters: int = 20) -> jnp.ndarray:
    """Approximate the isotropic TV proximal by projected gradient on its dual.

    Solve ``min_u 0.5 ||u - x||^2 + lam TV(u)`` through the dual field ``p``
    with ``|p| <= lam`` pointwise and ``u = x + div p`` (Chambolle 2004), using
    the guaranteed step ``1 / ||div||^2 = 1 / 12``. Only the three dual
    components persist between iterations, so the prox holds about four volumes.
    """
    lam = jnp.asarray(lam_over_L, dtype=x.dtype)
    tau = jnp.asarray(1.0 / 12.0, dtype=x.dtype)
    eps = jnp.asarray(jnp.finfo(x.dtype).eps, dtype=x.dtype)

    def prox_impl(lam_val: jnp.ndarray) -> jnp.ndarray:
        lam_safe = jnp.maximum(lam_val, eps)

        def body(
            p: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _: object
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
            gx, gy, gz = grad3(x + div3(*p))
            q1, q2, q3 = p[0] + tau * gx, p[1] + tau * gy, p[2] + tau * gz
            shrink = jnp.maximum(1.0, jnp.sqrt(q1 * q1 + q2 * q2 + q3 * q3) / lam_safe)
            return (q1 / shrink, q2 / shrink, q3 / shrink), None

        zeros = jnp.zeros_like(x)
        p, _ = jax.lax.scan(body, (zeros, zeros, zeros), None, length=int(iters))
        return x + div3(*p)

    return jax.lax.cond(lam > 0, prox_impl, lambda _: x, lam)


def _normalize_constraint_config(cfg: FistaConfig) -> tuple[bool, float | None, float | None]:
    """Validate FISTA feasibility constraints and return scalar bounds."""
    lower = None if cfg.lower_bound is None else float(cfg.lower_bound)
    upper = None if cfg.upper_bound is None else float(cfg.upper_bound)

    if lower is not None and not math.isfinite(lower):
        raise ValueError("fista_tv constraints: lower_bound must be finite when provided")
    if upper is not None and not math.isfinite(upper):
        raise ValueError("fista_tv constraints: upper_bound must be finite when provided")

    effective_lower = lower
    if bool(cfg.positivity):
        effective_lower = max(0.0, lower) if lower is not None else 0.0

    if upper is not None and effective_lower is not None and upper < effective_lower:
        raise ValueError(
            "fista_tv constraints: upper_bound must be greater than or equal to "
            "the effective lower bound"
        )

    return bool(cfg.positivity), lower, upper


def _project_constraints(
    x: jnp.ndarray,
    *,
    positivity: bool,
    lower_bound: float | None,
    upper_bound: float | None,
) -> jnp.ndarray:
    """Project a volume onto optional elementwise physical constraints."""
    if lower_bound is not None:
        x = jnp.maximum(x, jnp.asarray(lower_bound, dtype=x.dtype))
    if positivity:
        x = jnp.maximum(x, jnp.asarray(0.0, dtype=x.dtype))
    if upper_bound is not None:
        x = jnp.minimum(x, jnp.asarray(upper_bound, dtype=x.dtype))
    return x


def _prepare_fista_runtime(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    init_x: jnp.ndarray | None,
    config: FistaConfig | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> _FistaRuntime:
    cfg = FistaConfig() if config is None else config
    if cfg.ray_integrator not in RAY_INTEGRATORS:
        raise ValueError("ray_integrator must be sampled or exact")
    volume_mask = cfg.support
    regulariser = validate_regulariser(
        cfg.regulariser,
        cfg.huber_delta,
        context="fista_tv config",
    )
    huber_delta = float(cfg.huber_delta)
    positivity, lower_bound, upper_bound = _normalize_constraint_config(cfg)
    constraints = _FistaConstraints(
        positivity=positivity,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    constraints_enabled = positivity or lower_bound is not None or upper_bound is not None

    validate_grid(grid, "fista_tv grid")
    n_views, _, _ = validate_projection_stack(
        projections,
        detector,
        geometry=geometry,
        context="fista_tv projections",
    )
    validate_optional_broadcastable_shape(
        volume_mask,
        (grid.nx, grid.ny, grid.nz),
        context="fista_tv support",
        name="support",
        fix="use a support mask broadcastable to shape (grid.nx, grid.ny, grid.nz).",
    )
    if init_x is not None:
        validate_volume(init_x, grid, context="fista_tv init_x", name="init_x")

    x0 = None if init_x is None else jnp.asarray(init_x, dtype=jnp.float32)
    if constraints_enabled and x0 is not None:
        x0 = _project_constraints(
            x0,
            positivity=constraints.positivity,
            lower_bound=constraints.lower_bound,
            upper_bound=constraints.upper_bound,
        )

    poses = stack_view_poses(geometry, n_views)
    validate_pose_stack(poses, n_views, context="fista_tv geometry")
    projector = _batched_projector(cfg, n_views, det_grid)

    lipschitz = cfg.L
    if lipschitz is None and projector is not None:
        lipschitz = float(
            _batched_lipschitz(
                poses,
                volume_mask,
                grid=grid,
                detector=detector,
                projector=projector,
                iters=int(cfg.power_iters),
            )
        )
    if lipschitz is None:
        lipschitz = power_method_L(
            geometry,
            grid,
            detector,
            projections.shape,
            iters=cfg.power_iters,
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode="stream",
            T_all=poses,
            vol_mask=volume_mask,
            det_grid=det_grid,
            ray_integrator=cfg.ray_integrator,
        )
    if regulariser == "huber_tv" and float(cfg.lambda_tv) != 0.0:
        lipschitz += float(cfg.lambda_tv) * 12.0 / huber_delta

    return _FistaRuntime(
        config=cfg,
        regulariser=regulariser,
        huber_delta=huber_delta,
        constraints=constraints,
        x0=x0,
        poses=poses,
        lipschitz=float(lipschitz),
        volume_mask=volume_mask,
        detector_grid=det_grid,
        projector=projector,
    )


def _batched_projector(
    cfg: FistaConfig, n_views: int, det_grid: object
) -> tuple[str, str, int] | None:
    """Choose batched operators, or None for the ray-model reference path."""
    model, backend = cfg.projector_model, cfg.projector_backend
    if det_grid is not None or cfg.ray_integrator != "sampled":
        if model == "joseph" or backend == "pallas":
            raise ValueError(
                "fista_tv: explicit detector grids and exact integration require "
                "projector_model='ray' and projector_backend='jax'"
            )
        return None
    model, backend = resolve_projector(model, backend, context="fista_tv")
    if model == "ray" and backend == "jax":
        return None
    requested = 64 if cfg.views_per_batch is None else int(cfg.views_per_batch)
    return model, backend, max(1, min(requested, n_views))


@functools.partial(jax.jit, static_argnames=("grid", "detector", "projector", "iters"))
def _batched_lipschitz(
    poses: jnp.ndarray,
    mask: jnp.ndarray | None,
    *,
    grid: Grid,
    detector: Detector,
    projector: tuple[str, str, int],
    iters: int,
) -> jnp.ndarray:
    model, backend, batch = projector
    forward, adjoint = projection_operators(poses, grid, detector, None, backend, batch, model)
    mask = None if mask is None else jnp.asarray(mask, jnp.float32)
    return normal_operator_norm(
        forward, adjoint, (grid.nx, grid.ny, grid.nz), iters=iters, mask=mask
    )


def _data_term(
    grid: Grid, detector: Detector, projections: jnp.ndarray, runtime: _FistaRuntime
) -> tuple[
    Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]], Callable[[jnp.ndarray], jnp.ndarray]
]:
    """Return the data term's value-and-gradient and value functions."""
    cfg = runtime.config
    regulariser = runtime.regulariser
    huber_delta = runtime.huber_delta
    mask = None if runtime.volume_mask is None else jnp.asarray(runtime.volume_mask, jnp.float32)
    if runtime.projector is not None:
        model, backend, batch = runtime.projector
        least_squares, squared_error = least_squares_operators(
            runtime.poses, grid, detector, backend, batch, model
        )

        def batched_value_and_grad(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            v, g = least_squares(z if mask is None else z * mask, projections)
            return v, g if mask is None else g * mask

        def batched_value(x: jnp.ndarray) -> jnp.ndarray:
            return squared_error(x if mask is None else x * mask, projections)

    def val_and_grad_fn(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if runtime.projector is not None:
            v, g = batched_value_and_grad(z)
            if regulariser == "huber_tv" and float(cfg.lambda_tv) != 0.0:
                g = g + jnp.asarray(cfg.lambda_tv, dtype=z.dtype) * huber_tv_grad(z, huber_delta)
            return v, g
        g, v = grad_data_term(
            None,
            grid,
            detector,
            projections,
            z,
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode=cfg.grad_mode,
            T_all=runtime.poses,
            vol_mask=runtime.volume_mask,
            det_grid=runtime.detector_grid,
            ray_integrator=cfg.ray_integrator,
        )
        if regulariser == "huber_tv" and float(cfg.lambda_tv) != 0.0:
            g = g + jnp.asarray(cfg.lambda_tv, dtype=z.dtype) * huber_tv_grad(z, huber_delta)
        return v, g

    def data_value_fn(x: jnp.ndarray) -> jnp.ndarray:
        if runtime.projector is not None:
            return batched_value(x)
        return data_term_value(
            None,
            grid,
            detector,
            projections,
            x,
            views_per_batch=cfg.views_per_batch,
            projector_unroll=cfg.projector_unroll,
            checkpoint_projector=cfg.checkpoint_projector,
            gather_dtype=cfg.gather_dtype,
            grad_mode=cfg.grad_mode,
            T_all=runtime.poses,
            vol_mask=runtime.volume_mask,
            det_grid=runtime.detector_grid,
            ray_integrator=cfg.ray_integrator,
        )

    return val_and_grad_fn, data_value_fn


@functools.partial(jax.jit, static_argnames=("grid", "detector"))
def _run_fista_scan(
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    runtime: _FistaRuntime,
) -> FistaScanState:
    cfg = runtime.config
    regulariser = runtime.regulariser
    huber_delta = runtime.huber_delta
    constraints = runtime.constraints
    L = runtime.lipschitz

    val_and_grad_fn, data_value_fn = _data_term(grid, detector, projections, runtime)
    val_and_grad = jax.jit(val_and_grad_fn, donate_argnums=(0,))
    data_value = jax.jit(data_value_fn, donate_argnums=(0,))
    tv_prox_jit = jax.jit(tv_proximal, static_argnames=("iters",))

    def regulariser_value_fn(x: jnp.ndarray) -> jnp.ndarray:
        if regulariser == "huber_tv":
            return huber_tv_value(x, huber_delta)
        return isotropic_tv_value(x)

    use_early_stop = (
        (cfg.recon_rel_tol is not None)
        and float(cfg.recon_rel_tol) > 0.0
        and int(cfg.recon_patience) > 0
    )
    tol = jnp.float32(float(cfg.recon_rel_tol) if use_early_stop else 0.0)
    patience = jnp.int32(int(cfg.recon_patience) if use_early_stop else 0)
    early_flag = jnp.bool_(use_early_stop)

    def step(state: FistaScanState, k: jnp.ndarray) -> tuple[FistaScanState, None]:
        def run_active(active_state: FistaScanState) -> FistaScanState:
            _, g = val_and_grad(active_state.z)
            y = active_state.z - (1.0 / L) * g
            if regulariser == "huber_tv":
                x_new = y
            else:
                x_new = tv_prox_jit(y, cfg.lambda_tv / L, iters=int(cfg.tv_prox_iters))
            x_new = _project_constraints(
                x_new,
                positivity=constraints.positivity,
                lower_bound=constraints.lower_bound,
                upper_bound=constraints.upper_bound,
            )
            t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * active_state.t * active_state.t))
            z_new = x_new + ((active_state.t - 1.0) / t_new) * (x_new - active_state.x)
            z_new = _project_constraints(
                z_new,
                positivity=constraints.positivity,
                lower_bound=constraints.lower_bound,
                upper_bound=constraints.upper_bound,
            )
            data_loss_val = data_value(x_new)
            reg_value = regulariser_value_fn(x_new)
            obj = data_loss_val + cfg.lambda_tv * reg_value
            obj32 = obj.astype(jnp.float32)
            rel_change = jnp.abs(obj - active_state.prev_obj) / jnp.maximum(
                jnp.abs(active_state.prev_obj),
                1e-6,
            )
            small = jnp.logical_and(
                early_flag,
                jnp.logical_and(active_state.has_prev, rel_change <= tol),
            )
            streak_next = jnp.where(
                early_flag,
                jnp.where(small, jnp.minimum(active_state.streak + 1, patience), jnp.int32(0)),
                active_state.streak,
            )
            done_next = jnp.logical_or(
                active_state.done,
                jnp.logical_and(early_flag, streak_next >= patience),
            )
            return active_state._replace(
                x=x_new,
                z=z_new,
                t=t_new,
                loss=active_state.loss.at[k].set(obj32),
                prev_obj=obj,
                streak=streak_next,
                done=done_next,
                has_prev=jnp.bool_(True),
                last_obj=obj,
                iters_done=active_state.iters_done + jnp.int32(1),
            )

        def run_skip(skip_state: FistaScanState) -> FistaScanState:
            return skip_state._replace(
                loss=skip_state.loss.at[k].set(skip_state.last_obj.astype(jnp.float32))
            )

        return jax.lax.cond(state.done, run_skip, run_active, state), None

    loss_arr0 = jnp.zeros((int(cfg.iters),), dtype=jnp.float32)
    x0 = runtime.x0
    if x0 is None:
        x0 = _project_constraints(
            jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
            positivity=constraints.positivity,
            lower_bound=constraints.lower_bound,
            upper_bound=constraints.upper_bound,
        )
    init_carry = FistaScanState(
        x=x0,
        z=x0,
        t=jnp.float32(1.0),
        loss=loss_arr0,
        prev_obj=jnp.float32(0.0),
        streak=jnp.int32(0),
        done=jnp.bool_(False),
        has_prev=jnp.bool_(False),
        last_obj=jnp.float32(0.0),
        iters_done=jnp.int32(0),
    )
    carry_final, _ = jax.lax.scan(step, init_carry, jnp.arange(int(cfg.iters)))
    return carry_final


def _emit_fista_callback(callback: LossCallback | None, result: _FistaResult) -> None:
    if result.losses.size == 0:
        return
    final_step = max(result.effective_iters - 1, 0)
    emit_loss_callback_endpoints(
        callback,
        (
            (0, float(result.losses[0])),
            (final_step, float(result.losses[final_step])),
        ),
    )


def fista_tv(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    init_x: jnp.ndarray | None = None,
    config: FistaConfig | None = None,
    callback: LossCallback | None = None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, dict[str, object]]:
    """Run FISTA with TV regularization using an explicit solver configuration.

    If ``callback`` is provided, it fires on the first recorded loss sample and on
    the final recorded loss sample. The callback arguments are ``(step, loss)``,
    where ``step`` is the zero-based iteration index that produced ``loss``. When
    early stopping truncates active iterations, the final callback reports the
    last active iteration rather than the repeated padded tail entry.
    """
    runtime = _prepare_fista_runtime(
        geometry,
        grid,
        detector,
        projections,
        init_x=init_x,
        config=config,
        det_grid=det_grid,
    )
    final = _run_fista_scan(grid, detector, projections, runtime)
    result = _FistaResult(
        volume=final.x,
        losses=final.loss,
        lipschitz=runtime.lipschitz,
        effective_iters=int(final.iters_done),
        early_stop=bool(final.done),
        regulariser=runtime.regulariser,
        huber_delta=runtime.huber_delta,
    )
    _emit_fista_callback(callback, result)
    return result.volume, result.info()
