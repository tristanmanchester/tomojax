"""Fixed-volume projection objectives for alignment updates."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from tomojax.core.backend_policy import (
    BackendProvenance,
    ProjectorBackendInput,
    backend_provenance,
    normalize_projector_backend,
)
from tomojax.core.cuda_projector_resolver import resolve_cuda_callable
from tomojax.core.projector import (
    cone_views,
    cuda_unsupported_exception_type,
    project_views,
)
from tomojax.core.view_chunks import chunk_schedule, chunk_size

from .loss_adapters import LossAdapter

if TYPE_CHECKING:
    from tomojax.core.geometry import Detector, Grid


DifferentiationMode = Literal["none", "unrolled", "implicit"]
InnerInitPolicy = Literal[
    "zeros", "current_level_volume", "previous_fold_volume", "previous_stage_volume"
]

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObjectiveProvenance:
    """Provenance describing how an alignment objective was constructed."""

    outer_loss_source: str
    outer_loss_kind: str
    inner_data_term: str
    inner_regulariser: str
    validation_split: str
    differentiation_mode: DifferentiationMode
    initialization_policy: InnerInitPolicy

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible provenance mapping."""
        return {
            "outer_loss_source": self.outer_loss_source,
            "outer_loss_kind": self.outer_loss_kind,
            "inner_data_term": self.inner_data_term,
            "inner_regulariser": self.inner_regulariser,
            "validation_split": self.validation_split,
            "differentiation_mode": self.differentiation_mode,
            "initialization_policy": self.initialization_policy,
        }


def _cone_projector_backend(requested: ProjectorBackendInput) -> str:
    from tomojax.core.cone import use_cuda_cone

    if normalize_projector_backend(requested) == "jax" or not use_cuda_cone():
        return "jax"
    return "cuda"


def project_and_score_stack(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    loss_adapter: LossAdapter,
    views_per_batch: int = 1,
    projector_unroll: int = 1,
    checkpoint_projector: bool = True,
    gather_dtype: str = "fp32",
    backend: ProjectorBackendInput = "jax",
    require_differentiable_projector: bool = True,
    ray_integrator: str = "sampled",
    frames: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Project and score views in parallel rays, or in cone-beam lab ``frames``."""
    backend = normalize_projector_backend(backend)
    n_views = int(pose_stack.shape[0])
    if n_views == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    b = chunk_size(n_views, views_per_batch, default=n_views)
    num_chunks = (n_views + b - 1) // b
    loss_mask = loss_adapter.state.mask
    parallel_project = functools.partial(
        project_views,
        grid=grid,
        detector=detector,
        volume=volume,
        use_checkpoint=checkpoint_projector,
        unroll=int(projector_unroll),
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        ray_integrator=ray_integrator,
    )

    cone_backend = _cone_projector_backend(backend)

    def vm_project(T: jnp.ndarray, start: jnp.ndarray | int) -> jnp.ndarray:
        """Project the views ``start, start + 1, ...`` at poses ``T``."""
        if frames is None:
            return parallel_project(T)
        # One batched cone call per chunk, not one launch per view.
        chunk = jax.lax.dynamic_slice_in_dim(frames, start, T.shape[0])
        return cone_views(T, grid, detector, volume, chunk, cone_backend)

    if (
        ray_integrator == "sampled"
        and frames is None
        and backend == "cuda"
        and loss_adapter.name == "l2"
        and not require_differentiable_projector
    ):
        score = _cuda_l2_score(
            pose_stack=pose_stack,
            grid=grid,
            detector=detector,
            volume=volume,
            det_grid=det_grid,
            targets=targets,
            gather_dtype=gather_dtype,
        )
        if score is not None:
            return score

    def body(loss_acc: jnp.ndarray, i: jnp.ndarray) -> tuple[jnp.ndarray, None]:
        start_shifted, valid_mask, _view_idx = chunk_schedule(i, n_views=n_views, chunk_size=b)
        T_chunk = jax.lax.dynamic_slice_in_dim(pose_stack, start_shifted, b)
        y_chunk = jax.lax.dynamic_slice_in_dim(targets, start_shifted, b)
        mask = (
            None if loss_mask is None else jax.lax.dynamic_slice_in_dim(loss_mask, start_shifted, b)
        )
        losses = loss_adapter.per_view_loss(vm_project(T_chunk, start_shifted), y_chunk, mask)
        return loss_acc + jnp.sum(losses * valid_mask), None

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if num_chunks == 1:
        return body(zero, jnp.int32(0))[0]
    return jax.lax.scan(body, zero, jnp.arange(num_chunks, dtype=jnp.int32))[0]


def _cuda_l2_score(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    targets: jnp.ndarray,
    gather_dtype: str,
) -> jnp.ndarray | None:
    """Half the CUDA residual sum of squares, or None where the CUDA kernel declines."""
    fallback_reason = cuda_sinogram_fallback_reason(
        pose_stack=pose_stack,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
        gather_dtype=gather_dtype,
    )
    cuda_score, callable_reason = resolve_cuda_callable(
        "forward_project_residual_sse_T_cuda",
        missing_reason="cuda_residual_sse_callable_missing",
    )
    fallback_reason = fallback_reason or callable_reason
    if cuda_score is not None and fallback_reason is None:
        unsupported_exc = cuda_unsupported_exception_type(resolve_cuda_callable) or ValueError
        try:
            options = _cuda_projector_options(
                gather_dtype=gather_dtype,
                det_grid=det_grid,
                state_mode="cached",
            )
            return jnp.float32(0.5) * cuda_score(
                pose_stack,
                grid,
                detector,
                volume,
                targets,
                options=options,
            )
        except unsupported_exc as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
    LOG.debug(
        "Falling back to JAX projector after CUDA residual scoring rejection: %s", fallback_reason
    )
    return None


def alignment_projector_backend_provenance(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    backend: ProjectorBackendInput,
    require_differentiable_projector: bool,
    api_surface: str,
    gather_dtype: str = "fp32",
    ray_integrator: str = "sampled",
    frames: jnp.ndarray | None = None,
) -> BackendProvenance:
    """Report which projector backend is used by an alignment objective."""
    requested = normalize_projector_backend(backend)
    if frames is not None:
        actual = _cone_projector_backend(requested)
        reason = None
        if requested == "cuda" and actual == "jax":
            reason = "CUDA cone kernels are unavailable"
        return backend_provenance(
            requested_backend=requested,
            actual_backend=actual,
            api_surface=api_surface,
            fallback_reason=reason,
            differentiability="gradient_safe",
        )
    if requested == "jax":
        return backend_provenance(
            requested_backend="jax",
            actual_backend="jax",
            api_surface=api_surface,
            differentiability="gradient_safe",
        )
    if ray_integrator == "exact":
        return backend_provenance(
            requested_backend=requested,
            actual_backend="jax",
            api_surface=api_surface,
            fallback_reason="exact alignment scoring uses the differentiable JAX integrator",
            differentiability="gradient_safe",
        )
    if require_differentiable_projector:
        return backend_provenance(
            requested_backend="cuda",
            actual_backend="jax",
            api_surface=api_surface,
            fallback_reason="alignment objectives require gradient-safe projector semantics",
            differentiability="gradient_safe",
        )
    reason = cuda_sinogram_fallback_reason(
        pose_stack=pose_stack,
        grid=grid,
        detector=detector,
        volume=volume,
        det_grid=det_grid,
        gather_dtype=gather_dtype,
    )
    return backend_provenance(
        requested_backend="cuda",
        actual_backend=("jax" if reason else "cuda"),
        api_surface=api_surface,
        fallback_reason=reason,
        differentiability="performance_only",
    )


def cuda_sinogram_fallback_reason(
    *,
    pose_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    gather_dtype: str,
) -> str | None:
    support_fn, reason = resolve_cuda_callable(
        "cuda_projector_sinogram_unsupported_reason",
        missing_reason="cuda_sinogram_support_check_missing",
    )
    if support_fn is not None:
        try:
            reason = support_fn(
                pose_stack,
                grid,
                detector,
                volume,
                options=_cuda_projector_options(
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                    state_mode="cached",
                ),
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOG.debug(
                "CUDA sinogram support probe failed; falling back to JAX projector: %s",
                reason,
            )
    return reason


def _cuda_projector_options(
    *,
    gather_dtype: str,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    state_mode: str,
) -> object | None:
    options_cls, _reason = resolve_cuda_callable(
        "CudaProjectorOptions",
        missing_reason="cuda_options_missing",
    )
    if options_cls is None:
        return None
    return options_cls(
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        state_mode=state_mode,
    )
