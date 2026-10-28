"""Adapters from typed loss specs to vectorised per-view loss functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .loss_kernels import (
    _compute_otsu_threshold,
    _loss_huber,
    _loss_l2,
    _loss_l2_otsu_soft,
    _safe_epsilon,
)
from .loss_specs import (
    AlignmentLossSpec,
    HuberLossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    canonicalize_loss_kind,
    loss_spec_name,
    loss_spec_params,
    loss_spec_supports_setup_validation_lm,
    parse_loss_spec,
)
from .loss_state import LossState


class PerViewLossFn(Protocol):
    def __call__(
        self,
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray | None,
        view_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray: ...


type GaussNewtonWeightFn = Callable[[jnp.ndarray, jnp.ndarray | None], jnp.ndarray]


@dataclass(frozen=True, slots=True)
class LossAdapter:
    """Prepared loss callable plus metadata used by alignment objectives."""

    spec: AlignmentLossSpec
    name: str
    state: LossState
    per_view_loss: PerViewLossFn
    gauss_newton_weights: GaussNewtonWeightFn
    supports_setup_validation_lm: bool

    def residual_model(self, residual: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the true objective and square-root IRLS weights at this residual.

        The residual already includes any target-only foreground weighting.
        Huber weights multiply both residuals and Jacobian columns, not just
        residuals. Line search must use the true objective, not its surrogate.
        """
        if isinstance(self.spec, HuberLossSpec):
            delta = self.spec.delta
            magnitude = jnp.abs(residual)
            quadratic = jnp.minimum(magnitude, delta)
            value = jnp.sum(0.5 * quadratic**2 + delta * (magnitude - quadratic))
            return value, jnp.sqrt(delta / jnp.maximum(delta, magnitude))
        return jnp.float32(0.5) * jnp.vdot(residual, residual).real, jnp.float32(1.0)


def _otsu_thresholds(targets: jnp.ndarray) -> np.ndarray:
    Ys = np.asarray(targets)
    return np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)


def _build_l2_otsu_loss(
    state: LossState,
    targets: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, LossState], jnp.ndarray]:
    temp = _safe_epsilon(state.params, "temp", 0.5)
    thr = _otsu_thresholds(targets)
    state.mask = jax.device_put(jax.nn.sigmoid((targets - jnp.asarray(thr)[:, None, None]) / temp))
    return _loss_l2_otsu_soft


def _build_loss_from_kind(
    kind: str,
    params: dict[str, float] | None,
    targets: jnp.ndarray,
) -> tuple[PerViewLossFn, LossState]:
    k = canonicalize_loss_kind(kind)
    p = {} if params is None else {str(a): float(b) for a, b in params.items()}
    state = LossState(kind=k, params=p)
    if k == "l2_otsu":
        f = _build_l2_otsu_loss(state, targets)
    else:
        f = _loss_huber if k == "huber" else _loss_l2

    def per_view_fn(
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray | None,
        view_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if mask_chunk is None and state.mask is None:
            return jax.vmap(lambda a, b: f(a, b, state))(pred_chunk, tar_chunk)

        bsz = pred_chunk.shape[0]
        local_indices = jnp.arange(bsz, dtype=jnp.int32)
        if view_indices is None:
            global_indices = local_indices
        else:
            global_indices = jnp.asarray(view_indices, dtype=jnp.int32)

        def apply_one(
            a: jnp.ndarray,
            b: jnp.ndarray,
            local_idx: jnp.ndarray,
            global_idx: jnp.ndarray,
        ) -> jnp.ndarray:
            ls = LossState(kind=state.kind, params=state.params)
            if state.mask is not None:
                ls.mask = state.mask[global_idx]
            if mask_chunk is not None:
                ls.mask = mask_chunk[local_idx]
            return f(a, b, ls)

        return jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            pred_chunk,
            tar_chunk,
            local_indices,
            global_indices,
        )

    return per_view_fn, state


def _gauss_newton_weight_builder(spec: AlignmentLossSpec) -> GaussNewtonWeightFn:
    if isinstance(spec, L2LossSpec | HuberLossSpec):
        return lambda y_chunk, _mask_chunk: jnp.ones_like(y_chunk)
    if isinstance(spec, L2OtsuLossSpec):
        temp = jnp.float32(max(float(spec.temp), 1e-6))

        def _otsu_weights(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray | None) -> jnp.ndarray:
            if mask_chunk is None:
                return jnp.ones_like(y_chunk)
            base = mask_chunk.astype(jnp.float32)
            return jax.nn.sigmoid((base - 0.5) / temp)

        return _otsu_weights
    raise TypeError(f"Unsupported loss spec: {type(spec)!r}")


def loss_supports_setup_validation_lm(spec: AlignmentLossSpec) -> bool:
    """Return whether a loss can be represented as validation-LM residuals."""
    return loss_spec_supports_setup_validation_lm(spec)


def build_loss_adapter(spec: AlignmentLossSpec, targets: jnp.ndarray) -> LossAdapter:
    """Build a prepared loss adapter for a target projection stack."""
    name = loss_spec_name(spec)
    params = loss_spec_params(spec)
    per_view_loss, state = _build_loss_from_kind(name, params, targets)
    return LossAdapter(
        spec=spec,
        name=name,
        state=state,
        per_view_loss=per_view_loss,
        gauss_newton_weights=_gauss_newton_weight_builder(spec),
        supports_setup_validation_lm=loss_supports_setup_validation_lm(spec),
    )


def build_loss(
    kind: str,
    params: dict[str, float] | None,
    targets: jnp.ndarray,
) -> tuple[PerViewLossFn, LossState]:
    """Build a per-view loss function and its precomputed state."""
    adapter = build_loss_adapter(parse_loss_spec(kind, params), targets)
    return adapter.per_view_loss, adapter.state


__all__ = [
    "GaussNewtonWeightFn",
    "LossAdapter",
    "PerViewLossFn",
    "build_loss",
    "build_loss_adapter",
    "loss_supports_setup_validation_lm",
]
