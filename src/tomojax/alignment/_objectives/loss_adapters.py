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
)
from .loss_state import LossState


class PerViewLossFn(Protocol):
    def __call__(
        self,
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray | None,
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


def _otsu_mask(targets: jnp.ndarray, temp: float) -> jnp.ndarray:
    thr = _otsu_thresholds(targets)
    return jax.device_put(jax.nn.sigmoid((targets - jnp.asarray(thr)[:, None, None]) / temp))


def _per_view_loss(state: LossState) -> PerViewLossFn:
    f = {"l2_otsu": _loss_l2_otsu_soft, "huber": _loss_huber}.get(state.kind, _loss_l2)

    def per_view_fn(
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Each view's loss; ``mask_chunk`` is these views' slice of ``state.mask``."""
        if mask_chunk is None:
            return jax.vmap(lambda a, b: f(a, b, state))(pred_chunk, tar_chunk)

        def apply_one(a: jnp.ndarray, b: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
            return f(a, b, LossState(kind=state.kind, params=state.params, mask=m))

        return jax.vmap(apply_one)(pred_chunk, tar_chunk, mask_chunk)

    return per_view_fn


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


def loss_mask(spec: AlignmentLossSpec, targets: jnp.ndarray) -> jnp.ndarray | None:
    """The per-pixel mask a loss derives from its targets once, or None."""
    if not isinstance(spec, L2OtsuLossSpec):
        return None
    return _otsu_mask(targets, _safe_epsilon(loss_spec_params(spec), "temp", 0.5))


def loss_adapter(spec: AlignmentLossSpec, mask: jnp.ndarray | None) -> LossAdapter:
    """The loss adapter for ``spec`` with its precomputed ``mask`` (see :func:`loss_mask`)."""
    name = loss_spec_name(spec)
    params = {str(a): float(b) for a, b in loss_spec_params(spec).items()}
    state = LossState(kind=canonicalize_loss_kind(name), params=params, mask=mask)
    return LossAdapter(
        spec=spec,
        name=name,
        state=state,
        per_view_loss=_per_view_loss(state),
        gauss_newton_weights=_gauss_newton_weight_builder(spec),
        supports_setup_validation_lm=loss_supports_setup_validation_lm(spec),
    )


def build_loss_adapter(spec: AlignmentLossSpec, targets: jnp.ndarray) -> LossAdapter:
    """Build a prepared loss adapter for a target projection stack."""
    return loss_adapter(spec, loss_mask(spec, targets))


__all__ = [
    "GaussNewtonWeightFn",
    "LossAdapter",
    "PerViewLossFn",
    "build_loss_adapter",
    "loss_adapter",
    "loss_mask",
    "loss_supports_setup_validation_lm",
]
