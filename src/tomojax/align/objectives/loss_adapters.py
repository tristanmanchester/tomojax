from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

try:  # optional, used for Chamfer loss
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:  # pragma: no cover
    distance_transform_edt = None  # type: ignore

from .loss_kernels import (
    _compute_otsu_threshold,
    _loss_barron,
    _loss_cauchy,
    _loss_chamfer_edge,
    _loss_charbonnier,
    _loss_correntropy,
    _loss_edge_aware_l2,
    _loss_fft_mag,
    _loss_grad_l1,
    _loss_grad_orient,
    _loss_huber,
    _loss_l2,
    _loss_l2_otsu_soft,
    _loss_mi_kde,
    _loss_mind,
    _loss_ms_ssim,
    _loss_ngf,
    _loss_phase_corr_soft,
    _loss_poisson_nll,
    _loss_pwls,
    _loss_renyi_mi,
    _loss_ssim,
    _loss_ssim_otsu,
    _loss_student_t,
    _loss_swd,
    _loss_tversky,
    _loss_welsch,
    _loss_zncc,
    _safe_epsilon,
    _validated_renyi_alpha,
)
from .loss_specs import (
    AlignmentLossSpec,
    EdgeL2LossSpec,
    L2LossSpec,
    L2OtsuLossSpec,
    PWLSLossSpec,
    canonicalize_loss_kind,
    loss_spec_name,
    loss_spec_params,
    loss_spec_supports_setup_validation_lm,
    parse_loss_spec,
)
from .loss_state import LossState

PerViewLossFn: TypeAlias = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
    jnp.ndarray,
]
GaussNewtonWeightFn: TypeAlias = Callable[[jnp.ndarray, jnp.ndarray | None], jnp.ndarray]
LossBuilderFn: TypeAlias = Callable[
    [LossState, jnp.ndarray],
    Callable[[jnp.ndarray, jnp.ndarray, LossState], jnp.ndarray],
]


@dataclass(frozen=True, slots=True)
class LossAdapter:
    spec: AlignmentLossSpec
    name: str
    state: LossState
    per_view_loss: PerViewLossFn
    supports_gauss_newton: bool
    gauss_newton_weights: GaussNewtonWeightFn
    supports_setup_validation_lm: bool


def _plain_loss_builder(
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray, LossState], jnp.ndarray],
) -> LossBuilderFn:
    def _builder(state: LossState, targets: jnp.ndarray):
        del targets
        return loss_fn

    return _builder


def _precompute_kde_state(state: LossState, targets: jnp.ndarray) -> None:
    bins = int(state.params.get("bins", 32))
    lo = float(np.min(targets))
    hi = float(np.max(targets))
    hi = hi if hi > lo else (lo + 1.0)
    state.bins_x = jnp.linspace(lo, hi, bins)
    state.bins_y = jnp.linspace(lo, hi, bins)
    state.bw_x = float(state.params.get("bw_x", (hi - lo) / max(8.0, bins)))
    state.bw_y = float(state.params.get("bw_y", (hi - lo) / max(8.0, bins)))


def _otsu_thresholds(targets: jnp.ndarray) -> np.ndarray:
    Ys = np.asarray(targets)
    return np.array([_compute_otsu_threshold(Ys[i]) for i in range(Ys.shape[0])], dtype=np.float32)


def _build_chamfer_edge_loss(state: LossState, targets: jnp.ndarray):
    if distance_transform_edt is None:
        raise ValueError("scipy.ndimage is required for chamfer_edge loss")
    Ys = np.asarray(targets, np.float32)
    dts = []
    for i in range(Ys.shape[0]):
        gy, gx = np.gradient(Ys[i])
        mag = np.sqrt(gx * gx + gy * gy)
        edges = mag > (0.5 * max(1e-6, float(np.mean(mag))))
        dts.append(distance_transform_edt(~edges).astype(np.float32))
    state.dt_edge = jax.device_put(jnp.asarray(np.stack(dts, axis=0)))
    return _loss_chamfer_edge


def _build_mi_kde_loss(state: LossState, targets: jnp.ndarray):
    _precompute_kde_state(state, targets)
    return _loss_mi_kde


def _build_nmi_kde_loss(state: LossState, targets: jnp.ndarray):
    _precompute_kde_state(state, targets)
    state.params["nmi"] = 1.0
    return _loss_mi_kde


def _build_renyi_mi_loss(state: LossState, targets: jnp.ndarray):
    _validated_renyi_alpha(state.params)
    _precompute_kde_state(state, targets)
    return _loss_renyi_mi


def _build_l2_otsu_loss(state: LossState, targets: jnp.ndarray):
    temp = _safe_epsilon(state.params, "temp", 0.5)
    thr = _otsu_thresholds(targets)
    state.mask = jax.device_put(jax.nn.sigmoid((targets - jnp.asarray(thr)[:, None, None]) / temp))
    return _loss_l2_otsu_soft


def _build_ssim_otsu_loss(state: LossState, targets: jnp.ndarray):
    Ys = np.asarray(targets)
    thr = _otsu_thresholds(targets)
    mask = (Ys >= thr[:, None, None]).astype(np.float32)
    state.mask = jax.device_put(jnp.asarray(mask))
    return _loss_ssim_otsu


def _build_tversky_loss(state: LossState, targets: jnp.ndarray):
    state.thr = jax.device_put(jnp.asarray(_otsu_thresholds(targets))[:, None, None])
    return _loss_tversky


_LOSS_BUILDERS: dict[str, LossBuilderFn] = {
    "l2": _plain_loss_builder(_loss_l2),
    "charbonnier": _plain_loss_builder(_loss_charbonnier),
    "huber": _plain_loss_builder(_loss_huber),
    "cauchy": _plain_loss_builder(_loss_cauchy),
    "welsch": _plain_loss_builder(_loss_welsch),
    "zncc": _plain_loss_builder(_loss_zncc),
    "ssim": _plain_loss_builder(_loss_ssim),
    "ms_ssim": _plain_loss_builder(_loss_ms_ssim),
    "grad_l1": _plain_loss_builder(_loss_grad_l1),
    "edge_l2": _plain_loss_builder(_loss_edge_aware_l2),
    "ngf": _plain_loss_builder(_loss_ngf),
    "grad_orient": _plain_loss_builder(_loss_grad_orient),
    "phasecorr": _plain_loss_builder(_loss_phase_corr_soft),
    "fft_mag": _plain_loss_builder(_loss_fft_mag),
    "chamfer_edge": _build_chamfer_edge_loss,
    "poisson": _plain_loss_builder(_loss_poisson_nll),
    "pwls": _plain_loss_builder(_loss_pwls),
    "student_t": _plain_loss_builder(_loss_student_t),
    "barron": _plain_loss_builder(_loss_barron),
    "correntropy": _plain_loss_builder(_loss_correntropy),
    "mi": _build_mi_kde_loss,
    "nmi": _build_nmi_kde_loss,
    "renyi_mi": _build_renyi_mi_loss,
    "l2_otsu": _build_l2_otsu_loss,
    "ssim_otsu": _build_ssim_otsu_loss,
    "tversky": _build_tversky_loss,
    "swd": _plain_loss_builder(_loss_swd),
    "mind": _plain_loss_builder(_loss_mind),
}


def _build_loss_from_kind(
    kind: str,
    params: dict[str, float] | None,
    targets: jnp.ndarray,
) -> tuple[PerViewLossFn, LossState]:
    k = canonicalize_loss_kind(kind)
    p = {} if params is None else {str(a): float(b) for a, b in params.items()}
    state = LossState(kind=k, params=p)
    builder = _LOSS_BUILDERS.get(k)
    if builder is None:
        raise ValueError(f"Unknown loss kind: {kind}")
    f = builder(state, targets)

    def per_view_fn(
        pred_chunk: jnp.ndarray,
        tar_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray | None,
        view_indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # Vectorized application over batch of views (b, nv, nu)
        if (
            mask_chunk is None
            and state.mask is None
            and state.dt_edge is None
            and state.bins_x is None
            and state.thr is None
        ):
            return jax.vmap(lambda a, b: f(a, b, state))(pred_chunk, tar_chunk)

        bsz = pred_chunk.shape[0]
        local_indices = jnp.arange(bsz, dtype=jnp.int32)
        if view_indices is None:
            global_indices = local_indices
        else:
            global_indices = jnp.asarray(view_indices, dtype=jnp.int32)

        def apply_one(a, b, local_idx, global_idx):
            ls = LossState(kind=state.kind, params=state.params)
            if state.mask is not None:
                ls.mask = state.mask[global_idx]
            if mask_chunk is not None:
                ls.mask = mask_chunk[local_idx]
            if state.dt_edge is not None:
                ls.dt_edge = state.dt_edge[global_idx]
            ls.bins_x = state.bins_x
            ls.bins_y = state.bins_y
            ls.bw_x = state.bw_x
            ls.bw_y = state.bw_y
            if state.thr is not None:
                ls.thr = state.thr[global_idx]
            return f(a, b, ls)

        return jax.vmap(apply_one, in_axes=(0, 0, 0, 0))(
            pred_chunk,
            tar_chunk,
            local_indices,
            global_indices,
        )

    return per_view_fn, state


def _edge_aware_ls_weights(y_chunk: jnp.ndarray) -> jnp.ndarray:
    edge_kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], jnp.float32) / 8.0
    edge_ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], jnp.float32) / 8.0
    y4 = y_chunk[..., None]
    gx = jax.lax.conv_general_dilated(
        y4,
        edge_kx[:, :, None, None],
        (1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    gy = jax.lax.conv_general_dilated(
        y4,
        edge_ky[:, :, None, None],
        (1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    mag = jnp.sqrt(gx[..., 0] ** 2 + gy[..., 0] ** 2)
    return jnp.sqrt(1.0 + mag)


def _gauss_newton_weight_builder(spec: AlignmentLossSpec) -> tuple[bool, GaussNewtonWeightFn]:
    if isinstance(spec, L2LossSpec):
        return True, lambda y_chunk, mask_chunk: jnp.ones_like(y_chunk)
    if isinstance(spec, L2OtsuLossSpec):
        temp = jnp.float32(max(float(spec.temp), 1e-6))

        def _otsu_weights(y_chunk: jnp.ndarray, mask_chunk: jnp.ndarray | None) -> jnp.ndarray:
            if mask_chunk is None:
                return jnp.ones_like(y_chunk)
            base = mask_chunk.astype(jnp.float32)
            return jax.nn.sigmoid((base - 0.5) / temp)

        return True, _otsu_weights
    if isinstance(spec, PWLSLossSpec):
        a = jnp.float32(spec.a)
        b = jnp.float32(spec.b)
        return True, lambda y_chunk, mask_chunk: jnp.sqrt(
            1.0 / (a * jnp.clip(y_chunk, 0.0) + b + 1e-6)
        )
    if isinstance(spec, EdgeL2LossSpec):
        return True, lambda y_chunk, mask_chunk: _edge_aware_ls_weights(y_chunk)
    return False, lambda y_chunk, mask_chunk: jnp.ones_like(y_chunk)


def loss_supports_setup_validation_lm(spec: AlignmentLossSpec) -> bool:
    return loss_spec_supports_setup_validation_lm(spec)


def build_loss_adapter(spec: AlignmentLossSpec, targets: jnp.ndarray) -> LossAdapter:
    name = loss_spec_name(spec)
    params = loss_spec_params(spec)
    per_view_loss, state = _build_loss_from_kind(name, params, targets)
    supports_gauss_newton, gauss_newton_weights = _gauss_newton_weight_builder(spec)
    return LossAdapter(
        spec=spec,
        name=name,
        state=state,
        per_view_loss=per_view_loss,
        supports_gauss_newton=supports_gauss_newton,
        gauss_newton_weights=gauss_newton_weights,
        supports_setup_validation_lm=loss_supports_setup_validation_lm(spec),
    )


def build_loss(
    kind: str,
    params: dict[str, float] | None,
    targets: jnp.ndarray,
) -> tuple[PerViewLossFn, LossState]:
    adapter = build_loss_adapter(parse_loss_spec(kind, params), targets)
    return adapter.per_view_loss, adapter.state


__all__ = [
    "GaussNewtonWeightFn",
    "LossAdapter",
    "LossBuilderFn",
    "PerViewLossFn",
    "build_loss",
    "build_loss_adapter",
    "loss_supports_setup_validation_lm",
]
