from __future__ import annotations

from typing import Literal, Mapping, Sequence, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from .dofs import DOF_INDEX


type GaugeFixMode = Literal["mean_translation", "none"]


class GaugeStats(TypedDict, total=False):
    mode: str
    dofs: list[str]
    dx_mean_before: float
    dz_mean_before: float
    dx_mean_after: float
    dz_mean_after: float


_VALID_GAUGE_FIXES = {"mean_translation", "none"}
_TRANSLATION_DOFS = ("dx", "dz")


def normalize_gauge_fix(raw: object) -> GaugeFixMode:
    """Normalize a gauge-fixing mode from API, CLI, or config input."""
    value = str(raw).strip().lower().replace("-", "_")
    if value in {"off", "false", "disabled", "disable"}:
        value = "none"
    if value not in _VALID_GAUGE_FIXES:
        valid = ", ".join(sorted(_VALID_GAUGE_FIXES))
        raise ValueError(f"gauge_fix must be one of {valid}; got {raw!r}")
    return value  # type: ignore[return-value]


def active_gauge_dofs(*, mode: GaugeFixMode, active_mask: Sequence[bool]) -> tuple[str, ...]:
    """Return translation DOFs affected by the selected gauge mode."""
    if mode == "none":
        return ()
    mask = tuple(bool(v) for v in active_mask)
    return tuple(name for name in _TRANSLATION_DOFS if mask[DOF_INDEX[name]])


def validate_alignment_gauge_feasible(
    *,
    mode: GaugeFixMode,
    active_mask: Sequence[bool],
    bounds_lower: object,
    bounds_upper: object,
) -> None:
    """Validate that active translation bounds admit a zero-mean trajectory."""
    if mode == "none":
        return
    lower = np.asarray(bounds_lower, dtype=np.float64)
    upper = np.asarray(bounds_upper, dtype=np.float64)
    if lower.shape != (5,) or upper.shape != (5,):
        raise ValueError(
            "alignment gauge bounds must have shape (5,) for [alpha,beta,phi,dx,dz]"
        )
    for name in active_gauge_dofs(mode=mode, active_mask=active_mask):
        idx = DOF_INDEX[name]
        lo = float(lower[idx])
        hi = float(upper[idx])
        if lo > 0.0 or hi < 0.0:
            raise ValueError(
                f"gauge_fix='mean_translation' requires active {name} bounds to include 0; "
                f"got {name}={lo:g}:{hi:g}"
            )


def _project_box_zero_mean(
    values: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> jnp.ndarray:
    """Project one vector onto a box-constrained zero-sum set."""
    values = jnp.asarray(values, dtype=jnp.float32)
    lower = jnp.asarray(lower, dtype=jnp.float32)
    upper = jnp.asarray(upper, dtype=jnp.float32)
    n = jnp.maximum(jnp.asarray(values.shape[0], dtype=jnp.float32), jnp.float32(1.0))

    finite_scale = jnp.maximum(jnp.max(jnp.abs(values)), jnp.float32(1.0))
    surrogate_lower = jnp.where(jnp.isfinite(lower), lower, -finite_scale * 4.0)
    surrogate_upper = jnp.where(jnp.isfinite(upper), upper, finite_scale * 4.0)
    lam_lo = jnp.min(values - surrogate_upper) - finite_scale - jnp.float32(1.0)
    lam_hi = jnp.max(values - surrogate_lower) + finite_scale + jnp.float32(1.0)

    def body(_, carry):
        lo, hi = carry
        mid = (lo + hi) * jnp.float32(0.5)
        total = jnp.sum(jnp.clip(values - mid, lower, upper))
        lo = jnp.where(total > 0.0, mid, lo)
        hi = jnp.where(total > 0.0, hi, mid)
        return lo, hi

    lam_lo, lam_hi = jax.lax.fori_loop(0, 64, body, (lam_lo, lam_hi))
    projected = jnp.clip(values - (lam_lo + lam_hi) * jnp.float32(0.5), lower, upper)
    residual_mean = jnp.sum(projected) / n
    return projected - residual_mean


def apply_alignment_gauge(
    params5: jnp.ndarray,
    *,
    mode: GaugeFixMode,
    active_mask: Sequence[bool],
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray | str | list[str]]]:
    """Apply the selected alignment gauge and return JAX-friendly stats."""
    params = jnp.asarray(params5, dtype=jnp.float32)
    mode = normalize_gauge_fix(mode)
    active = tuple(bool(v) for v in active_mask)
    gauge_dofs = active_gauge_dofs(mode=mode, active_mask=active)

    dx_before = jnp.mean(params[:, DOF_INDEX["dx"]])
    dz_before = jnp.mean(params[:, DOF_INDEX["dz"]])
    out = params
    if mode == "mean_translation":
        for name in gauge_dofs:
            idx = DOF_INDEX[name]
            col = _project_box_zero_mean(out[:, idx], bounds_lower[idx], bounds_upper[idx])
            out = out.at[:, idx].set(col)

    stats: Mapping[str, jnp.ndarray | str | list[str]] = {
        "mode": mode,
        "dofs": list(gauge_dofs),
        "dx_mean_before": dx_before,
        "dz_mean_before": dz_before,
        "dx_mean_after": jnp.mean(out[:, DOF_INDEX["dx"]]),
        "dz_mean_after": jnp.mean(out[:, DOF_INDEX["dz"]]),
    }
    return out, stats


def gauge_stats_to_python(
    stats: Mapping[str, object],
    *,
    include_means: bool = True,
) -> GaugeStats:
    """Convert gauge stats containing JAX scalars into JSON/log friendly values."""
    out: GaugeStats = {
        "mode": str(stats.get("mode", "none")),
        "dofs": [str(v) for v in stats.get("dofs", [])],
    }
    if include_means:
        for key in (
            "dx_mean_before",
            "dz_mean_before",
            "dx_mean_after",
            "dz_mean_after",
        ):
            value = stats.get(key)
            if value is not None:
                out[key] = float(value)  # type: ignore[literal-required]
    return out


__all__ = [
    "GaugeFixMode",
    "GaugeStats",
    "active_gauge_dofs",
    "apply_alignment_gauge",
    "gauge_stats_to_python",
    "normalize_gauge_fix",
    "validate_alignment_gauge_feasible",
]
