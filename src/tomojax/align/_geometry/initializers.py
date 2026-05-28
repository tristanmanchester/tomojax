"""Initializers and deterministic view splits for setup geometry alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax.numpy as jnp

    from tomojax.align._objectives.loss_adapters import LossAdapter
    from tomojax.core.geometry import Geometry


@dataclass(frozen=True, slots=True)
class DetectorCenterSeed:
    """Projection-domain detector-u seed in native detector pixels."""

    det_u_px: float
    intercept_px: float
    amplitude_px: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-native seed summary."""
        return {
            "det_u_px": float(self.det_u_px),
            "intercept_px": float(self.intercept_px),
            "amplitude_px": float(self.amplitude_px),
            "status": self.status,
        }


def train_heldout_view_indices(
    n_views: int,
    *,
    holdout_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic interleaved train/held-out view indices for characterization."""
    n = int(n_views)
    if n < 4:
        raise ValueError("held-out split requires at least 4 views")
    stride = max(3, int(holdout_stride))
    all_indices = np.arange(n, dtype=np.int32)
    heldout = all_indices[(all_indices % stride) == (stride // 2)]
    if heldout.size == 0:
        heldout = all_indices[:: max(2, n // 4)]
    train_mask = np.ones((n,), dtype=bool)
    train_mask[heldout] = False
    train = all_indices[train_mask]
    if train.size == 0 or heldout.size == 0:
        raise ValueError("held-out split produced an empty partition")
    return train, heldout


def projection_com_det_u_seed(
    projections: jnp.ndarray,
    geometry: Geometry,
    loss_adapter: LossAdapter,
) -> DetectorCenterSeed:
    """Estimate a cheap detector-u initializer from projection centre-of-mass evidence.

    This is deliberately only an initializer/diagnostic. It does not choose the
    calibration result; setup geometry is solved by the active-state objective path.
    """
    y = np.asarray(projections, dtype=np.float32)
    n_views, _nv, nu = y.shape
    u = np.arange(nu, dtype=np.float32) - (float(nu) - 1.0) * 0.5
    mask = getattr(loss_adapter.state, "mask", None)
    if mask is not None:
        weights = np.asarray(mask, dtype=np.float32)
    else:
        shifted = y - np.min(y, axis=(1, 2), keepdims=True)
        weights = np.maximum(shifted, 0.0)
    denom = np.sum(weights, axis=(1, 2))
    numerator = np.sum(weights * u[None, None, :], axis=(1, 2))
    valid = denom > 1e-6
    if int(np.count_nonzero(valid)) < 3:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="insufficient_projection_mass",
        )
    com = np.zeros((n_views,), dtype=np.float32)
    com[valid] = numerator[valid] / denom[valid]
    theta = np.deg2rad(np.asarray(geometry.thetas_deg, dtype=np.float32))
    design = np.stack(
        [
            np.ones_like(theta[valid]),
            np.cos(theta[valid]),
            np.sin(theta[valid]),
        ],
        axis=1,
    )
    coeffs, *_ = np.linalg.lstsq(design, com[valid], rcond=None)
    intercept = float(coeffs[0])
    amplitude = float(np.hypot(coeffs[1], coeffs[2]))
    return DetectorCenterSeed(
        det_u_px=intercept,
        intercept_px=intercept,
        amplitude_px=amplitude,
        status="ok",
    )


def projection_pair_det_u_seed(
    projections: jnp.ndarray,
    geometry: Geometry,
    *,
    max_pair_angle_error_deg: float = 2.0,
) -> DetectorCenterSeed:
    """Estimate detector-u centre from mirrored 0/180 projection pairs.

    For parallel-beam data, projections separated by 180 degrees should match
    after reversing detector-u. A detector-centre error appears as half the
    measured residual horizontal lag, with opposite sign in TomoJAX's
    detector-centre convention.
    """
    y = np.asarray(projections, dtype=np.float32)
    if y.ndim != 3:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="invalid_projection_shape",
        )
    n_views, _nv, nu = y.shape
    if n_views < 2 or nu < 4:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="insufficient_projection_pairs",
        )

    theta = np.asarray(geometry.thetas_deg, dtype=np.float32).reshape(-1)
    if theta.size != n_views:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="angle_projection_count_mismatch",
        )

    max_err = float(max_pair_angle_error_deg)
    pair_estimates: list[float] = []
    pair_lags: list[float] = []
    used: set[tuple[int, int]] = set()
    for i, angle in enumerate(theta):
        target = (float(angle) + 180.0) % 360.0
        diffs = np.abs(((theta - target + 180.0) % 360.0) - 180.0)
        j = int(np.argmin(diffs))
        if i == j or float(diffs[j]) > max_err:
            continue
        key = (min(i, j), max(i, j))
        if key in used:
            continue
        used.add(key)
        lag = _mirrored_projection_lag_px(y[i], y[j])
        if not np.isfinite(lag):
            continue
        pair_lags.append(float(lag))
        pair_estimates.append(float(-0.5 * lag))

    if not pair_estimates:
        return DetectorCenterSeed(
            det_u_px=0.0,
            intercept_px=0.0,
            amplitude_px=0.0,
            status="no_usable_opposite_angle_pairs",
        )

    estimates = np.asarray(pair_estimates, dtype=np.float32)
    lags = np.asarray(pair_lags, dtype=np.float32)
    det_u = float(np.median(estimates))
    return DetectorCenterSeed(
        det_u_px=det_u,
        intercept_px=det_u,
        amplitude_px=float(np.median(np.abs(lags))),
        status=f"ok_pairs={int(estimates.size)}",
    )


def _mirrored_projection_lag_px(a: np.ndarray, b: np.ndarray) -> float:
    """Return subpixel u-lag after mirroring the second projection."""
    profile_a = _projection_u_profile(a)
    profile_b = _projection_u_profile(np.flip(b, axis=1))
    if profile_a.size != profile_b.size or profile_a.size < 4:
        return float("nan")
    corr = np.correlate(profile_a, profile_b, mode="full")
    peak = int(np.argmax(corr))
    lag = float(peak - (profile_a.size - 1))
    if 0 < peak < corr.size - 1:
        y0, y1, y2 = float(corr[peak - 1]), float(corr[peak]), float(corr[peak + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            lag += 0.5 * (y0 - y2) / denom
    return lag


def _projection_u_profile(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(img)
    if not bool(np.any(finite)):
        return np.zeros((int(img.shape[1]),), dtype=np.float32)
    safe = np.where(finite, img, np.nanmedian(img[finite]))
    lo, hi = np.percentile(safe, [1.0, 99.0])
    clipped = np.clip(safe, lo, hi)
    profile = np.sum(clipped - np.median(clipped), axis=0)
    profile = profile.astype(np.float32, copy=False)
    profile -= float(np.mean(profile))
    scale = float(np.std(profile))
    if scale <= 1e-6 or not np.isfinite(scale):
        return np.zeros_like(profile, dtype=np.float32)
    return profile / scale
