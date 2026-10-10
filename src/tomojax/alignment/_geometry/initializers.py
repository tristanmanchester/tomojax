"""Initializers and deterministic view splits for setup geometry alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import fft

if TYPE_CHECKING:
    import jax.numpy as jnp

    from tomojax.alignment._objectives.loss_adapters import LossAdapter
    from tomojax.core.geometry import Detector, Geometry, Grid


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
    theta = np.deg2rad(np.asarray(geometry.angles, dtype=np.float32))
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

    theta = np.asarray(geometry.angles, dtype=np.float32).reshape(-1)
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
    if not bool(np.all(np.isfinite(corr))) or float(np.ptp(corr)) <= 1e-6:
        return float("nan")
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


def reprojection_det_u_seed(
    projections: jnp.ndarray,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    *,
    max_fraction: float = 0.25,
    tolerance_px: float = 0.01,
) -> DetectorCenterSeed:
    """Find the detector-u offset whose FBP reprojects most consistently.

    For each candidate offset the projections are shifted back, reconstructed
    by FBP and reprojected; a consistent offset minimises the relative
    reprojection residual after a best scalar fit. A coarse scan over
    ``max_fraction`` of the detector width is refined by golden-section search.
    Works for any scan geometry, including laminography and partial arcs.
    """
    from tomojax.alignment._prealign import reprojection_residual

    residual = reprojection_residual(geometry, grid, detector, projections)
    half = max_fraction * detector.nu
    candidates = np.linspace(-half, half, 25)
    values = [residual(float(c)) for c in candidates]
    best = int(np.argmin(values))
    low = candidates[max(best - 1, 0)]
    high = candidates[min(best + 1, len(candidates) - 1)]
    ratio = (np.sqrt(5.0) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = residual(left), residual(right)
    while high - low > tolerance_px:
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - ratio * (high - low)
            f_left = residual(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (high - low)
            f_right = residual(right)
    shift_px = 0.5 * (low + high)
    # Data displaced by +s pixels correspond to a detector centre at -s pixels.
    return DetectorCenterSeed(
        det_u_px=-shift_px,
        intercept_px=-shift_px,
        amplitude_px=float(min(values)),
        status="ok_reprojection",
    )


def sinogram_det_u_seed(
    projections: np.ndarray | jnp.ndarray,
    angles_deg: np.ndarray,
    *,
    rows: int = 3,
    max_fraction: float = 0.25,
    ratio: float = 0.5,
    drop: int = 20,
) -> DetectorCenterSeed | None:
    """The rotation axis of a parallel scan over a half turn or more, by Vo et al. (2014).

    A sinogram over 180 degrees and its mirror image, stacked, make a
    360-degree sinogram when the mirror is shifted by twice the axis offset;
    a 360-degree sinogram's Fourier transform has no energy outside a double
    wedge, so the offset minimises the energy there. Coarse on columns binned
    by 4, then to a quarter pixel; the median over ``rows`` detector rows.
    ``ratio`` is the object's size as a fraction of the field (the wedge's
    width), ``drop`` the low frequencies left out. None when the views cover
    less than a half turn.
    """
    angles = np.asarray(angles_deg, np.float64)
    order = np.argsort(angles)
    first = angles[order] - angles[order][0] < 180.0 - 1e-6
    half_turn = order[first]
    if len(half_turn) < 16 or angles[half_turn[-1]] - angles[half_turn[0]] < 175.0:
        return None
    data = np.asarray(projections)
    nv, nu = data.shape[1], data.shape[2]
    picks = np.unique(np.linspace(0.1 * (nv - 1), 0.9 * (nv - 1), max(1, rows)).round().astype(int))
    offsets = []
    for row in picks:
        sino = np.asarray(data[half_turn, row, :], np.float32)
        if not np.isfinite(sino).all() or np.ptp(sino) <= 0:
            continue
        # Coarse: columns binned by 4 (and views by 2), every binned pixel.
        views = len(sino) // 2 * 2
        coarse = sino[:views, : nu // 4 * 4].reshape(views // 2, 2, -1, 4).mean(axis=(1, 3))
        reach = max(2, int(max_fraction * coarse.shape[1]))
        steps = np.arange(-reach, reach + 1, dtype=np.float64)
        best = steps[int(np.argmin(_vo_metrics(coarse, steps, ratio, drop)))] * 4
        # Fine: whole pixels around it, then quarter pixels.
        for spread, step in ((6.0, 1.0), (1.0, 0.25)):
            near = best + np.arange(-spread, spread + step / 2, step)
            best = float(near[int(np.argmin(_vo_metrics(sino, near, ratio, drop)))])
        offsets.append(best)
    if not offsets:
        return None
    offset = float(np.median(offsets))
    # An axis at +s pixels from the detector centre is a detector centre at -s.
    return DetectorCenterSeed(
        det_u_px=-offset,
        intercept_px=-offset,
        amplitude_px=float(np.std(offsets)) if len(offsets) > 1 else 0.0,
        status="ok_sinogram",
    )


def _vo_metrics(sino: np.ndarray, offsets: np.ndarray, ratio: float, drop: int) -> np.ndarray:
    """Vo's metric for each axis offset (pixels from the centre column) of ``sino``."""
    views, cols = sino.shape
    mask = _vo_mask(2 * views, cols, 0.5 * ratio * cols, drop)
    mirror = sino[:, ::-1]
    # Shift the mirror by twice the offset (sub-pixel, along u, by Fourier
    # phase), with the far edge of the flipped sinogram where it wraps.
    freq = np.fft.fftfreq(cols).astype(np.float32)
    spectrum = np.asarray(fft.fft(mirror, axis=1, workers=-1))
    out = np.empty(len(offsets))
    for i, offset in enumerate(offsets):
        shift = 2.0 * float(offset)
        phase = np.exp(-2j * np.pi * freq * np.float32(shift)).astype(np.complex64)
        moved = np.real(np.asarray(fft.ifft(spectrum * phase, axis=1, workers=-1)))
        edge = int(np.ceil(abs(shift)))
        if edge:
            filler = sino[::-1]
            if shift > 0:
                moved[:, :edge] = filler[:, :edge]
            else:
                moved[:, -edge:] = filler[:, -edge:]
        both = np.vstack([sino, moved])
        energy = np.abs(np.fft.fftshift(np.asarray(fft.fft2(both, workers=-1))))
        out[i] = float(np.mean(energy * mask))
    return out


def _vo_mask(rows: int, cols: int, radius: float, drop: int) -> np.ndarray:
    """Vo's double-wedge mask: where a centred 360-degree sinogram has no energy."""
    du, dv = 1.0 / cols, (rows - 1.0) / (rows * 2.0 * np.pi)
    centre_row, centre_col = int(np.ceil(rows / 2.0) - 1), int(np.ceil(cols / 2.0) - 1)
    reach = np.round((np.arange(rows) - centre_row) * dv / radius / du)
    low = np.clip(centre_col - np.abs(reach), 0, cols - 1).astype(int)
    high = np.clip(centre_col + np.abs(reach), 0, cols - 1).astype(int)
    col = np.arange(cols)[None, :]
    mask = ((col >= low[:, None]) & (col <= high[:, None])).astype(np.float32)
    drop = min(int(drop), int(np.ceil(0.05 * rows)))
    mask[centre_row - drop : centre_row + drop + 1, :] = 0.0
    mask[:, centre_col - 1 : centre_col + 2] = 0.0
    return mask
