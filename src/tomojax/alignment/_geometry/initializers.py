"""Initializers and deterministic view splits for setup geometry alignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy import fft

from .axis_initializer import axis_in_detector, fit_axis_direction
from .common_lines import prepare_common_lines

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

    from tomojax.alignment._model.dofs import DofBounds
    from tomojax.alignment._model.state import SetupGeometryState
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


def common_line_setup_seed(
    projections: np.ndarray | jax.Array,
    geometry: Geometry,
    *,
    detector: Detector,
    setup: SetupGeometryState,
    active_dofs: tuple[str, ...],
    bounds: DofBounds,
    factor: int,
) -> dict[str, object] | None:
    """Seed an axis and then its centre without fitting projection translations."""
    from tomojax.core.geometry import LaminographyGeometry, ParallelGeometry, RotationAxisGeometry

    active = ("axis_rot_x_deg" in active_dofs, "axis_rot_y_deg" in active_dofs)
    if not isinstance(geometry, LaminographyGeometry | ParallelGeometry | RotationAxisGeometry):
        return None
    if not any(active) and (
        "det_u_px" not in active_dofs or isinstance(geometry, ParallelGeometry)
    ):
        return None
    # The v origin is fixed during this seed; include any existing calibration.
    detector_v = replace(
        detector,
        center=(
            detector.center[0],
            detector.center[1] + float(setup.det_v_px) / factor * detector.dv,
        ),
    )
    lines = prepare_common_lines(projections, np.asarray(geometry.angles), detector_v)
    if lines is None:
        return None
    nominal = np.asarray(setup.nominal_axis_unit, np.float64)
    if isinstance(geometry, LaminographyGeometry):
        nominal = np.asarray(
            replace(
                geometry, tilt_deg=geometry.tilt_deg + np.rad2deg(float(setup.tilt_rad))
            ).axis_unit_lab
        )
    else:
        nominal = axis_in_detector(
            nominal, np.rad2deg([float(setup.tilt_rad), 0.0]), detector_roll_deg=0.0
        )
    rotations = np.rad2deg([float(setup.axis_rot_x_rad), float(setup.axis_rot_y_rad)])
    roll = np.rad2deg(float(setup.detector_roll_rad))
    result: dict[str, object] = {}
    if any(active):
        limits = {name: np.rad2deg([lo, hi]) for name, lo, hi in bounds}
        angular_bounds = np.asarray(
            [
                limits.get("axis_rot_x_deg", (-180.0, 180.0)),
                limits.get("axis_rot_y_deg", (-60.0, 60.0)),
            ]
        )
        for i in np.flatnonzero(active):
            rotations[i] = np.clip(rotations[i], *angular_bounds[i])
        fit = fit_axis_direction(
            lines.sample,
            nominal,
            rotations_deg=(float(rotations[0]), float(rotations[1])),
            active=active,
            bounds_deg=(
                (float(angular_bounds[0, 0]), float(angular_bounds[0, 1])),
                (float(angular_bounds[1, 0]), float(angular_bounds[1, 1])),
            ),
            detector_roll_deg=roll,
        )
        rotations = np.asarray(fit.rotations_deg)
        result.update(
            axis_seed_applied=True,
            axis_seed_rotations_deg=fit.rotations_deg,
            axis_seed_correlation_before=fit.correlation_before,
            axis_seed_correlation_after=fit.correlation_after,
            axis_seed_on_search_boundary=fit.on_search_boundary,
        )
    if "det_u_px" in active_dofs:
        center = lines.center(
            axis_in_detector(nominal, rotations, detector_roll_deg=roll), detector_roll_deg=roll
        )
        if center is not None:
            result.update(
                detector_center_seed_applied=True,
                detector_center_seed_method="fourier_common_line_phase",
                detector_center_seed_det_u_px=(center - detector.center[0] / detector.du) * factor,
            )
    return result or None


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
    by FBP (a few CGLS iterations for segmented cone scans) and reprojected; a
    consistent offset minimises the relative reprojection residual after a best
    scalar fit. A coarse scan over
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
    wedge, so the offset minimises the energy there. The coarse search bins
    columns by up to 4 without reducing below 32, and averages dense angular
    rows without reducing below 128. Full-resolution searches refine to a
    quarter pixel; the result is the median over ``rows`` detector rows.
    ``ratio`` is the object's size as a fraction of the field (the wedge's
    width), ``drop`` the low frequencies left out. Nonuniformly spaced views
    are interpolated onto a uniform angular grid before the Fourier metric;
    repeated angles are averaged. None when the views cover less than a half
    turn.
    """
    raw_angles = np.asarray(angles_deg, np.float64)
    angles = np.mod(raw_angles, 360.0)
    order = np.argsort(angles, kind="stable")
    # Cut the circular sequence at its unobserved arc, not at degree zero.
    # Individual whole turns and acquisition order cannot change the views
    # used by the half-turn sinogram. Equal gaps keep the lowest-angle start.
    gaps = np.diff(np.r_[angles[order[-1]] - 360.0, angles[order]])
    roundoff = 8 * np.spacing(max(360.0, float(np.max(np.abs(raw_angles)))))
    start = int(np.flatnonzero(gaps >= np.max(gaps) - roundoff)[0])
    order = np.roll(order, -start)
    relative = np.mod(angles[order] - angles[order[0]], 360.0)
    first = relative < 180.0 - 1e-6
    half_turn = order[first]
    if len(half_turn) < 16 or relative[first][-1] < 175.0:
        return None
    nv, nu = projections.shape[1], projections.shape[2]
    picks = np.unique(np.linspace(0.1 * (nv - 1), 0.9 * (nv - 1), max(1, rows)).round().astype(int))
    # Transfer only the sampled rows, not a potentially multi-GB device scan.
    # Select them together so a device input needs just one gather and copy.
    data = np.asarray(projections[:, picks, :])
    offsets = []
    for row in range(len(picks)):
        sino = np.asarray(data[half_turn, row, :], np.float32)
        if not np.isfinite(sino).all() or np.ptp(sino) <= 0:
            continue
        sino = _uniform_sinogram(sino, relative[first])
        # Preserve enough detector samples for the coarse Fourier metric:
        # a fixed 4x reduction can alias narrow sinograms into a false basin.
        binning = max(1, min(4, nu // 32))
        # The coarse offset only needs the low angular frequencies; averaging
        # dense views avoids paying for their fine-resolution bandwidth here.
        angular_binning = max(2, len(sino) // 128)
        views = len(sino) // angular_binning * angular_binning
        coarse = (
            sino[:views, : nu // binning * binning]
            .reshape(views // angular_binning, angular_binning, -1, binning)
            .mean(axis=(1, 3))
        )
        reach = max(2, int(max_fraction * coarse.shape[1]))
        steps = np.arange(-reach, reach + 1, dtype=np.float64)
        best = steps[int(np.argmin(_vo_metrics(coarse, steps, ratio, drop)))] * binning
        # Dropping rightmost columns moves the coarse detector midpoint left.
        best -= (nu % binning) / 2
        # Fine: whole pixels around it, then quarter pixels.
        for spread, step in ((6.0, 1.0), (1.0, 0.25)):
            near = best + np.arange(-spread, spread + step / 2, step)
            metrics = _vo_metrics(sino, near, ratio, drop)
            best = float(near[int(np.argmin(metrics))])
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


def _uniform_sinogram(sino: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Equally spaced angular rows for Vo's angular Fourier transform."""
    unique, starts, counts = np.unique(angles, return_index=True, return_counts=True)
    if len(unique) != len(angles):
        sino = np.add.reduceat(sino, starts, axis=0) / counts[:, None]
    target = np.linspace(unique[0], unique[-1], len(unique))
    if np.allclose(unique, target, rtol=0, atol=1e-5):
        return sino.astype(np.float32, copy=False)
    right = np.clip(np.searchsorted(unique, target), 1, len(unique) - 1)
    left = right - 1
    weight = ((target - unique[left]) / (unique[right] - unique[left]))[:, None]
    return (sino[left] * (1 - weight) + sino[right] * weight).astype(np.float32)


def _vo_metrics(sino: np.ndarray, offsets: np.ndarray, ratio: float, drop: int) -> np.ndarray:
    """Vo's metric for each axis offset (pixels from the centre column) of ``sino``."""
    views, cols = sino.shape
    folded = _vo_folded_mask(views, cols, ratio, drop)
    mirror = sino[:, ::-1]
    # Shift the mirror by twice the offset (sub-pixel, along u, by Fourier
    # phase), with the far edge of the flipped sinogram where it wraps.
    freq = np.fft.rfftfreq(cols).astype(np.float32)
    spectrum = np.asarray(fft.rfft(mirror, axis=1, workers=-1))
    out = np.empty(len(offsets))
    both = np.empty((2 * views, cols), dtype=sino.dtype)
    both[:views] = sino
    for i, offset in enumerate(offsets):
        shift = 2.0 * float(offset)
        phase = np.exp(-2j * np.pi * freq * np.float32(shift)).astype(np.complex64)
        moved = np.asarray(fft.irfft(spectrum * phase, n=cols, axis=1, workers=-1))
        edge = int(np.ceil(abs(shift)))
        if edge:
            filler = sino[::-1]
            if shift > 0:
                moved[:, :edge] = filler[:, :edge]
            else:
                moved[:, -edge:] = filler[:, -edge:]
        both[views:] = moved
        energy = np.abs(np.asarray(fft.rfft2(both, workers=-1)))
        energy *= folded
        out[i] = float(np.sum(energy) / (2 * views * cols))
    return out


@lru_cache(maxsize=2)
def _vo_folded_mask(views: int, cols: int, ratio: float, drop: int) -> np.ndarray:
    """Reuse the immutable coarse/fine frequency weights across rows and searches."""
    mask = np.fft.ifftshift(_vo_mask(2 * views, cols, 0.5 * ratio * cols, drop))
    # Real sinograms have conjugate Fourier pairs. Fold the mask onto the
    # stored half-plane, including each DC/Nyquist column only once. The mask
    # itself is not symmetric, so simply doubling positive frequencies fails.
    folded = mask[:, : cols // 2 + 1].copy()
    stop = (cols + 1) // 2
    folded[:, 1:stop] += mask[(-np.arange(2 * views)) % (2 * views)][:, cols - 1 : cols - stop : -1]
    folded.flags.writeable = False
    return folded


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


__all__ = [
    "DetectorCenterSeed",
    "common_line_setup_seed",
    "fit_axis_direction",
    "reprojection_det_u_seed",
    "sinogram_det_u_seed",
]
