"""Estimate the data discrepancy at the resolution of the reconstruction grid."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import numpy as np

from tomojax._typed_arrays import Float64Array, numpy_float64_array
from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.robust import MAD_TO_SD

if TYPE_CHECKING:
    import jax

    from tomojax.core import checkpoint
    from tomojax.geometry import Detector, Geometry, Grid


@runtime_checkable
class _Bands(Protocol):
    def band(self, first: int, stop: int) -> tuple[np.ndarray, int, int]: ...


def _patch(
    data: jax.Array | np.ndarray | _Bands, view: int, row: int, col: int, h: int, w: int
) -> Float64Array:
    if not isinstance(data, _Bands):
        return np.asarray(data[view, row : row + h, col : col + w], dtype=np.float64)
    pixels = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        buffer, first, _ = data.band(row + i, row + i + 1)
        pixels[i] = buffer[view, first, col : col + w]
    return pixels


@lru_cache(maxsize=32)
def _cosines(size: int) -> Float64Array:
    """Orthonormal DCT-II on a detector patch with reflecting boundaries."""
    k = np.arange(size, dtype=np.float64)
    basis = numpy_float64_array(np.cos((np.pi / size) * k[:, None] * (k + 0.5)))
    basis *= np.sqrt(2 / size)
    basis[0] /= np.sqrt(2)
    return basis


def _gain(u: Float64Array, v: Float64Array, *, aligned: bool) -> Float64Array:
    """Power of interpolation phase fluctuations and unresolved frequencies."""
    second_u = (2 + np.cos(2 * np.pi * u)) / 3
    mean_u = np.sinc(u) ** 2
    second_v = 1 if aligned else (2 + np.cos(2 * np.pi * v)) / 3
    mean_v = 1 if aligned else np.sinc(v) ** 2
    variance = second_u * second_v - (mean_u * mean_v) ** 2
    inside_nyquist = u**2 + v**2 <= 0.25
    return np.where(inside_nyquist, variance, 1)


def _haar(patch: Float64Array) -> Float64Array:
    for axis in (0, 1):
        if patch.shape[axis] > 1:
            difference = np.diff(patch, axis=axis)
            indices = range(0, patch.shape[axis] - 1, 2)
            patch = numpy_float64_array(np.take(difference, indices, axis=axis)) / math.sqrt(2)
    return patch.ravel()


class Calibration(NamedTuple):
    """Measurement SD and estimated total RMS error in projection units."""

    noise: float
    rms: float

    def target(self, size: int) -> float:
        return self.rms**2 * size


def estimate(
    data: np.ndarray | jax.Array,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    *,
    linear: bool = True,
    centered: bool = False,
) -> Calibration | None:
    """Estimate noise and resolution-limited power on at most 65,536 detector pixels.

    The centered Haar MAD estimates white Gaussian measurement noise. Below the
    isotropic voxel Nyquist frequency, use the variance of linear interpolation
    over a uniform subvoxel phase. Outside this band, count the full signal
    power. Subtract the contribution of white noise to both estimates of signal
    power before adding its measurement variance. Frequencies use isocenter
    magnification and voxel pitches along the two detector axes. Other
    interpolation models use the measurement term alone.
    """
    views, rows, cols = data.shape
    if rows * cols == 1:
        return None
    h, w = min(32, rows), min(32, cols)
    rng = np.random.default_rng(0)
    count = min(views * (rows // h) * (cols // w), 65536 // (h * w))
    selected = (np.arange(count) + rng.random(count)) * views / count
    rs = rng.integers(rows - h + 1, size=count)
    cs = rng.integers(cols - w + 1, size=count)
    parts = cone_parts(geometry, detector)
    ends = np.cumsum([n or views for n, _, _ in parts]) if parts else None
    fu, fv = np.arange(w) / (2 * w * detector.du), np.arange(h)[:, None] / (2 * h * detector.dv)
    pitch = np.array((grid.vx, grid.vy, grid.vz))
    row_offset = detector.center[1] - (detector.nv - 1) * detector.dv / 2
    first_z = grid_volume_origin(grid)[2]
    details: list[Float64Array] = []
    energy = gain = 0.0
    for index in sorted(range(count), key=rs.__getitem__):
        i = int(selected[index])
        patch = _patch(data, i, int(rs[index]), int(cs[index]), h, w)
        details.append(_haar(patch))
        if linear:
            pose = np.asarray(geometry.pose_matrix(i), dtype=np.float64)
            rotation = pose[:3, :3]
            axes = rotation[[0, 2]]
            if parts and ends is not None:
                _, beam, det = parts[int(np.searchsorted(ends, i, side="right"))]
                _, u_dir, v_dir = beam.detector_frame(det)
                axes = beam.magnification * (np.stack([u_dir, v_dir]) @ rotation)
            pitches = numpy_float64_array(np.linalg.norm(axes * pitch, axis=1))
            u, v = float(pitches[0]), float(pitches[1])
            z = float(pose[2, 2])
            row_phase = (z * first_z + float(pose[2, 3]) - row_offset) / detector.dv
            aligned = (
                parts is None and abs(z) == 1 and detector.dv == grid.vz and row_phase.is_integer()
            )
            rejected = _gain(fu * u, fv * v, aligned=aligned)
            spectrum = _cosines(h) @ patch @ _cosines(w).T
            energy += float(np.mean(rejected * spectrum**2))
            gain += float(np.mean(rejected))
    high = np.concatenate(details)
    median = float(np.median(high))
    sigma = float(np.median(np.abs(high - median))) * MAD_TO_SD
    model = max(0, (energy - sigma**2 * gain) / count)
    noise_power = sigma**2 * (1 - int(centered) / (rows * cols))
    return Calibration(sigma, math.sqrt(noise_power + model))


def sirt_stopping(
    count: int, energy: float | jax.Array | None, target: float | None, size: int
) -> dict[str, object]:
    """Discrepancy at the point projected for the last complete SIRT sweep."""
    reached = target is not None and energy is not None and float(energy) <= target
    return {
        "effective_iterations": count,
        "termination": "projection_discrepancy" if reached else "iteration_limit",
        "early_stop": reached,
        "projection_error_rms": None if target is None else (target / size) ** 0.5,
        "pre_update_residual_rms": None if energy is None else (float(energy) / size) ** 0.5,
    }


def sirt_slab_report(finished: checkpoint.Slabs) -> dict[str, object]:
    """Summarize independent solves using a voxel-depth-weighted RMS."""
    blocks = sorted(finished.reports.items())
    info: dict[str, object] = dict(blocks[0][1])
    info["slab_stopping"] = [
        {"start": start, "stop": stop, **record} for (start, stop), record in blocks
    ]
    depth = sum(stop - start for (start, stop), _ in blocks)
    counts = [
        value for _, record in blocks if isinstance(value := record["effective_iterations"], int)
    ]
    info["effective_iterations"] = info["subset_updates"] = max(counts)
    info["early_stop"] = all(record["early_stop"] for _, record in blocks)
    if any(record["termination"] == "iteration_limit" for _, record in blocks):
        info["termination"] = "iteration_limit"
    for name in ("projection_error_rms", "measurement_noise_rms", "pre_update_residual_rms"):
        energy = 0.0
        for (start, stop), record in blocks:
            value = record[name]
            if not isinstance(value, float):
                info[name] = None
                break
            energy += (stop - start) * value**2
        else:
            info[name] = (energy / depth) ** 0.5
    return info
