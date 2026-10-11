"""Global per-view detector shift search by projection matching.

Local Gauss-Newton alignment converges only from poses close to the truth. Large
per-view stage shifts are instead found by cross-correlation, which searches
every shift up to a bound: reconstruct with the current shifts removed,
reproject, and move each measured view to its correlation peak against the
reprojection. Repeating sharpens the reconstruction and the estimates.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import logging
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import next_fast_len

from tomojax.core.cone import cone_model
from tomojax.core.geometry.cone import segments_of
from tomojax.core.geometry.views import stack_view_poses

if TYPE_CHECKING:
    from tomojax.alignment._config import AlignConfig
    from tomojax.geometry import Detector, Geometry, Grid

LOG = logging.getLogger(__name__)

# Views per chunk: correlation spectra are padded to 1.5x the detector size.
_CHUNK = 32


@jax.jit
def _shift_views(images: jax.Array, shifts: jax.Array) -> jax.Array:
    """Move each view by ``shifts`` pixels (u, v) with linear interpolation and zero fill."""
    _, nv, nu = images.shape
    v, u = jnp.meshgrid(
        jnp.arange(nv, dtype=jnp.float32), jnp.arange(nu, dtype=jnp.float32), indexing="ij"
    )

    def one(image: jax.Array, shift: jax.Array) -> jax.Array:
        coords = jnp.stack([v - shift[1], u - shift[0]])
        return jax.scipy.ndimage.map_coordinates(image, list(coords), order=1, mode="constant")

    return jax.vmap(one)(images, shifts)


@partial(jax.jit, static_argnames=("max_u", "max_v"))
def _correlation_peaks(
    measured: jax.Array, model: jax.Array, *, max_u: int, max_v: int
) -> jax.Array:
    """Return the sub-pixel shift (u, v) carrying each model view onto the measured view."""
    _, nv, nu = measured.shape
    # At searched |lag| <= M, replicas of a length-N linear correlation
    # cannot overlap once the FFT length is at least N + M.
    size = (
        cast("int", next_fast_len(nv + max_v, real=True)),
        cast("int", next_fast_len(nu + max_u, real=True)),
    )

    def centred(images: jax.Array) -> jax.Array:
        # Subtracting a sample first keeps a constant view exactly zero.
        images = images - images[:, :1, :1]
        return images - jnp.mean(images, axis=(1, 2), keepdims=True)

    measured, model = centred(measured), centred(model)
    spectrum = jnp.fft.rfft2(measured, s=size) * jnp.conj(jnp.fft.rfft2(model, s=size))
    corr = jnp.fft.irfft2(spectrum, s=size)
    # Zero padding makes these bounded lags linear, not circular overlaps.
    lag_v, lag_u = jnp.arange(-max_v, max_v + 1), jnp.arange(-max_u, max_u + 1)
    corr = corr[:, lag_v[:, None] % size[0], lag_u[None, :] % size[1]]
    overlap = (
        jnp.maximum(nv - jnp.abs(lag_v), 0)[:, None] * jnp.maximum(nu - jnp.abs(lag_u), 0)[None, :]
    )
    count = jnp.maximum(overlap, 1)
    # Compare the actual overlapping rectangles with local means and norms.
    # Unnormalised correlation favours large overlaps and bright cropped
    # features, even when a smaller overlap matches the true shift exactly.
    measured_sum = _overlap_sums(measured, lag_v, lag_u)
    model_sum = _overlap_sums(model, -lag_v, -lag_u)
    measured_var = jnp.maximum(
        _overlap_sums(measured * measured, lag_v, lag_u) - measured_sum**2 / count, 0.0
    )
    model_var = jnp.maximum(
        _overlap_sums(model * model, -lag_v, -lag_u) - model_sum**2 / count, 0.0
    )
    product = measured_var * model_var
    corr = (corr - measured_sum * model_sum / count) / jnp.sqrt(
        jnp.where(product > 0, product, 1.0)
    )
    corr = jnp.where((overlap > 0) & (product > 0), corr, -jnp.inf)

    peaks = jax.vmap(_correlation_peak)(corr)
    return peaks - jnp.asarray([max_u, max_v])


def _correlation_peak(c: jax.Array) -> jax.Array:
    """Refine a bounded correlation peak, including coupling between detector axes."""
    maximum = jnp.max(c)
    near = c >= maximum - 8 * jnp.finfo(c.dtype).eps * jnp.abs(maximum)
    v = jnp.arange(c.shape[0]) - c.shape[0] // 2
    u = jnp.arange(c.shape[1]) - c.shape[1] // 2
    # Among equal maxima (a feature-free axis) take the one closest to no shift.
    distance = v[:, None] ** 2 + u[None, :] ** 2
    flat = jnp.argmin(jnp.where(near, distance, jnp.iinfo(distance.dtype).max))
    iv, iu = flat // c.shape[1], flat % c.shape[1]

    def refine(values: jax.Array) -> jax.Array:
        left, centre, right = values
        curvature = left - 2 * centre + right
        tolerance = 8 * jnp.finfo(c.dtype).eps * jnp.abs(centre)
        ok = jnp.isfinite(left) & jnp.isfinite(right) & (curvature < -tolerance)
        return jnp.where(ok, 0.5 * (left - right) / jnp.where(ok, curvature, -1.0), 0.0)

    interior_u = (iu > 0) & (iu < c.shape[1] - 1)
    interior_v = (iv > 0) & (iv < c.shape[0] - 1)
    # Independent parabolas where the coupled fit is unavailable.
    separate = jnp.stack(
        [
            jnp.where(
                interior_u, refine(c[iv, jnp.clip(iu + jnp.arange(-1, 2), 0, c.shape[1] - 1)]), 0.0
            ),
            jnp.where(
                interior_v, refine(c[jnp.clip(iv + jnp.arange(-1, 2), 0, c.shape[0] - 1), iu]), 0.0
            ),
        ]
    )
    fallback = jnp.stack([iu, iv]) + separate
    # An elongated peak's best lattice point can lie more than a pixel from its
    # maximum: recentre on the fitted maximum, a pixel at a time, before accepting it.
    step, valid = _correlation_peak_step(c, iv, iu)
    for _ in range(2):
        move = jnp.where(valid, jnp.rint(jnp.clip(step, -1, 1)), 0).astype(iu.dtype)
        iu, iv = iu + move[0], iv + move[1]
        step, valid = _correlation_peak_step(c, iv, iu)
    peak = jnp.stack([iu, iv]) + step
    return jnp.where(valid & jnp.all(jnp.abs(step) <= 1.0), peak, fallback)


def _correlation_peak_step(
    c: jax.Array, iv: jax.Array, iu: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The stationary point of a coupled log-quadratic local correlation fit."""
    rows = jnp.clip(iv + jnp.arange(-1, 2), 0, c.shape[0] - 1)
    cols = jnp.clip(iu + jnp.arange(-1, 2), 0, c.shape[1] - 1)
    patch = c[rows[:, None], cols[None, :]]
    positive = jnp.all(patch > 0)
    patch = jnp.log(jnp.where(patch > 0, patch, 1.0))
    hx = patch[1, 0] - 2 * patch[1, 1] + patch[1, 2]
    hy = patch[0, 1] - 2 * patch[1, 1] + patch[2, 1]
    cross = (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0]) / 4
    gx, gy = (patch[1, 2] - patch[1, 0]) / 2, (patch[2, 1] - patch[0, 1]) / 2
    determinant = hx * hy - cross * cross
    interior = (iu > 0) & (iu < c.shape[1] - 1) & (iv > 0) & (iv < c.shape[0] - 1)
    tolerance = 8 * jnp.finfo(c.dtype).eps
    curved = (hx < -tolerance) & (hy < -tolerance)
    curved &= determinant > tolerance * jnp.maximum(jnp.abs(hx), jnp.abs(hy))
    valid = interior & positive & jnp.all(jnp.isfinite(patch)) & curved
    denominator = jnp.where(valid, determinant, 1.0)
    step = jnp.stack([(cross * gy - hy * gx) / denominator, (cross * gx - hx * gy) / denominator])
    return step, valid


def _overlap_sums(images: jax.Array, lag_v: jax.Array, lag_u: jax.Array) -> jax.Array:
    """Sum each image over the rectangles overlapping an image shifted by these lags."""
    _, nv, nu = images.shape
    integral = jnp.pad(jnp.cumsum(jnp.cumsum(images, axis=1), axis=2), ((0, 0), (1, 0), (1, 0)))
    v0, v1 = jnp.clip(lag_v, 0, nv)[:, None], jnp.clip(nv + lag_v, 0, nv)[:, None]
    u0, u1 = jnp.clip(lag_u, 0, nu)[None, :], jnp.clip(nu + lag_u, 0, nu)[None, :]
    return integral[:, v1, u1] - integral[:, v0, u1] - integral[:, v1, u0] + integral[:, v0, u0]


def _detector_translation_map(
    geometry: Geometry, detector: Detector, poses: np.ndarray
) -> np.ndarray:
    """Physical detector-u/v motion per lab translation at the object origin."""
    model = cone_model(geometry, detector)
    n = len(poses)
    if model is None:
        return np.broadcast_to(np.eye(3)[[0, 2]], (n, 2, 3))
    frames = np.asarray(model.frames(n), np.float64)
    centre, u, v, source = (frames[:, i] for i in range(4))
    normal = np.cross(u, v)
    ray = poses[:, :3, 3] - source
    denominator = np.sum(normal * ray, axis=1)
    distance = np.sum(normal * (centre - source), axis=1)
    axes = np.stack([u, v], axis=1)
    along = np.einsum("nki,ni->nk", axes, ray)
    return (distance / denominator)[:, None, None] * (
        axes - along[:, :, None] * normal[:, None, :] / denominator[:, None, None]
    )


def _remove_object_translation(
    shifts: np.ndarray, translation_basis: np.ndarray, spacing: tuple[float, float]
) -> np.ndarray:
    """Remove the shift pattern of one rigid object translation, which the volume absorbs."""
    physical = shifts * np.asarray(spacing)
    basis = translation_basis.reshape(-1, 3)
    translation, *_ = np.linalg.lstsq(basis, physical.reshape(-1), rcond=1e-8)
    residual = physical - (basis @ translation).reshape(-1, 2)
    return residual / np.asarray(spacing)


def _reprojector(
    geometry: Geometry, poses: jax.Array, grid: Grid, detector: Detector
) -> Callable[[slice, jax.Array], jax.Array]:
    """Return ``(views, volume) -> projections`` for the geometry's beam."""
    model = cone_model(geometry, detector)
    if model is not None:
        from tomojax.core.cone import cone_project

        cone = model.coefficients(poses, grid)
        return lambda views, volume: cone_project(volume, cone[views], grid, detector)
    from tomojax.core.joseph import forward_project_planes, plane_coefficients

    coefficients = plane_coefficients(poses, grid, detector)
    backend = "pallas" if jax.default_backend() == "gpu" else "jax"
    return lambda views, volume: forward_project_planes(
        coefficients[views], volume, grid, detector, backend=backend
    )


def _quick_reconstruction(
    geometry: Geometry, grid: Grid, detector: Detector, data: jax.Array
) -> jax.Array:
    """A fast reconstruction to correlate against.

    Filtered backprojection, or for segmented cone scans, which FDK cannot
    take, a few CGLS iterations.
    """
    from tomojax.recon import CGLSConfig, FBPConfig, cgls, fbp

    if segments_of(geometry) is None:
        return fbp(geometry, grid, detector, data, config=FBPConfig(filter="hann"))
    # From the host, CGLS streams the views and keeps no projection-sized vector.
    config = CGLSConfig(iterations=6, stream_projections=True)
    volume, _ = cgls(geometry, grid, detector, np.asarray(data), config=config)
    return volume


def estimate_view_shifts(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    *,
    iters: int = 8,
    max_shift_fraction: float = 0.25,
    tolerance_px: float = 0.05,
) -> np.ndarray:
    """Estimate per-view detector shifts by iterated projection matching.

    Returns an ``(n_views, 2)`` array of physical (u, v) shifts: positive
    values mean the object appears displaced along the physical detector axes
    +u/+v in that view, not necessarily along lab x/z. Shifts up to
    ``max_shift_fraction`` of the detector size are searched. The part of the
    shifts produced by one rigid object translation is removed, because the
    reconstruction absorbs it.
    """
    data = jnp.asarray(projections, dtype=jnp.float32)
    n = int(data.shape[0])
    poses = stack_view_poses(geometry, n)
    nominal = np.asarray(poses, np.float64)
    translation_basis = _detector_translation_map(geometry, detector, nominal) @ nominal[:, :3, :3]
    reproject = _reprojector(geometry, poses, grid, detector)

    max_u = max(1, int(max_shift_fraction * detector.nu))
    max_v = max(1, int(max_shift_fraction * detector.nv))
    chunks = [slice(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]

    def correlate(volume: jax.Array) -> np.ndarray:
        # Keep only the small peak tables, then synchronize once per iteration.
        return np.asarray(
            jnp.concatenate(
                [
                    _correlation_peaks(
                        data[c],
                        reproject(c, volume),
                        max_u=max_u,
                        max_v=max_v,
                    )
                    for c in chunks
                ]
            ),
            np.float64,
        )

    spacing = (float(detector.du), float(detector.dv))
    shifts = np.zeros((n, 2))
    for iteration in range(int(iters)):
        corrected = jnp.concatenate(
            [_shift_views(data[c], jnp.asarray(-shifts[c], jnp.float32)) for c in chunks]
        )
        volume = _quick_reconstruction(geometry, grid, detector, corrected)
        del corrected
        peaks = correlate(volume)
        updated = _remove_object_translation(peaks, translation_basis, spacing)
        change = float(np.max(np.abs(updated - shifts)))
        shifts = updated
        LOG.info("Shift search %d: max change %.3f px", iteration + 1, change)
        if change <= tolerance_px:
            break
    return shifts * np.asarray(spacing)


def reprojection_residual(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
) -> Callable[[float], float]:
    """Return ``shift_px -> relative residual`` of a quick reconstruction's reprojection.

    The views are moved back by a common detector-u shift, reconstructed (FBP, or
    a few CGLS iterations for segmented cone scans) and reprojected; the residual
    after a best scalar amplitude fit is smallest when the shift matches the data.
    """
    data = jnp.asarray(projections, dtype=jnp.float32)
    n = int(data.shape[0])
    reproject = _reprojector(geometry, stack_view_poses(geometry, n), grid, detector)
    chunks = [slice(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]
    norm = float(jnp.linalg.norm(data))

    def residual(shift_px: float) -> float:
        offset = jnp.asarray([[-shift_px, 0.0]], jnp.float32)
        corrected = jnp.concatenate(
            [_shift_views(data[c], jnp.broadcast_to(offset, (c.stop - c.start, 2))) for c in chunks]
        )
        volume = _quick_reconstruction(geometry, grid, detector, corrected)
        predicted = jnp.concatenate([reproject(c, volume) for c in chunks])
        scale = jnp.vdot(predicted, corrected) / jnp.maximum(jnp.vdot(predicted, predicted), 1e-30)
        return float(jnp.linalg.norm(scale * predicted - corrected)) / max(norm, 1e-30)

    return residual


def translation_params_from_shifts(
    shifts: np.ndarray,
    translation_map: np.ndarray,
    params: np.ndarray,
    *,
    active: tuple[bool, bool],
) -> np.ndarray:
    """Fit active (dx, dz) to physical u/v shifts through their calibrated map.

    ``translation_map`` maps parameter-frame (x, y, z) translations to
    detector-u/v motion. Near-null directions are discarded rather than turned
    into enormous offsets. Every frozen translation, including dy in a
    six-column pose table, is kept and its contribution is subtracted first.
    """
    mapping = translation_map[:, :, [0, 2]]
    mask = np.asarray(active, bool)
    fixed = np.where(mask, 0.0, params[:, 3:5])
    residual = shifts - np.einsum("nij,nj->ni", mapping, fixed)
    if params.shape[1] > 5:
        residual = residual - translation_map[:, :, 1] * params[:, 5, None]
    inverse = np.linalg.pinv(mapping * mask[None, None, :], rtol=1e-3)
    estimate = np.einsum("nij,nj->ni", inverse, residual)
    seeded = np.array(params, dtype=np.float32, copy=True)
    seeded[:, 3:5] = np.where(mask, estimate, fixed)
    return seeded


def seeded_translation_params(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jax.Array | np.ndarray,
    cfg: AlignConfig,
) -> jax.Array | None:
    """Return initial poses with searched translations, or None when not requested.

    Local solvers converge only near the truth; a global shift search first
    brings large per-view stage shifts within reach.
    """
    from tomojax.alignment._config import _active_dof_mask_for_cfg

    active = _active_dof_mask_for_cfg(cfg)
    if not cfg.seed_translations or not any(active[3:5]):
        return None
    n = int(np.shape(projections)[0])
    shifts = estimate_view_shifts(geometry, grid, detector, projections)
    poses = np.asarray(stack_view_poses(geometry, n), np.float64)
    mapping = _detector_translation_map(geometry, detector, poses)
    if cfg.pose_translation_frame == "object":
        mapping = mapping @ poses[:, :3, :3]
    params = translation_params_from_shifts(
        shifts,
        mapping,
        np.zeros((n, len(active)), np.float32),
        active=(bool(active[3]), bool(active[4])),
    )
    LOG.info(
        "Seeded translations from a shift search: rms %.3f, max %.3f (physical units)",
        float(np.sqrt(np.mean(shifts**2))),
        float(np.max(np.abs(shifts))),
    )
    return jnp.asarray(params)


def implied_detector_offset(nominal: np.ndarray, aligned: np.ndarray) -> tuple[float, float]:
    """Return the detector-u centre offset implied by recovered poses, and the fit residual.

    A detector-centre (centre-of-rotation) offset displaces every view's image by
    the same amount along u, while a rigid object translation displaces it by a
    view-dependent amount. Fitting each view's u displacement of the object origin
    as ``-offset + (R_i t)_x`` separates the two over the scan. Both returns are
    physical lengths, with the sign of ``Detector.center``.
    """
    shift = np.asarray(aligned, np.float64)[:, 0, 3] - np.asarray(nominal, np.float64)[:, 0, 3]
    design = np.column_stack([-np.ones(len(shift)), np.asarray(nominal, np.float64)[:, 0, :3]])
    solution, *_ = np.linalg.lstsq(design, shift, rcond=None)
    residual = shift - design @ solution
    return float(solution[0]), float(np.sqrt(np.mean(residual**2)))
