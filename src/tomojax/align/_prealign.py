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
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.cone import beam_of
from tomojax.core.geometry.views import stack_view_poses

if TYPE_CHECKING:
    from tomojax.align._config import AlignConfig
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
    size = (nv + 2 * max_v, nu + 2 * max_u)

    def centred(images: jax.Array) -> jax.Array:
        return images - jnp.mean(images, axis=(1, 2), keepdims=True)

    spectrum = jnp.fft.rfft2(centred(measured), s=size) * jnp.conj(
        jnp.fft.rfft2(centred(model), s=size)
    )
    corr = jnp.fft.irfft2(spectrum, s=size)
    # Keep lags within the search bound; zero padding prevents wrapped overlaps.
    lag_v = jnp.fft.fftfreq(size[0], 1.0 / size[0])
    lag_u = jnp.fft.fftfreq(size[1], 1.0 / size[1])
    allowed = (jnp.abs(lag_v)[:, None] <= max_v) & (jnp.abs(lag_u)[None, :] <= max_u)
    corr = jnp.where(allowed[None], corr, -jnp.inf)

    def peak(c: jax.Array) -> jax.Array:
        flat = jnp.argmax(c)
        iv, iu = flat // size[1], flat % size[1]

        def refine(axis_values: jax.Array) -> jax.Array:
            left, centre, right = axis_values
            curvature = left - 2 * centre + right
            ok = jnp.isfinite(left) & jnp.isfinite(right) & (curvature < 0)
            return jnp.where(ok, 0.5 * (left - right) / jnp.where(ok, curvature, -1.0), 0.0)

        du = refine(c[iv, (iu + jnp.arange(-1, 2)) % size[1]])
        dv = refine(c[(iv + jnp.arange(-1, 2)) % size[0], iu])
        return jnp.stack([lag_u[iu] + du, lag_v[iv] + dv])

    return jax.vmap(peak)(corr)


def _remove_object_translation(
    shifts: np.ndarray, rotations: np.ndarray, spacing: tuple[float, float]
) -> np.ndarray:
    """Remove the shift pattern of one rigid object translation, which the volume absorbs."""
    physical = shifts * np.asarray(spacing)
    basis = rotations[:, [0, 2], :].reshape(-1, 3)
    translation, *_ = np.linalg.lstsq(basis, physical.reshape(-1), rcond=1e-8)
    residual = physical - (basis @ translation).reshape(-1, 2)
    return residual / np.asarray(spacing)


def _reprojector(
    geometry: Geometry, poses: jax.Array, grid: Grid, detector: Detector
) -> Callable[[slice, jax.Array], jax.Array]:
    """Return ``(views, volume) -> projections`` for the geometry's beam."""
    beam = beam_of(geometry)
    if beam is not None:
        from tomojax.core.cone import cone_coefficients, cone_project

        cone = cone_coefficients(poses, grid, detector, beam)
        return lambda views, volume: cone_project(volume, cone[views], grid, detector)
    from tomojax.core.joseph import forward_project_planes, plane_coefficients

    coefficients = plane_coefficients(poses, grid, detector)
    backend = "pallas" if jax.default_backend() == "gpu" else "jax"
    return lambda views, volume: forward_project_planes(
        coefficients[views], volume, grid, detector, backend=backend
    )


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
    values mean the object appears displaced along +u/+v in that view, which is
    the detector-frame translation of :func:`apply_pose_updates`. Shifts up to
    ``max_shift_fraction`` of the detector size are searched. The part of the
    shifts produced by one rigid object translation is removed, because the
    reconstruction absorbs it.
    """
    from tomojax.recon import FBPConfig, fbp

    data = jnp.asarray(projections, dtype=jnp.float32)
    n = int(data.shape[0])
    poses = stack_view_poses(geometry, n)
    rotations = np.asarray(poses, np.float64)[:, :3, :3]
    reproject = _reprojector(geometry, poses, grid, detector)

    max_u = max(1, int(max_shift_fraction * detector.nu))
    max_v = max(1, int(max_shift_fraction * detector.nv))
    chunks = [slice(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)]

    def correlate(volume: jax.Array) -> np.ndarray:
        # Reproject and correlate a chunk of views at a time.
        return np.concatenate(
            [
                np.asarray(
                    _correlation_peaks(
                        data[c],
                        reproject(c, volume),
                        max_u=max_u,
                        max_v=max_v,
                    ),
                    np.float64,
                )
                for c in chunks
            ]
        )

    spacing = (float(detector.du), float(detector.dv))
    shifts = np.zeros((n, 2))
    for iteration in range(int(iters)):
        corrected = jnp.concatenate(
            [_shift_views(data[c], jnp.asarray(-shifts[c], jnp.float32)) for c in chunks]
        )
        volume = fbp(geometry, grid, detector, corrected, config=FBPConfig(filter_name="hann"))
        del corrected
        peaks = correlate(volume)
        updated = _remove_object_translation(peaks, rotations, spacing)
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
    """Return ``shift_px -> relative residual`` of FBP reprojection.

    The views are moved back by a common detector-u shift, reconstructed by FBP
    and reprojected; the residual after a best scalar amplitude fit is smallest
    when the shift matches the data.
    """
    from tomojax.recon import FBPConfig, fbp

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
        volume = fbp(geometry, grid, detector, corrected, config=FBPConfig(filter_name="hann"))
        predicted = jnp.concatenate([reproject(c, volume) for c in chunks])
        scale = jnp.vdot(predicted, corrected) / jnp.maximum(jnp.vdot(predicted, predicted), 1e-30)
        return float(jnp.linalg.norm(scale * predicted - corrected)) / max(norm, 1e-30)

    return residual


def translation_params_from_shifts(
    shifts: np.ndarray,
    poses: np.ndarray,
    params: np.ndarray,
    *,
    frame: str,
    active: tuple[bool, bool],
) -> np.ndarray:
    """Set active (dx, dz) so each view's translation produces its detector shift.

    ``shifts`` are physical (u, v). Detector-frame translations equal them; an
    object-frame translation is rotated by the view's pose first, so a view can
    lose one observable direction. Near-null directions are discarded rather
    than turned into enormous offsets. Frozen translations are kept and their
    contribution is subtracted first.
    """
    n = shifts.shape[0]
    if frame == "detector":
        mapping = np.broadcast_to(np.eye(2), (n, 2, 2))
    else:
        mapping = poses[:, [0, 2], :3][:, :, [0, 2]]
    mask = np.asarray(active, bool)
    fixed = np.where(mask, 0.0, params[:, 3:5])
    residual = shifts - np.einsum("nij,nj->ni", mapping, fixed)
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
    from tomojax.align._config import _active_dof_mask_for_cfg

    active = _active_dof_mask_for_cfg(cfg)
    if not cfg.seed_translations or not any(active[3:5]):
        return None
    n = int(np.shape(projections)[0])
    shifts = estimate_view_shifts(geometry, grid, detector, projections)
    beam = beam_of(geometry)
    # A cone beam magnifies object shifts near the axis onto the detector.
    translations = shifts if beam is None else shifts / beam.magnification
    params = translation_params_from_shifts(
        translations,
        np.asarray(stack_view_poses(geometry, n), np.float64),
        np.zeros((n, len(active)), np.float32),
        frame=cfg.pose_translation_frame,
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
    physical lengths, with the sign of ``Detector.det_center``.
    """
    shift = np.asarray(aligned, np.float64)[:, 0, 3] - np.asarray(nominal, np.float64)[:, 0, 3]
    design = np.column_stack([-np.ones(len(shift)), np.asarray(nominal, np.float64)[:, 0, :3]])
    solution, *_ = np.linalg.lstsq(design, shift, rcond=None)
    residual = shift - design @ solution
    return float(solution[0]), float(np.sqrt(np.mean(residual**2)))


def fold_detector_offset(nominal: np.ndarray, params5: np.ndarray) -> tuple[float, np.ndarray]:
    """Move the detector-u offset implied by detector-frame poses into the detector.

    Detector-frame ``dx`` is a lab-x shift of the object, which moves its image
    along detector u exactly as a detector-centre offset of opposite sign does.
    Returns the offset to add to ``Detector.det_center[0]`` and the poses with
    that constant added to every ``dx``; together they predict the same data.
    """
    params = np.array(params5, dtype=np.float64, copy=True)
    aligned = np.array(nominal, dtype=np.float64, copy=True)
    aligned[:, 0, 3] += params[:, 3]
    offset, _ = implied_detector_offset(nominal, aligned)
    params[:, 3] += offset
    return offset, params.astype(np.asarray(params5).dtype)
