"""Global per-view detector shift search by projection matching.

Local Gauss-Newton alignment converges only from poses close to the truth. Large
per-view stage shifts are instead found by cross-correlation, which searches
every shift up to a bound: reconstruct with the current shifts removed,
reproject, and move each measured view to its correlation peak against the
reprojection. Repeating sharpens the reconstruction and the estimates.
"""

from __future__ import annotations

from functools import partial
import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.views import stack_view_poses

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Geometry, Grid

LOG = logging.getLogger(__name__)


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
    from tomojax.core.joseph import forward_project_planes, plane_coefficients
    from tomojax.recon import FBPConfig, fbp

    data = jnp.asarray(projections, dtype=jnp.float32)
    n = int(data.shape[0])
    poses = stack_view_poses(geometry, n)
    rotations = np.asarray(poses, np.float64)[:, :3, :3]
    coefficients = plane_coefficients(poses, grid, detector)
    backend = "pallas" if jax.default_backend() == "gpu" else "jax"

    def forward(volume: jax.Array) -> jax.Array:
        return jnp.concatenate(
            [
                forward_project_planes(
                    coefficients[s : s + 32], volume, grid, detector, backend=backend
                )
                for s in range(0, n, 32)
            ]
        )

    max_u = max(1, int(max_shift_fraction * detector.nu))
    max_v = max(1, int(max_shift_fraction * detector.nv))
    spacing = (float(detector.du), float(detector.dv))
    shifts = np.zeros((n, 2))
    for iteration in range(int(iters)):
        corrected = _shift_views(data, jnp.asarray(-shifts, jnp.float32))
        volume = fbp(geometry, grid, detector, corrected, config=FBPConfig(filter_name="hann"))
        peaks = np.asarray(
            _correlation_peaks(data, forward(volume), max_u=max_u, max_v=max_v), np.float64
        )
        updated = _remove_object_translation(peaks, rotations, spacing)
        change = float(np.max(np.abs(updated - shifts)))
        shifts = updated
        LOG.info("Shift search %d: max change %.3f px", iteration + 1, change)
        if change <= tolerance_px:
            break
    return shifts * np.asarray(spacing)


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
    fixed = np.where(mask, 0.0, params[:, 3:])
    residual = shifts - np.einsum("nij,nj->ni", mapping, fixed)
    inverse = np.linalg.pinv(mapping * mask[None, None, :], rtol=1e-3)
    estimate = np.einsum("nij,nj->ni", inverse, residual)
    seeded = np.array(params, dtype=np.float32, copy=True)
    seeded[:, 3:] = np.where(mask, estimate, fixed)
    return seeded
