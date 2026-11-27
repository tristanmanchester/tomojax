"""Rows for FDK: continued past the detector's edges, ramp-filtered, and how far they reach."""

from __future__ import annotations

from functools import partial
import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry.base import grid_volume_origin

from ._continuation import Continuation, continued_columns

if TYPE_CHECKING:
    import jax

    from tomojax.core.geometry import ConeBeam, Detector, Grid

    from ._continuation import Host


def grid_columns(
    parts: tuple[tuple[int | None, ConeBeam, Detector], ...], grid: Grid, poses: np.ndarray
) -> tuple[int, int]:
    """Columns below and above the detector that ``grid``'s voxels project onto in some view.

    Filtered rows are nonzero beyond the measured detector: a voxel outside the
    field of view needs them where it projects, as it does in parallel-beam
    FBP. At most three detector widths are kept on a side, which a grid that
    nears the source plane would otherwise exceed without bound.
    """
    origin: Host = np.asarray(grid_volume_origin(grid), np.float64)
    spacing: Host = np.asarray([grid.vx, grid.vy, grid.vz], np.float64)
    count: Host = np.asarray([grid.nx, grid.ny, grid.nz], np.float64)
    faces = np.stack([origin - spacing / 2, origin + (count - 0.5) * spacing])
    # A box projects inside the hull of its eight projected corners.
    corners = faces[np.indices((2, 2, 2)).reshape(3, 8).T, np.arange(3)]
    world: Host = np.einsum("nij,kj->nki", poses[:, :3, :3], corners) + poses[:, None, :3, 3]
    low, high, start, nu = math.inf, -math.inf, 0, parts[0][2].nu
    for views, beam, own in parts:
        stop = len(poses) if views is None else start + views
        center, u_dir, v_dir = beam.detector_frame(own)
        normal = np.cross(u_dir, v_dir)
        source = beam.source()
        ray: Host = world[start:stop] - source
        hit: Host = source + ray * (((center - source) @ normal) / (ray @ normal))[..., None]
        columns: Host = ((hit - center) @ u_dir) / own.du + (own.nu - 1) / 2
        low, high, start = min(low, float(np.min(columns))), max(high, float(np.max(columns))), stop
    limit = 3 * nu
    below = min(limit, max(0, math.ceil(-low - 1e-6)))
    above = min(limit, max(0, math.ceil(high - (nu - 1) - 1e-6)))
    return below, above


def filter_operator(
    spectrum: np.ndarray, nu: int, pad_lo: int, pad_hi: int, pad: int
) -> np.ndarray:
    """The matrix that ramp-filters continued rows as ``spectrum`` does.

    Its ``nu + 2 * continued_columns(nu)`` rows are the continued row's columns
    and its ``pad_lo + nu + pad_hi`` columns the filtered ones the
    backprojection reads. Filtering is a circular convolution with the
    zero-padded row; as a matrix product it runs on the GPU's matrix units
    several times faster than the FFTs for rows up to a couple of thousand
    pixels.
    """
    n_fft = 2 * (spectrum.shape[0] - 1)
    impulse = np.fft.irfft(spectrum.astype(np.float64), n=n_fft)
    columns = np.arange(-pad_lo, nu + pad_hi)
    return impulse[(columns[None, :] - np.arange(-pad, nu + pad)[:, None]) % n_fft].astype(
        np.float32
    )


def filter_rows(rows: jax.Array, kernel: jax.Array, nu: int, pad_lo: int, pad_hi: int) -> jax.Array:
    """Ramp-filter continued rows, keeping ``pad_lo``/``pad_hi`` columns beyond the detector.

    Filtered rows extend past the data; an offset detector's backprojection
    needs that tail on its short side.
    """
    n_fft = 2 * (int(kernel.shape[0]) - 1)
    out = jnp.fft.irfft(jnp.fft.rfft(rows, n=n_fft, axis=-1) * kernel, n=n_fft, axis=-1)
    first = (rows.shape[-1] - nu) // 2 - pad_lo
    if first >= 0:
        return out[..., first : first + pad_lo + nu + pad_hi]
    # Virtual columns beyond the continuation read the circular transform's tail.
    return jnp.concatenate(
        [out[..., n_fft + first :], out[..., : first + pad_lo + nu + pad_hi]], -1
    )


def continued_rows(
    views: jax.Array,
    weights: jax.Array,
    cosine: jax.Array,
    radius2: jax.Array | None,
    distance: tuple[float, ...],
) -> jax.Array:
    """Weighted rows continued past the detector's edges (see ``_continuation``).

    Angular (Parker, offset-detector) weights vary along the row, so they apply
    before the ramp filter, and before the continuation: an edge whose rays an
    opposing view measures has its weight fall to zero there and continues as
    zero, and any other edge continues with the weight it ends on. The cosine
    weights cover the continued columns.
    """
    nu = views.shape[-1]
    pad = continued_columns(nu)
    rows = views * weights[:, None, :]
    if radius2 is None:
        return rows * cosine
    fits = Continuation(np.asarray(distance), nu, pad)
    before, after = fits.continued(partial(jnp.asarray, dtype=rows.dtype), rows, radius2)
    return jnp.concatenate([before, rows, after], axis=-1) * cosine
