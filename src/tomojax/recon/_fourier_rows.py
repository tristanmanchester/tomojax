"""Bounded source-row staging and exact FP32 detector-v interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ._fourier_grid import sample_projection_rows


@dataclass
class ProjectionRows:
    """Owned source samples, ordered by detector row then view then u."""

    samples: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    fraction: np.ndarray


def prepare_projection_rows(projections, order, positions) -> np.ndarray | ProjectionRows:
    """Stage at most two measured rows per requested slice, without interpolation.

    Only contributing neighbors are copied, even for strided/mapped input. A
    sampled invalid value fails here, on the reader, before later GPU work can
    change the completed-slab failure contract. Integer rows need no
    interpolation and keep the host sampling path.
    """
    nv = projections.shape[1]
    positions = np.clip(positions, -1, nv)
    indices = np.floor(positions).astype(np.int64)
    fraction = np.asarray(positions - indices, dtype=np.float32)
    if np.all(fraction == 0) and np.all((indices >= 0) & (indices < nv)):
        return sample_projection_rows(projections, order, positions)
    valid_lower = (indices >= 0) & (indices < nv) & (fraction != 1)
    valid_upper = (indices + 1 >= 0) & (indices + 1 < nv) & (fraction != 0)
    unique = np.unique(np.concatenate((indices[valid_lower], (indices + 1)[valid_upper])))
    lower = np.where(valid_lower, np.searchsorted(unique, indices), -1)
    upper = np.where(valid_upper, np.searchsorted(unique, indices + 1), -1)
    samples = np.asarray(projections[order[None, :], unique[:, None], :], dtype=np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("fourier_reconstruct: sampled projection rows must be finite in FP32")
    return ProjectionRows(samples, lower, upper, fraction)


@lru_cache(maxsize=1)
def _row_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "raw float32 source, raw int64 lower, raw int64 upper, "
        "raw float32 fraction, int64 nviews, int64 nu, int64 count",
        "float32 output",
        r"""
        long long u = i % nu;
        long long view = (i / nu) % nviews;
        long long z = i / (nu * nviews);
        if (z >= count) { output = 0.0f; return; }
        long long lo = lower[z], hi = upper[z];
        float f = fraction[z], value = 0.0f;
        // Match host multiply/add rounding, zero-weight masking and signed zeros.
        if (lo >= 0)
            value = __fadd_rn(value, __fmul_rn(
                source[(lo * nviews + view) * nu + u], __fsub_rn(1.0f, f)));
        if (hi >= 0)
            value = __fadd_rn(value, __fmul_rn(source[(hi * nviews + view) * nu + u], f));
        output = value;
        """,
        "tomojax_fourier_detector_rows",
    )


def interpolate_projection_rows(rows: ProjectionRows, source, depth: int, xp):
    """An owned CUDA FFT slab, and the metadata arrays it needs until the kernel runs."""
    nviews, nu = rows.samples.shape[1:]
    output = xp.empty((depth, nviews, nu), dtype=xp.float32)
    lower, upper, fraction = (xp.asarray(a) for a in (rows.lower, rows.upper, rows.fraction))
    count = np.int64(len(rows.fraction))
    _row_kernel()(source, lower, upper, fraction, np.int64(nviews), np.int64(nu), count, output)
    return output, (lower, upper, fraction)
