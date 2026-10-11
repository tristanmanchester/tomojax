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
    change the completed-slab failure contract. Integer gathers and exceptional
    near-overflow arithmetic retain the existing host sampling path.
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
    minimum = np.min(samples, initial=np.float32(0))
    maximum = np.max(samples, initial=np.float32(0))
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError("fourier_reconstruct: sampled projection rows must be finite in FP32")
    if max(-minimum, maximum) > np.finfo(np.float32).max / 2:
        return sample_projection_rows(projections, order, positions)
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
    """Materialize an owned FFT slab; return metadata owners for async CUDA use.

    The NumPy path also permits an existing portable plan to consume staged
    rows, for example when independently testing allocation-retry orchestration.
    """
    nviews, nu = rows.samples.shape[1:]
    if xp is np:
        output = np.zeros((depth, nviews, nu), dtype=np.float32)
        for indices, weight in ((rows.lower, 1 - rows.fraction), (rows.upper, rows.fraction)):
            valid = indices >= 0
            if np.any(valid):
                output[np.flatnonzero(valid)] += source[indices[valid]] * weight[valid, None, None]
        return output, ()
    output = xp.empty((depth, nviews, nu), dtype=xp.float32)
    lower, upper, fraction = (
        xp.asarray(rows.lower),
        xp.asarray(rows.upper),
        xp.asarray(rows.fraction),
    )
    _row_kernel()(
        source,
        lower,
        upper,
        fraction,
        np.int64(nviews),
        np.int64(nu),
        np.int64(len(rows.fraction)),
        output,
    )
    return output, (lower, upper, fraction)
