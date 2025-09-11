from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from ..core.geometry import Grid, Detector, Geometry
from ..core.projector import forward_project_view, forward_project_view_T
from ..core.operators import view_loss
from .filters import get_filter
from ..utils.logging import progress_iter


def _fft_filter_sinogram(sino: jnp.ndarray, du: float, filter_name: str) -> jnp.ndarray:
    """Apply 1D frequency-domain filter along detector-u axis per (view, v-row).

    sino: (n_views, nv, nu)
    """
    n_views, nv, nu = sino.shape
    H = get_filter(filter_name, nu, du)
    Hc = jnp.asarray(H, dtype=sino.dtype)
    # FFT along last axis (u)
    F = jnp.fft.rfft(sino, axis=-1)
    # Align H for rfft length
    H_r = Hc[: F.shape[-1]]
    Ff = F * H_r
    out = jnp.fft.irfft(Ff, n=nu, axis=-1)
    return out


def backproject_vjp_T(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    filtered: jnp.ndarray,
) -> jnp.ndarray:
    """Backprojection via autodiff VJP of pose-aware forward projector (single view)."""
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    zero_vol = jnp.zeros((nx, ny, nz), dtype=jnp.float32)

    def fwd(vol):
        return forward_project_view_T(T, grid, detector, vol, use_checkpoint=True)

    _, vjp = jax.vjp(fwd, zero_vol)
    return vjp(filtered.astype(jnp.float32))[0]


def fbp(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    filter_name: str = "ramp",
    scale: float | None = None,
) -> jnp.ndarray:
    """Filtered backprojection for parallel-ray geometry using VJP backproject.

    projections: (n_views, nv, nu)
    Returns: (nx, ny, nz)
    """
    proj = jnp.asarray(projections, dtype=jnp.float32)
    n_views, nv, nu = proj.shape
    # Filter along u
    filt = _fft_filter_sinogram(proj, du=float(detector.du), filter_name=filter_name)

    acc = jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32)
    for i in progress_iter(range(n_views), total=n_views, desc="FBP: backproject views"):
        T_i = jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32)
        bp = backproject_vjp_T(T_i, grid, detector, filt[i])
        acc = acc + bp

    if scale is None:
        # Heuristic scaling: divide by number of views
        acc = acc / float(n_views)
    else:
        acc = acc * float(scale)
    return acc
