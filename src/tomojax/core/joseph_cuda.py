"""Native CUDA execution of the linear Joseph matched pair, without JAX imports.

A plan owns affine geometry on the current CuPy device. Its input and output
arrays remain on that device and use CuPy's current stream; callers own their
lifetimes. Transpose scratch space is bounded by the chosen view batch size.
"""

from typing import Any

import numpy as np

from ._joseph_cuda_kernels import RUN, THREADS, forward_kernel, gather_kernel
from ._plane_coefficients import coefficients
from .geometry.base import Detector, Grid


class JosephCUDA:
    """Matched physical plane interpolation; batches bound transpose scratch space.

    Volumes are (nz, ny, nx), or solver-private (nx, ny, nz) with ``z_fast``.
    """

    def __init__(
        self,
        poses: np.ndarray,
        grid: Grid,
        detector: Detector,
        *,
        views_per_batch: int,
        z_fast: bool = False,
    ) -> None:
        import cupy as cp

        self.cp = cp
        self.shape = (grid.nx, grid.ny, grid.nz) if z_fast else (grid.nz, grid.ny, grid.nx)
        self.image_shape = (len(poses), detector.nv, detector.nu)
        self.batch = min(int(views_per_batch), len(poses))
        self.coeff = cp.asarray(coefficients(poses, grid, detector, xp=np))
        dims = (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv)
        self.args = tuple(np.int32(n) for n in dims)
        narrow = grid.nx * grid.ny * grid.nz < 2**31
        self.forward_kernel = forward_kernel(narrow=narrow, z_fast=z_fast)
        self.gather_kernel = gather_kernel(z_fast=z_fast)
        # Forward rows per thread tile, as compiled: they follow contiguous z when z-fast.
        self.tile_v = 16 if z_fast else 1

    def forward(self, volume: Any) -> Any:
        """Project a device volume; return (view, v, u) CuPy storage."""
        cp = self.cp
        volume = cp.ascontiguousarray(volume, dtype=cp.float32)
        n, nv, nu = self.image_shape
        out = cp.empty(self.image_shape, cp.float32)
        tiles = -(-nu // (128 // self.tile_v)) * -(-nv // self.tile_v)
        # Interleave nearby views for volume-cache reuse; a flat grid avoids
        # CUDA's grid.y limit for large detectors.
        for first in range(0, n, 64):
            stop = min(first + 64, n)
            self.forward_kernel(
                (tiles * (stop - first),),
                (THREADS,),
                (
                    self.coeff[first:stop],
                    volume,
                    out[first:stop],
                    *self.args,
                    np.int32(stop - first),
                    np.int32(0),
                    self.args[2],
                ),
            )
        return out

    def adjoint(self, images: Any) -> Any:
        """Apply the exact transpose, using bounded per-batch detector scratch."""
        cp = self.cp
        out = cp.zeros(self.shape, cp.float32)
        nx, ny, nz = (int(n) for n in self.args[:3])
        runs = nx * ny * -(-nz // RUN)
        for first in range(0, self.image_shape[0], self.batch):
            stop = min(first + self.batch, self.image_shape[0])
            uv = cp.ascontiguousarray(images[first:stop].transpose(0, 2, 1), dtype=cp.float32)
            self.gather_kernel(
                (-(-runs // THREADS),),
                (THREADS,),
                (self.coeff[first:stop], uv, out, np.int32(stop - first), *self.args, np.int32(0)),
            )
        return out


def slab_forward(volume: Any, coeff: Any, *, grid: Grid, detector: Detector, z0: int) -> Any:
    """Project locally stored z planes using unchanged global-coordinate weights."""
    import cupy as cp

    n, nv, nu = len(coeff), detector.nv, detector.nu
    depth = len(volume)
    # As in the resident solver, nearby detector rows follow contiguous z taps.
    volume = cp.ascontiguousarray(volume.transpose(2, 1, 0))
    out = cp.empty((n, nv, nu), cp.float32)
    kernel = forward_kernel(narrow=volume.size < 2**31, slab=True, z_fast=True)
    tiles = -(-nu // 8) * -(-nv // 16)
    kernel(
        (tiles * n,),
        (THREADS,),
        (
            coeff,
            volume,
            out,
            *map(np.int32, (grid.nx, grid.ny, grid.nz, nu, nv, n, z0, depth)),
        ),
    )
    return out


def slab_adjoint_into(
    images: Any, coeff: Any, *, out: Any, grid: Grid, detector: Detector, z0: int
) -> None:
    """Add the matched transpose to a local z slab of ``out``."""
    import cupy as cp

    depth = len(out)
    runs = grid.nx * grid.ny * -(-depth // RUN)
    uv = cp.ascontiguousarray(images.transpose(0, 2, 1))
    gather_kernel()(
        (-(-runs // THREADS),),
        (THREADS,),
        (
            coeff,
            uv,
            out,
            *map(np.int32, (len(coeff), grid.nx, grid.ny, depth, detector.nu, detector.nv, z0)),
        ),
    )
