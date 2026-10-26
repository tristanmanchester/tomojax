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
    """Matched physical plane interpolation; batches bound transpose scratch space."""

    def __init__(
        self, poses: np.ndarray, grid: Grid, detector: Detector, *, views_per_batch: int
    ) -> None:
        import cupy as cp

        self.cp = cp
        self.shape = (grid.nz, grid.ny, grid.nx)
        self.image_shape = (len(poses), detector.nv, detector.nu)
        self.batch = min(int(views_per_batch), len(poses))
        self.coeff = cp.asarray(coefficients(poses, grid, detector, xp=np))
        self.args = tuple(np.int32(n) for n in (*self.shape[::-1], detector.nu, detector.nv))
        self.forward_kernel = forward_kernel(narrow=grid.nx * grid.ny * grid.nz < 2**31)
        self.gather_kernel = gather_kernel()

    def forward(self, volume: Any) -> Any:
        """Project a device volume; return (view, v, u) CuPy storage."""
        cp = self.cp
        volume = cp.ascontiguousarray(volume, dtype=cp.float32)
        n, nv, nu = self.image_shape
        out = cp.empty(self.image_shape, cp.float32)
        tiles = -(-nu // 128) * nv
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
                ),
            )
        return out

    def adjoint(self, images: Any) -> Any:
        """Apply the exact transpose, using bounded per-batch detector scratch."""
        cp = self.cp
        out = cp.zeros(self.shape, cp.float32)
        nz, ny, nx = self.shape
        runs = nx * ny * -(-nz // RUN)
        for first in range(0, self.image_shape[0], self.batch):
            stop = min(first + self.batch, self.image_shape[0])
            uv = cp.ascontiguousarray(images[first:stop].transpose(0, 2, 1), dtype=cp.float32)
            self.gather_kernel(
                (-(-runs // THREADS),),
                (THREADS,),
                (self.coeff[first:stop], uv, out, np.int32(stop - first), *self.args),
            )
        return out
