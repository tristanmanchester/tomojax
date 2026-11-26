"""Native CUDA execution of the linear Joseph matched pair, without JAX imports.

A plan owns affine geometry on the current CuPy device. Its input and output
arrays remain on that device and use CuPy's current stream; callers own their
lifetimes. Transpose scratch space is bounded by the chosen view batch size.
"""

from typing import Any, Literal

import numpy as np

from ._axial_kernels import (
    aligned_axial,
    axial_coefficients,
    axial_forward,
    transpose_into,
)
from ._joseph_cuda_kernels import (
    RUN,
    THREADS,
    forward_kernel,
    gather_kernel,
)
from ._plane_coefficients import coefficients
from .geometry.base import Detector, Grid

type Sampling = Literal["generic", "axial", "aligned"]


def sampling(coeff: np.ndarray) -> Sampling:
    """Identify separability and four-plane alignment from the actual ray maps."""
    if not axial_coefficients(coeff):
        return "generic"
    return "aligned" if aligned_axial(coeff) else "axial"


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
        coeff = coefficients(poses, grid, detector, xp=np)
        self.coeff = cp.asarray(coeff, dtype=cp.float32)
        self.z_fast = z_fast
        self.sampling = sampling(coeff)
        dims = (grid.nx, grid.ny, grid.nz, detector.nu, detector.nv)
        self.dims = dims
        self.args = tuple(np.int32(n) for n in dims)
        narrow = grid.nx * grid.ny * grid.nz < 2**31

        # Forward rows per thread tile, as compiled: they follow contiguous z when z-fast.
        self.tile_v = 16 if z_fast else 1
        if self.sampling == "generic":
            self.forward_kernel = forward_kernel(narrow=narrow, z_fast=z_fast)
            self.gather_kernel = gather_kernel(z_fast=z_fast)

    def forward(self, volume: Any) -> Any:
        """Project a device volume; return (view, v, u) CuPy storage."""
        cp = self.cp
        volume = cp.ascontiguousarray(volume, dtype=cp.float32)
        n, nv, nu = self.image_shape
        if self.sampling != "generic":
            return axial_forward(
                cp,
                self.coeff,
                volume,
                dims=self.dims,
                views=n,
                z_fast=self.z_fast,
                aligned=self.sampling == "aligned",
                z0=0,
                depth=self.dims[2],
            )
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
                    self.args[2],
                    np.int32(0),
                ),
            )
        return out

    def adjoint(self, images: Any) -> Any:
        """Apply the exact transpose, using bounded per-batch detector scratch."""
        cp = self.cp
        out = cp.zeros(self.shape, cp.float32)
        nx, ny, nz = (int(n) for n in self.args[:3])
        if self.sampling != "generic":
            transpose_into(
                cp,
                self.coeff,
                cp.ascontiguousarray(images, dtype=cp.float32),
                out,
                shape=(nx, ny, nz),
                image_shape=self.image_shape,
                cubic=False,
                z_fast=self.z_fast,
                vu=True,
                batch=self.batch,
            )
            return out
        runs = nx * ny * -(-nz // RUN)
        for first in range(0, self.image_shape[0], self.batch):
            stop = min(first + self.batch, self.image_shape[0])
            uv = cp.ascontiguousarray(images[first:stop].transpose(0, 2, 1), dtype=cp.float32)
            self.gather_kernel(
                (-(-runs // THREADS),),
                (THREADS,),
                (
                    self.coeff[first:stop],
                    uv,
                    out,
                    *(self.coeff, self.coeff, np.int32(0)),  # no run list
                    np.int32(stop - first),
                    *self.args,
                    np.int32(0),
                ),
            )
        return out


def slab_forward(
    volume: Any, coeff: Any, *, grid: Grid, detector: Detector, z0: int, mode: Sampling
) -> Any:
    """Project locally stored z planes using unchanged global-coordinate weights."""
    import cupy as cp

    n, nv, nu = len(coeff), detector.nv, detector.nu
    depth = len(volume)
    # As in the resident solver, nearby detector rows follow contiguous z taps.
    volume = cp.ascontiguousarray(volume.transpose(2, 1, 0))
    if mode != "generic":
        return axial_forward(
            cp,
            coeff,
            volume,
            dims=(grid.nx, grid.ny, grid.nz, nu, nv),
            views=n,
            z_fast=True,
            aligned=mode == "aligned",
            z0=z0,
            depth=depth,
        )
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
            *map(np.int32, (grid.nx, grid.ny, grid.nz, nu, nv, n, z0, depth, grid.nz, 0)),
        ),
    )
    return out


def slab_adjoint_into(
    images: Any, coeff: Any, *, out: Any, grid: Grid, detector: Detector, z0: int, axial: bool
) -> None:
    """Add the matched transpose to a local z slab of ``out``."""
    import cupy as cp

    depth = len(out)
    if axial:
        transpose_into(
            cp,
            coeff,
            cp.ascontiguousarray(images),
            out,
            shape=(grid.nx, grid.ny, depth),
            image_shape=(len(coeff), detector.nv, detector.nu),
            z0=z0,
            cubic=False,
            z_fast=False,
            vu=True,
        )
        return
    runs = grid.nx * grid.ny * -(-depth // RUN)
    uv = cp.ascontiguousarray(images.transpose(0, 2, 1))
    gather_kernel()(
        (-(-runs // THREADS),),
        (THREADS,),
        (
            coeff,
            uv,
            out,
            *(coeff, coeff, np.int32(0)),  # no run list
            *map(np.int32, (len(coeff), grid.nx, grid.ny, depth, detector.nu, detector.nv, z0)),
        ),
    )
