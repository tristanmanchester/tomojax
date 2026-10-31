"""Device arrays and cone kernels for slab sweeps: CUDA through CuPy, or JAX on NumPy.

An engine moves slab volumes and detector-row images between host and device
and applies the cone projector to a slab and a row window; the coefficients
it receives place both (see :mod:`._cone_slabs`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cache, partial
from typing import TYPE_CHECKING, Any, Protocol

import jax
import numpy as np

from tomojax.core.cone import (
    cone_backproject,
    cone_backproject_pair,
    cone_project,
    cone_project_ones,
)

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid


class Engine(Protocol):
    """Device arrays and the slab kernels of one execution backend."""

    xp: Any
    traffic: dict[str, int]
    grid: Grid
    detector: Detector

    def upload(self, array: np.ndarray) -> Any: ...

    def download(self, array: Any, out: np.ndarray) -> None: ...

    def project(self, volume: Any, coeff: Any, rows: int, *, out: Any, row_start: int) -> None: ...

    def transpose(
        self, images: tuple[Any, ...], coeff: Any, accumulate: tuple[Any, ...]
    ) -> tuple[Any, ...]: ...

    def project_ones(self, coeff: Any, rows: int) -> Any: ...

    def norm2(self, value: Any) -> float: ...


@cache
def _squares() -> Any:
    import cupy as cp

    return cp.ReductionKernel(
        "T x", "float64 y", "(double)x * (double)x", "a + b", "y = a", "0", "tomojax_squares"
    )


@dataclass
class CudaEngine:
    """CuPy arrays and the CUDA cone kernels on the current device and stream."""

    grid: Grid
    detector: Detector
    traffic: dict[str, int] = field(default_factory=lambda: {"to_device": 0, "to_host": 0})

    def __post_init__(self) -> None:
        import cupy as cp

        self.xp = cp
        self._grids: dict[int, Grid] = {}
        self._detectors: dict[int, Detector] = {}

    def _grid(self, nz: int) -> Grid:
        if nz not in self._grids:
            self._grids[nz] = replace(self.grid, nz=nz, vol_origin=None, vol_center=None)
        return self._grids[nz]

    def _detector(self, rows: int) -> Detector:
        if rows not in self._detectors:
            self._detectors[rows] = replace(self.detector, nv=rows)
        return self._detectors[rows]

    def upload(self, array: np.ndarray) -> Any:
        self.traffic["to_device"] += array.nbytes
        return self.xp.asarray(array)

    def download(self, array: Any, out: np.ndarray) -> None:
        self.traffic["to_host"] += out.nbytes
        if out.flags.c_contiguous:
            array.get(out=out)
        else:
            out[...] = array.get()

    def project(self, volume: Any, coeff: Any, rows: int, *, out: Any, row_start: int) -> None:
        """Add a slab's projection directly to circular rows of ``out``."""
        from tomojax.core.cone import forward_into

        forward_into(
            out,
            coeff,
            volume,
            grid=self._grid(volume.shape[0]),
            det=self._detector(rows),
            ring_start=row_start,
        )

    def transpose(
        self, images: tuple[Any, ...], coeff: Any, accumulate: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        from tomojax.core.cone import adjoint_into, adjoint_pair_into

        grid = self._grid(accumulate[0].shape[0])
        detector = self._detector(images[0].shape[1])
        if len(images) == 1:
            adjoint_into(accumulate[0], coeff, images[0], accumulate[0], grid=grid, det=detector)
        else:
            adjoint_pair_into(accumulate, coeff, *images, *accumulate, grid=grid, det=detector)
        return accumulate

    def project_ones(self, coeff: Any, rows: int) -> Any:
        from tomojax.core.cone import box_forward_into

        out = self.xp.empty((coeff.shape[0], rows, self.detector.nu), self.xp.float32)
        box_forward_into(out, coeff, grid=self.grid, det=self._detector(rows))
        return out

    def norm2(self, value: Any) -> float:
        """The FP64 sum of squares, without a squared temporary."""
        return float(_squares()(value))


@partial(jax.jit, static_argnames=("grid", "detector", "backend"))
def _project(volume, coeff, *, grid, detector, backend):
    return cone_project(volume, coeff, grid, detector, backend=backend)


@partial(jax.jit, static_argnames=("grid", "detector", "backend"))
def _project_ones(coeff, *, grid, detector, backend):
    return cone_project_ones(coeff, grid, detector, backend=backend)


@partial(jax.jit, static_argnames=("grid", "detector", "backend"))
def _transpose(images, coeff, accumulate, *, grid, detector, backend):
    return cone_backproject(images, coeff, grid, detector, backend=backend, accumulate=accumulate)


@partial(jax.jit, static_argnames=("grid", "detector", "backend"))
def _transpose_pair(images, coeff, accumulate, *, grid, detector, backend):
    return cone_backproject_pair(
        images, coeff, grid, detector, backend=backend, accumulate=accumulate
    )


@dataclass
class JaxEngine:
    """NumPy arrays with the JAX cone projector, for hosts without the CUDA kernels."""

    grid: Grid
    detector: Detector
    backend: str
    traffic: dict[str, int] = field(default_factory=lambda: {"to_device": 0, "to_host": 0})

    def __post_init__(self) -> None:
        self.xp = np

    def _shapes(self, nz: int, rows: int) -> dict[str, Any]:
        grid = replace(self.grid, nz=nz, vol_origin=None, vol_center=None)
        return {"grid": grid, "detector": replace(self.detector, nv=rows), "backend": self.backend}

    def upload(self, array: np.ndarray) -> np.ndarray:
        self.traffic["to_device"] += array.nbytes
        return np.array(array, np.float32)

    def download(self, array: np.ndarray, out: np.ndarray) -> None:
        self.traffic["to_host"] += out.nbytes
        out[...] = array

    def project(
        self, volume: np.ndarray, coeff: np.ndarray, rows: int, *, out: np.ndarray, row_start: int
    ) -> None:
        """Add a slab's projection to circular rows of the NumPy destination."""
        shapes = self._shapes(volume.shape[0], rows)
        values = np.array(_project(volume, coeff, **shapes))
        first = row_start % out.shape[1]
        length = min(rows, out.shape[1] - first)
        out[:, first : first + length] += values[:, :length]
        if length < rows:
            out[:, : rows - length] += values[:, length:]

    def transpose(
        self, images: tuple[np.ndarray, ...], coeff: np.ndarray, accumulate: tuple[np.ndarray, ...]
    ) -> tuple[np.ndarray, ...]:
        shapes = self._shapes(accumulate[0].shape[0], images[0].shape[1])
        if len(images) == 1:
            results = (_transpose(images[0], coeff, accumulate[0], **shapes),)
        else:
            results = _transpose_pair(tuple(images), coeff, tuple(accumulate), **shapes)
        for target, value in zip(accumulate, results, strict=True):
            target[...] = np.asarray(value)
        return accumulate

    def project_ones(self, coeff: np.ndarray, rows: int) -> np.ndarray:
        shapes = self._shapes(self.grid.nz, rows)
        shapes["grid"] = replace(self.grid, vol_origin=None, vol_center=None)
        return np.array(_project_ones(coeff, **shapes))

    def norm2(self, value: np.ndarray) -> float:
        """The FP64 sum of squares."""
        return float(np.sum(np.square(value, dtype=np.float64)))


def engine_for(grid: Grid, detector: Detector, backend: str) -> CudaEngine | JaxEngine:
    """The CUDA engine for the CUDA backend, otherwise the JAX projector on NumPy arrays."""
    if backend == "cuda":
        return CudaEngine(grid, detector)
    return JaxEngine(grid, detector, backend)
