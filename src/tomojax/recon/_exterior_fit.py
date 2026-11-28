"""Fit a coarse volume to the physical measured and continued ray intervals."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from tomojax._typed_arrays import Float32Array, Float64Array, max_float, numpy_float32_array
from tomojax.core import checkpoint
from tomojax.core.geometry.cone import cone_parts
from tomojax.core.geometry.views import host_view_poses
from tomojax.core.host_volume import host_array
from tomojax.geometry import Detector, Geometry, Grid, ParallelGeometry

from .cgls import CGLSConfig, cgls

if TYPE_CHECKING:
    from numpy.typing import NDArray


def view_poses(geometry: Geometry, views: int) -> Float32Array:
    poses = host_view_poses(geometry, views)
    if poses is None:
        from tomojax.core.geometry.views import stack_view_poses

        poses = np.asarray(stack_view_poses(geometry, views))
    return numpy_float32_array(poses)


def view_stride(geometry: Geometry, grid: Grid) -> int:
    """Largest angular stride moving a corner by at most one auxiliary voxel between views."""
    if type(geometry) is not ParallelGeometry or len(geometry.angles) < 2:
        return 1
    gap = max_float(np.abs(np.diff(np.unwrap(np.deg2rad(geometry.angles)))))
    distance = 0.5 * math.hypot(grid.nx * grid.vx, grid.ny * grid.vy)
    return max(1, math.floor(min(grid.vx, grid.vy) / (distance * gap))) if gap else 1


class _Runtime(NamedTuple):
    native: bool


def _fit(
    geometry: Geometry,
    wide: Grid,
    detector: Detector,
    full: np.ndarray,
    config: CGLSConfig,
    *,
    native: bool,
    axial: bool,
) -> tuple[Float32Array, bool]:
    """Resume the coarse fit independently and retain its finished coefficients for the fine fit."""
    from tomojax.geometry.detector_grid import detector_grid_from_geometry_inputs

    with checkpoint.scope("continued-exterior"), checkpoint.loop("volume") as fitted:
        if fitted.saved is not None:
            return fitted.saved.arrays[0], native and fitted.saved.scalars(_Runtime).native
        stride = view_stride(geometry, wide)
        if isinstance(geometry, ParallelGeometry) and stride > 1:
            geometry = replace(geometry, angles=np.asarray(geometry.angles)[::stride])
            full = np.ascontiguousarray(full[::stride])
        if axial:
            from ._exterior_cuda import axial_fit

            volume = axial_fit(view_poses(geometry, len(full)), wide, detector, full, config)
        else:
            reconstruction, info = cgls(
                geometry,
                wide,
                detector,
                full,
                config=config,
                det_grid=detector_grid_from_geometry_inputs(detector, geometry),
            )
            volume = numpy_float32_array(reconstruction)
            if native and cone_parts(geometry, detector) is not None:
                from tomojax.core.cone import use_cuda_cone

                # Reproject the reference cone iterate with the same native kernel.
                # Its CGLS reductions determine the fitted low-frequency background.
                native = use_cuda_cone()
            else:
                native = info.get("execution_backend") == "cupy"
        fitted.save((volume,), _Runtime(native))
        return volume, native


def fitted_columns(first: int, stop: int, nu: int, edge: int | None) -> slice:
    """Measured and continued columns that enter the physical least-squares fit."""
    return slice(first if edge == 1 else 0, stop if edge == 0 else nu)


def _crop(detector: Detector, support: NDArray[np.bool_]) -> tuple[Detector, slice]:
    selected = np.flatnonzero(support)
    first, stop = selected.item(0), selected.item(-1) + 1
    center = detector.center[0] + (first + stop - detector.nu) * detector.du / 2
    return replace(detector, nu=stop - first, center=(center, detector.center[1])), slice(
        first, stop
    )


@dataclass(frozen=True)
class CompletedRows:
    data: Float32Array
    unseen: NDArray[np.bool_]
    support: NDArray[np.bool_] | None

    def mass(self) -> Float64Array:
        """Positive attenuation on virtual rays that no voxel of the requested grid can reach."""
        observed = np.sum(self.data, axis=-1, where=self.unseen, dtype=np.float64)
        views = len(observed)
        uncertainty: Float64Array | float = (
            np.sqrt(np.mean(np.diff(observed, axis=0) ** 2, axis=0) / (2 * views))
            if views > 1
            else 0.0
        )
        return np.maximum(
            np.mean(observed, axis=0, dtype=np.float64)
            - math.sqrt(2 * math.log(2 * views)) * uncertainty,
            0.0,
        )

    def fit(
        self,
        geometry: Geometry,
        wide: Grid,
        detector: Detector,
        config: CGLSConfig,
        *,
        native: bool,
        axial: bool,
    ) -> tuple[Float32Array, bool]:
        if self.support is None:
            return _fit(geometry, wide, detector, self.data, config, native=native, axial=axial)
        if self.support.ndim == 1:
            panel, columns = _crop(detector, self.support)
            return _fit(
                geometry, wide, panel, self.data[..., columns], config, native=native, axial=axial
            )
        # On a rolled axial full turn each physical height has a different
        # measured interval. Each plane is independent under rotation about z.
        plane = replace(wide, nz=1, vol_center=(0.0, 0.0, 0.0))
        panel = replace(detector, nv=1, center=(detector.center[0], 0.0))
        volume = host_array(wide.shape)
        for row, measured in enumerate(self.support):
            if measured.any():
                chord, columns = _crop(panel, measured)
                with checkpoint.scope(f"exterior-chord-{row}"):
                    fitted, _ = _fit(
                        geometry,
                        plane,
                        chord,
                        self.data[:, row : row + 1, columns],
                        config,
                        native=native,
                        axial=native,
                    )
                volume[row] = fitted[0]
        return volume, native
