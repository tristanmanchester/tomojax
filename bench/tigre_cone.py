"""Physical cone-vector geometry for TIGRE, including tilted axes and per-view shifts.

TIGRE rotates its +x source and detector y/z axes by intrinsic ZYZ angles,
then subtracts ``offOrigin`` in the fixed object frame. Choosing that rotation
from the detector axes makes arbitrary rigid arrangements representable without
estimating an orbit or resampling measurements. Volume arrays remain (z, y, x),
projections (view, v, u), as in the existing centred cone comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from tomojax.geometry import Detector, Grid


def geometry(vectors: np.ndarray, *, grid: Grid, detector: Detector) -> tuple[Any, np.ndarray]:
    """TIGRE geometry and angles for ASTRA-order source, centre, pixel-u, pixel-v rows."""
    import tigre

    from tomojax.geometry import grid_volume_origin

    vectors = np.asarray(vectors, np.float64)
    source, centre, u, v = (vectors[:, 3 * k : 3 * k + 3] for k in range(4))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    rotation = np.stack([np.cross(u, v), u, v], axis=2)
    source_local = np.einsum("nji,nj->ni", rotation, source)
    centre_local = np.einsum("nji,nj->ni", rotation, centre)
    offset = source_local.copy()
    offset[:, 0] = 0
    centre_local -= offset
    geo = tigre.geometry(mode="cone", default=False)
    geo.DSO = source_local[:, 0]
    geo.DSD = geo.DSO - centre_local[:, 0]
    geo.nDetector = np.array([detector.nv, detector.nu])
    geo.dDetector = np.array([detector.dv, detector.du])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array([grid.nz, grid.ny, grid.nx])
    geo.dVoxel = np.array([grid.vz, grid.vy, grid.vx])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    # Offsets are translations, not rotations of the reconstruction array.
    centre = np.array(grid_volume_origin(grid)) + (geo.nVoxel[::-1] - 1) * geo.dVoxel[::-1] / 2
    geo.offOrigin = (centre - np.einsum("nij,nj->ni", rotation, offset))[:, ::-1]
    geo.offDetector = centre_local[:, [2, 1]]
    geo.rotDetector = np.zeros((len(source), 3))
    geo.accuracy = 0.5
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        # At a ZYZ pole the decomposition is nonunique, but its rotation is exact.
        angles = Rotation.from_matrix(rotation).as_euler("ZYZ")
    return geo, angles.astype(np.float32)
