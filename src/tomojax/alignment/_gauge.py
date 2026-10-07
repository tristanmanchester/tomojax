"""The alignment gauge: changes to an estimate that leave every projection unchanged.

Moving the object by a rigid motion ``G`` and every view's pose by its inverse
(``P_i -> P_i G``) predicts the same data in any geometry. With detector-frame
translations in a parallel beam, adding ``c`` to every view's ``dx`` and to the
detector centre does too. The projections cannot choose among these estimates,
so alignment reports the one with the least per-view motion: no common
rotation, and translations with no rigid-shift or constant detector-u part.

:func:`least_motion_gauge` finds that representative's gauge, :func:`apply_to_poses`
and :func:`apply_to_volume` move an estimate to it. All three take ``nominal``,
the ``(views, 4, 4)`` world-from-object poses the per-view parameters correct,
and pose tables with the columns of :data:`DOF_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from tomojax.core.geometry.base import grid_volume_origin
from tomojax.core.geometry.transforms import pose_angles, pose_rotations

from ._model.dofs import DOF_INDEX, DOF_NAMES

if TYPE_CHECKING:
    from collections.abc import Collection

    from tomojax.core.geometry import Grid

    from ._geometry.parametrizations import PoseTranslationFrame

_ROTATIONS = ("alpha", "beta", "phi")
_TRANSLATIONS = ("dx", "dy", "dz")  # lab or object x, y, z


@dataclass(frozen=True)
class Gauge:
    """A rigid object motion ``[rotation | shift]`` and a detector-u offset.

    Applying it maps each pose ``P_i`` to ``P_i @ [rotation | shift]`` and adds
    ``detector_offset`` (a physical length) to the detector centre and to every
    detector-frame ``dx``.
    """

    rotation: np.ndarray
    shift: np.ndarray
    detector_offset: float = 0.0

    @property
    def rotation_deg(self) -> list[float]:
        """The rotation as a rotation vector, in degrees."""
        return [float(v) for v in Rotation.from_matrix(self.rotation).as_rotvec(degrees=True)]

    def to_dict(self) -> dict[str, object]:
        """The gauge as JSON: rotation vector (degrees), shift and detector offset."""
        return {
            "rotation_deg": self.rotation_deg,
            "shift": [float(v) for v in self.shift],
            "detector_offset": float(self.detector_offset),
        }


def _translation_basis(
    params: np.ndarray, nominal: np.ndarray, frame: PoseTranslationFrame
) -> np.ndarray:
    """How an object shift ``q`` moves each view's translation parameters: ``A_i q``."""
    rotations = pose_rotations(params)
    if frame == "detector":
        return np.asarray(nominal, np.float64)[:, :3, :3] @ rotations
    return rotations


def least_motion_gauge(
    params: np.ndarray,
    nominal: np.ndarray,
    *,
    translation_frame: PoseTranslationFrame,
    active: Collection[str] = DOF_NAMES,
    invisible: Collection[str] = (),
    detector_offset: bool = False,
) -> Gauge:
    """The gauge taking ``params`` to the least-motion estimate.

    Only ``active`` parameters may change; ``invisible`` ones (along-beam ``dy``
    in a parallel beam) neither count nor constrain. ``detector_offset`` also
    moves a constant detector-frame ``dx`` into the detector centre.
    """
    params = np.asarray(params, np.float64)
    rotations = pose_rotations(params)
    # The rotation Q minimising sum_i |R_i Q - I|^2 inverts their chordal mean.
    u, _, vt = np.linalg.svd(rotations.sum(axis=0))
    rotation = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    if not {"alpha", "beta"} <= set(active):
        rotvec[:2] = 0.0
    if "phi" not in active:
        rotvec[2] = 0.0
    rotation = Rotation.from_rotvec(rotvec).as_matrix()

    # Shift q (and offset c) minimising the remaining translations. A shift
    # direction is unavailable if it would move a fixed visible translation;
    # that is judged on the nominal geometry, as the views' small rotations
    # couple every direction weakly to every translation.
    basis = _translation_basis(params, nominal, translation_frame)
    structure = _translation_basis(np.zeros_like(params), nominal, translation_frame)
    offset_column = detector_offset and translation_frame == "detector"
    rows, targets, fixed_rows = [], [], []
    for axis, name in enumerate(_TRANSLATIONS):
        if name in invisible:
            continue
        block, nominal_block = basis[:, axis, :], structure[:, axis, :]
        if offset_column:
            column = np.full((len(block), 1), 1.0 if name == "dx" else 0.0)
            block = np.concatenate([block, column], axis=1)
            nominal_block = np.concatenate([nominal_block, column], axis=1)
        if name in active:
            rows.append(block)
            targets.append(-params[:, DOF_INDEX[name]])
        else:
            fixed_rows.append(nominal_block)
    unknowns = 4 if offset_column else 3
    if not rows:
        return Gauge(rotation, np.zeros(3), 0.0)
    design, target = np.concatenate(rows), np.concatenate(targets)
    free = np.eye(unknowns)
    if fixed_rows:
        constraint = np.concatenate(fixed_rows)
        _, singular, vt_c = np.linalg.svd(constraint)
        rank = int(np.sum(singular > 1e-6 * max(1.0, float(singular[0]))))
        free = vt_c[rank:].T
    if free.shape[1] == 0:
        return Gauge(rotation, np.zeros(3), 0.0)
    solution = free @ np.linalg.lstsq(design @ free, target, rcond=None)[0]
    offset = float(solution[3]) if offset_column else 0.0
    return Gauge(rotation, solution[:3], offset)


def apply_to_poses(
    params: np.ndarray,
    nominal: np.ndarray,
    gauge: Gauge,
    *,
    translation_frame: PoseTranslationFrame,
    keep: Collection[str] = (),
) -> np.ndarray:
    """``params`` moved by ``gauge``; parameters in ``keep`` stay as they are."""
    params = np.asarray(params, np.float64)
    out = params.copy()
    rotations = pose_rotations(params)
    out[:, :3] = pose_angles(rotations @ gauge.rotation)
    moved = np.einsum(
        "nij,j->ni", _translation_basis(params, nominal, translation_frame), gauge.shift
    )
    for axis, name in enumerate(_TRANSLATIONS):
        out[:, DOF_INDEX[name]] += moved[:, axis]
    out[:, DOF_INDEX["dx"]] += gauge.detector_offset
    for name in keep:
        out[:, DOF_INDEX[name]] = params[:, DOF_INDEX[name]]
    return out.astype(np.float32)


def apply_to_volume(volume: np.ndarray, grid: Grid, gauge: Gauge) -> np.ndarray:
    """The volume that, with the moved poses, predicts the same data.

    Poses ``P_i G`` image the object ``V'(o) = V(G o)``; this resamples ``V``
    there, trilinearly, with zeros outside the grid.
    """
    spacing = np.array([grid.vx, grid.vy, grid.vz], np.float64)
    origin = np.asarray(grid_volume_origin(grid), np.float64)
    # Index i' -> position o' = origin + D i' -> G o' -> index (G o' - origin) / D.
    matrix = np.eye(4)
    matrix[:3, :3] = gauge.rotation * spacing[None, :] / spacing[:, None]
    matrix[:3, 3] = (gauge.rotation @ origin + gauge.shift - origin) / spacing
    moved = ndimage.affine_transform(
        np.asarray(volume, np.float32), matrix, order=1, mode="constant"
    )
    return moved.astype(np.float32)


# Moving the volume may push this fraction of the object past the grid edges: a
# holder or stem reaching the edge breaks the symmetry too weakly to fix the
# estimate (the solve drifts along it anyway), an object filling the grid does not.
_MAX_LOST = 0.05
# The object: voxels above this fraction of the volume's 99.9th percentile. The
# low background a reconstruction spreads over the whole grid does not count.
_OBJECT_LEVEL = 0.1
# Motions moving no voxel this far (in voxels) are left alone: resampling would
# only blur the volume.
_NEGLIGIBLE = 1e-3


def _largest_displacement(gauge: Gauge, grid: Grid) -> float:
    """Upper bound on how far ``gauge`` moves any voxel of ``grid``, in voxels."""
    spacing = np.array([grid.vx, grid.vy, grid.vz], np.float64)
    extent = np.array([grid.nx, grid.ny, grid.nz], np.float64) * spacing
    origin = np.asarray(grid_volume_origin(grid), np.float64)
    corners = origin + extent * np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).reshape(3, -1).T
    moved = corners @ gauge.rotation.T + gauge.shift - corners
    return float(np.max(np.abs(moved) / spacing))


def least_motion_estimate(
    volume: np.ndarray,
    params: np.ndarray,
    *,
    nominal: np.ndarray,
    grid: Grid,
    translation_frame: PoseTranslationFrame,
    active: Collection[str],
    beam: bool,
    detector_offset: bool = False,
) -> tuple[np.ndarray, np.ndarray, Gauge | None]:
    """Move an alignment result (volume and poses) to its least-motion estimate.

    ``beam`` says the geometry is a cone beam, where along-beam ``dy`` is
    visible; parameters not in ``active`` keep their values. Returns the moved
    volume and poses and the gauge applied; a detector offset in it belongs in
    the detector centre (``Detector.det_center[0] += gauge.detector_offset``).

    The motion is a symmetry only for an object inside the grid. If moving the
    volume would push part of the object out (more than 5% of its integral;
    the faint background a reconstruction leaves everywhere does not count),
    the grid edge already fixes the estimate: it is returned unchanged with
    gauge None.
    """
    invisible = () if beam else ("dy",)
    gauge = least_motion_gauge(
        params,
        nominal,
        translation_frame=translation_frame,
        active=active,
        invisible=invisible,
        detector_offset=detector_offset,
    )
    if _largest_displacement(gauge, grid) < _NEGLIGIBLE and abs(gauge.detector_offset) < 1e-6:
        return np.asarray(volume), np.asarray(params), Gauge(np.eye(3), np.zeros(3))
    moved_volume = apply_to_volume(volume, grid, gauge)
    size = np.abs(np.asarray(volume, np.float32))
    obj = np.where(size > _OBJECT_LEVEL * np.percentile(size, 99.9), size, 0.0)
    before = float(np.sum(obj, dtype=np.float64))
    after = float(np.sum(apply_to_volume(obj, grid, gauge), dtype=np.float64))
    if before > 0 and abs(after - before) > _MAX_LOST * before:
        return np.asarray(volume), np.asarray(params), None
    keep = tuple(name for name in DOF_NAMES if name not in active or name in invisible)
    moved = apply_to_poses(params, nominal, gauge, translation_frame=translation_frame, keep=keep)
    return moved_volume, moved, gauge


__all__ = [
    "Gauge",
    "apply_to_poses",
    "apply_to_volume",
    "least_motion_estimate",
    "least_motion_gauge",
]
