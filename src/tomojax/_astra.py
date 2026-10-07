"""Convert scans to and from ASTRA Toolbox geometries.

ASTRA keeps the object fixed and moves a source and detector around it, view
by view (``cone_vec``: source, detector centre and the detector's pixel
vectors ``u`` and ``v``). TomoJAX fixes the source and detector (a
:class:`~tomojax.geometry.ConeBeam`) and moves the object. The two describe
the same scans whenever the source and detector move together, as on a
turntable or a gantry: each ASTRA view becomes a rigid object pose.

Conversion fits a circular orbit to those poses (a ``ConeGeometry``) and
keeps what remains, the scan's departure from a perfect circle, as per-view
pose corrections. The ASTRA world frame is TomoJAX's object frame, so volumes
convert by transposing ``(z, y, x)`` arrays to ``(x, y, z)``; a vertical
offset of the source from the volume becomes the grid's centre.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from tomojax.geometry import (
    ConeBeam,
    ConeGeometry,
    Detector,
    Grid,
    beam_of,
    grid_volume_origin,
    stack_view_poses,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.geometry import Geometry

# The source and detector may move relative to each other by this fraction of
# a detector pixel over the scan; TomoJAX models them as one rigid assembly.
_RIGID_TOLERANCE = 1e-3


def cone_vectors(proj_geom: Mapping[str, Any]) -> np.ndarray:
    """ASTRA ``cone`` or ``cone_vec`` geometry as ``(views, 12)`` vectors."""
    kind = str(proj_geom["type"])
    if kind == "cone_vec":
        return np.asarray(proj_geom["Vectors"], np.float64).reshape(-1, 12)
    if kind != "cone":
        raise ValueError(f"from_astra reads cone and cone_vec geometries, not {kind!r}")
    angles = np.asarray(proj_geom["ProjectionAngles"], np.float64)
    source, detector = (
        float(proj_geom["DistanceOriginSource"]),
        float(proj_geom["DistanceOriginDetector"]),
    )
    du, dv = float(proj_geom["DetectorSpacingX"]), float(proj_geom["DetectorSpacingY"])
    s, c, zero = np.sin(angles), np.cos(angles), np.zeros_like(angles)
    return np.stack(
        [
            s * source, -c * source, zero,
            -s * detector, c * detector, zero,
            c * du, s * du, zero,
            zero, zero, np.full_like(angles, dv),
        ],
        axis=1,
    )  # fmt: skip


def grid_from_astra(vol_geom: Mapping[str, Any]) -> Grid:
    """The TomoJAX grid of an ASTRA 3-D volume geometry (array ``(z, y, x)``)."""
    nx, ny, nz = (int(vol_geom[k]) for k in ("GridColCount", "GridRowCount", "GridSliceCount"))
    window = vol_geom.get("option", vol_geom)
    low = np.array([float(window[f"WindowMin{a}"]) for a in "XYZ"])
    high = np.array([float(window[f"WindowMax{a}"]) for a in "XYZ"])
    vx, vy, vz = ((high - low) / np.array([nx, ny, nz])).tolist()
    cx, cy, cz = ((high + low) / 2).tolist()
    return Grid(nx, ny, nz, vx, vy, vz, vol_center=(cx, cy, cz))


def _detector_axes(
    vectors: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unit detector u, v and beam direction per view, and data in TomoJAX column order."""
    source, centre, u, v = (vectors[:, 3 * k : 3 * k + 3] for k in range(4))
    e_u = u / np.linalg.norm(u, axis=1, keepdims=True)
    e_v = v / np.linalg.norm(v, axis=1, keepdims=True)
    if np.max(np.abs(np.sum(e_u * e_v, axis=1))) > 1e-6:
        raise ValueError("from_astra needs rectangular detector pixels (u perpendicular to v)")
    beam = np.cross(e_v, e_u)
    if np.median(np.sum((centre - source) * beam, axis=1)) < 0:
        # ASTRA's columns run the other way to TomoJAX's u for this detector.
        return -e_u, e_v, -beam, data[:, :, ::-1]
    return e_u, e_v, beam, data


def _scanner(
    vectors: np.ndarray, e_u: np.ndarray, e_v: np.ndarray, beam: np.ndarray, shape: tuple[int, int]
) -> tuple[float, float, Detector]:
    """Source-to-axis and source-to-detector distances and the detector, shared by every view."""
    source, centre = vectors[:, :3], vectors[:, 3:6]
    du, dv = np.linalg.norm(vectors[:, 6:9], axis=1), np.linalg.norm(vectors[:, 9:12], axis=1)
    offset = centre - source
    sdd = np.sum(offset * beam, axis=1)
    cu, cv = np.sum(offset * e_u, axis=1), np.sum(offset * e_v, axis=1)
    for name, values in (("pixel size", (du, dv)), ("source-detector", (sdd, cu, cv))):
        spread = max(float(np.ptp(value)) for value in values)
        if spread > _RIGID_TOLERANCE * float(np.mean(du)):
            raise ValueError(
                f"the {name} changes by {spread:.3g} over the scan; TomoJAX models a rigid "
                "source and detector"
            )
    rows, cols = shape
    centre_uv = (float(np.mean(cu)), float(np.mean(cv)))
    detector = Detector(cols, rows, float(np.mean(du)), float(np.mean(dv)), centre_uv)
    sod = float(np.mean(np.sum(-source * beam, axis=1)))
    return sod, float(np.mean(sdd)), detector


def _orbit(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lab axis and turn angles of the circle best fitting object-to-lab rotations.

    Fits ``rotation[i] ~ turn @ R_z(theta[i])``; returns ``turn`` and ``theta``.
    """
    theta = np.zeros(len(rotation))
    turn = np.eye(3)
    for _ in range(3):
        rz = Rotation.from_euler("z", theta[:, None]).as_matrix()
        w, _, vt = np.linalg.svd(np.einsum("nij,nkj->ik", rotation, rz))
        turn = w @ np.diag([1.0, 1.0, np.linalg.det(w @ vt)]) @ vt
        relative = np.einsum("ji,njk->nik", turn, rotation)
        theta = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
    return turn, theta


def from_astra(
    projections: np.ndarray, proj_geom: Mapping[str, Any], vol_geom: Mapping[str, Any]
) -> tuple[np.ndarray, ConeGeometry, Grid, np.ndarray]:
    """Projections ``(view, v, u)``, nominal geometry, grid and per-view poses.

    ``projections`` are in ASTRA's ``(rows, views, columns)`` layout.
    """
    vectors = cone_vectors(proj_geom)
    rows, cols = int(proj_geom["DetectorRowCount"]), int(proj_geom["DetectorColCount"])
    data = np.asarray(projections, np.float32)
    if data.shape != (rows, len(vectors), cols):
        raise ValueError(
            f"projections have shape {data.shape}; the geometry needs (rows, views, columns) "
            f"= {(rows, len(vectors), cols)}"
        )
    e_u, e_v, beam, data = _detector_axes(vectors, np.transpose(data, (1, 0, 2)))
    sod, sdd, detector = _scanner(vectors, e_u, e_v, beam, (rows, cols))
    # Object (ASTRA world) to lab: rows e_u, beam, e_v; the source lands at (0, -sod, 0).
    rotation = np.stack([e_u, beam, e_v], axis=1)
    translation = -np.einsum("nij,nj->ni", rotation, vectors[:, :3] + sod * beam)
    turn, theta = _orbit(rotation)
    axis = turn[:, 2]
    # A shift along the axis commutes with the rotation: it moves the grid
    # instead (object coordinates X' = X + height z).
    height = float(np.mean(translation @ axis))
    translation = translation - height * rotation[:, :, 2]
    grid = grid_from_astra(vol_geom)
    cx, cy, cz = grid.vol_center or (0.0, 0.0, 0.0)
    grid = replace(grid, vol_center=(cx, cy, cz + height))
    cone = ConeBeam(sod, sdd, axis_offset=float(np.mean(translation[:, 0])))
    axis_unit = (float(axis[0]), float(axis[1]), float(axis[2]))
    # ConeGeometry aligns +z to the axis, then turns by theta about it.
    aligned = ConeGeometry(grid, detector, [0.0], cone, axis_unit=axis_unit).poses()[0]
    offset_turn = aligned[:3, :3].T @ turn
    gamma = float(np.arctan2(offset_turn[1, 0], offset_turn[0, 0]))
    angles_deg = [float(t) for t in np.rad2deg(theta + gamma)]
    geometry = ConeGeometry(grid, detector, angles_deg, cone, axis_unit=axis_unit)
    # What the circle leaves, as detector-frame corrections: rotation N^T R,
    # translation t - n. Pose angles compose R_y(beta) R_x(alpha) R_z(phi).
    nominal = geometry.poses()
    residual = np.einsum("nji,njk->nik", nominal[:, :3, :3], rotation)
    beta, alpha, phi = Rotation.from_matrix(residual).as_euler("YXZ").T
    shifts = translation - nominal[:, :3, 3]
    poses = np.column_stack([alpha, beta, phi, shifts[:, 0], shifts[:, 2], shifts[:, 1]])
    return data, geometry, grid, poses.astype(np.float32)


def to_astra(
    projections: np.ndarray, geometry: Geometry, grid: Grid, detector: Detector
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """ASTRA ``(rows, views, columns)`` projections, ``cone_vec`` geometry and volume."""
    beam = beam_of(geometry)
    if beam is None:
        raise ValueError("to_astra converts cone-beam scans")
    views = int(np.asarray(projections).shape[0])
    poses = np.asarray(stack_view_poses(geometry, views), np.float64)
    rotation, translation = poses[:, :3, :3], poses[:, :3, 3]
    centre, e_u, e_v = beam.detector_frame(detector)

    def to_object(point: np.ndarray) -> np.ndarray:
        return np.einsum("nji,nj->ni", rotation, point[None, :] - translation)

    vectors = np.concatenate(
        [
            to_object(beam.source()),
            to_object(centre),
            np.einsum("nji,j->ni", rotation, e_u * detector.du),
            np.einsum("nji,j->ni", rotation, e_v * detector.dv),
        ],
        axis=1,
    )
    proj_geom = {
        "type": "cone_vec",
        "DetectorRowCount": int(detector.nv),
        "DetectorColCount": int(detector.nu),
        "Vectors": vectors,
    }
    origin = np.asarray(grid_volume_origin(grid), np.float64)
    spacing = np.array([grid.vx, grid.vy, grid.vz])
    low = origin - spacing / 2
    high = low + spacing * np.array([grid.nx, grid.ny, grid.nz])
    window = {f"WindowMin{a}": float(lo) for a, lo in zip("XYZ", low, strict=True)}
    window |= {f"WindowMax{a}": float(hi) for a, hi in zip("XYZ", high, strict=True)}
    vol_geom = {
        "GridColCount": int(grid.nx),
        "GridRowCount": int(grid.ny),
        "GridSliceCount": int(grid.nz),
        "option": window,
    }
    data = np.ascontiguousarray(np.transpose(np.asarray(projections, np.float32), (1, 0, 2)))
    return data, proj_geom, vol_geom


__all__ = ["cone_vectors", "from_astra", "grid_from_astra", "to_astra"]
