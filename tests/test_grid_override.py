"""Changing the reconstruction grid must not re-encode a scan's measurements or poses."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.io import ProjectionDataset, build_geometry_from_dataset_metadata
from tomojax.recon import CGLSConfig


def _geometry(kind: str, posed: bool = False):
    grid = tj.Grid(6, 5, 4, 1.0, 0.8, 1.2)
    detector = tj.Detector(9, 7, 0.9, 1.1, (0.2, -0.3))
    angles = np.array(
        [17.123456789, 59.876543219, 132.135791357, 190.314159265, 250.987654321, 290.246802468]
    )
    if kind == "parallel":
        geometry = tj.ParallelGeometry(grid, detector, angles)
    elif kind == "lamino":
        geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=27.0)
    elif kind == "axis":
        geometry = tj.geometry.RotationAxisGeometry(grid, detector, angles, (0.2, -0.3, 0.93))
    else:
        geometry = tj.ConeGeometry(
            grid, detector, angles, tj.ConeBeam(18.0, 30.0, detector_roll_deg=7.0, axis_offset=0.4)
        )
    if kind == "segments":
        geometry = tj.geometry.ConeSegments(
            (
                replace(geometry, angles=angles[:3]),
                replace(
                    geometry, angles=angles[3:], beam=tj.ConeBeam(21.0, 35.0, axis_offset=-0.2)
                ),
            )
        )
    if not posed:
        return geometry
    # Public metadata builds pose wrappers, also for individual cone segments.
    if kind == "segments":
        meta = {
            "geometry_type": "cone",
            "cone_segments": [
                {"views": len(s.angles), "detector": s.detector.to_dict(), **s.geometry_metadata()}
                for s in geometry.segments
            ],
        }
    elif kind == "cone":
        meta = {"geometry_type": "cone", **geometry.geometry_metadata()}
    elif kind == "axis":
        meta = {"geometry_type": "lamino", "axis_unit_lab": [0.2, -0.3, 0.93]}
    else:
        meta = {"geometry_type": kind}
        if kind == "lamino":
            meta |= {"tilt_deg": 27.0, "tilt_about": "x"}
    table = np.zeros((6, 6), np.float32)
    table[:, :5] = [0.01, -0.02, 0.03, 0.2, -0.1]
    meta |= {
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "angles": angles,
        "align_params": table,
        "align_gauge": {"pose_translation_frame": "object"},
    }
    return build_geometry_from_dataset_metadata(meta, poses=True)[2]


@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis", "cone", "segments"])
@pytest.mark.parametrize("posed", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_grid_override_keeps_original_angles_poses_and_projection_storage(kind, posed, resident):
    geometry = _geometry(kind, posed)
    data = np.zeros((6, 7, 9), np.float32)
    projections = jnp.asarray(data) if resident else data
    source = ProjectionDataset(
        data, np.asarray(geometry.angles), grid=geometry.grid, detector=geometry.detector
    )
    scan = tj.Scan(projections, geometry, name="original", source=source)
    original_angles = scan.angles.copy()
    original_poses = np.asarray([geometry.pose_for_view(i) for i in range(6)])
    region = tj.Grid(5, 4, 3, 1.1, 0.9, 1.3)
    result = tj.reconstruct(scan, "cgls", iterations=0, grid=region)
    assert result.scan.projections is projections
    assert result.scan.source is source and result.scan.name == scan.name
    assert result.grid == region and result.scan.geometry.detector == geometry.detector
    np.testing.assert_array_equal(result.scan.angles, original_angles)
    np.testing.assert_array_equal(
        [result.scan.geometry.pose_for_view(i) for i in range(6)], original_poses
    )
    assert scan.grid == geometry.grid and scan.projections is projections
    np.testing.assert_array_equal(result.volume, np.zeros((5, 4, 3), np.float32))


@pytest.mark.parametrize("kind", ["parallel", "lamino", "axis", "cone", "segments"])
def test_equal_grid_override_returns_the_same_scan(kind):
    geometry = _geometry(kind, posed=True)
    scan = tj.Scan(jnp.zeros((6, 7, 9), jnp.float32), geometry)
    assert tj.reconstruct(scan, "cgls", iterations=0, grid=scan.grid).scan is scan


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_grid_override_reconstruction_matches_dense_operator_and_known_image(kind):
    geometry = _geometry(kind)
    region = tj.Grid(5, 4, 3, 1.1, 0.9, 1.3)
    reference_geometry = replace(geometry, grid=region)
    x, y, z = np.meshgrid(
        np.arange(5) - 2,
        np.arange(4) - 1.5,
        np.arange(3) - 1,
        indexing="ij",
    )
    truth = np.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2)).astype(np.float32)
    data = tj.project(reference_geometry, truth)
    result = tj.reconstruct(
        tj.Scan(data, geometry), "cgls", config=CGLSConfig(iterations=200, rtol=1e-8), grid=region
    )
    # A dense FP64 solve uses columns of the unchanged reference projector.
    basis = np.eye(truth.size, dtype=np.float32).reshape(-1, *truth.shape)
    matrix = np.stack(
        [np.asarray(tj.project(reference_geometry, column)).ravel() for column in basis], axis=1
    ).astype(np.float64)
    dense = np.linalg.lstsq(matrix, np.asarray(data).ravel(), rcond=None)[0].reshape(truth.shape)
    image = np.asarray(result.volume)
    assert np.linalg.norm(image - truth) / np.linalg.norm(truth) < 2e-4
    assert np.linalg.norm(image - dense) / np.linalg.norm(dense) < 2e-4
