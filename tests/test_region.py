"""Data-selected regions preserve calibrated scans and do not force layers thin."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.geometry import stack_view_poses
from tomojax.io import build_geometry_from_dataset_metadata


def _layer_scan(layers: tuple[float, ...]) -> tuple[tj.Scan, list[tuple[np.ndarray, np.ndarray]]]:
    angles = np.linspace(0, 360, 128, endpoint=False)
    tilt = np.deg2rad(35)
    fixed = np.array([[1, 0, 0], [0, np.cos(tilt), np.sin(tilt)], [0, -np.sin(tilt), np.cos(tilt)]])
    u, v = np.meshgrid(np.arange(32) - 15.5, np.arange(32) - 15.5)
    detector_points = np.stack([u, v], -1)
    components = []
    for height in layers:
        for offset in (-7, 0, 7):
            components.extend(
                [
                    (np.array([offset, 0, height]), np.diag([0.8**2, 9**2, 1.3**2])),
                    (np.array([0, offset, height]), np.diag([9**2, 0.8**2, 1.3**2])),
                ]
            )
    data = []
    for angle in np.deg2rad(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = fixed @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        image = np.zeros((32, 32))
        for mean, covariance in components:
            marginal = (rotation @ covariance @ rotation.T)[np.ix_([0, 2], [0, 2])]
            delta = detector_points - (rotation @ mean)[[0, 2]]
            exponent = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(marginal), delta)
            image += (
                0.02
                * np.sqrt(2 * np.pi * np.linalg.det(covariance) / np.linalg.det(marginal))
                * np.exp(-0.5 * exponent)
            )
        data.append(image)
    geometry = tj.LaminographyGeometry(
        tj.Grid(32, 32, 32, 1, 1, 1), tj.Detector(32, 32, 1, 1), angles, tilt=35
    )
    return tj.Scan(np.array(data, np.float32), geometry, name="analytic layers"), components


@pytest.mark.slow
@pytest.mark.numerical
@pytest.mark.parametrize("layers", [(4.0,), (-10.0, 10.0)])
def test_region_from_independent_layers_retains_object_and_context(
    layers: tuple[float, ...],
) -> None:
    """Analytic ray marginals select a region without clipping either real layer."""
    scan, components = _layer_scan(layers)
    estimate = tj.estimate_region(scan)
    grid = estimate.grid
    assert estimate.scan.projections is scan.projections
    assert estimate.scan.name == scan.name
    np.testing.assert_array_equal(estimate.scan.angles, scan.angles)
    assert estimate.scan.detector == scan.detector
    assert grid.vz == scan.detector.dv
    assert grid.vol_center is not None
    lower, upper = grid.vol_center[2] - grid.nz / 2, grid.vol_center[2] + grid.nz / 2
    assert lower < min(layers) - 3.9
    assert upper > max(layers) + 3.9
    if len(layers) == 1:
        assert estimate.info["status"] == "conditional_cv_region"
        assert grid.nz < 64
    else:
        assert estimate.info["status"] == "broad_uncertain"
        assert grid.nz >= 64
    xyz = np.stack(
        np.meshgrid(
            *[
                (np.arange(n) - (n - 1) / 2) + center
                for n, center in zip(
                    (grid.nz, grid.ny, grid.nx), grid.vol_center[::-1], strict=True
                )
            ],
            indexing="ij",
        )[::-1],
        -1,
    )
    truth = np.zeros(xyz.shape[:-1], np.float32)
    for mean, covariance in components:
        delta = xyz - mean
        truth += 0.02 * np.exp(
            -0.5 * np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
        )
    reconstructed = np.asarray(
        tj.reconstruct(estimate.scan, "fista", iterations=60, nonnegative=True, tv_weight=0).volume
    )
    error = np.linalg.norm(reconstructed - truth) / np.linalg.norm(truth)
    assert error < 0.4


@pytest.mark.numerical
def test_uninformative_region_is_broad_and_does_not_recalibrate() -> None:
    """Uninformative data keeps roll, center and poses while reporting uncertainty."""
    scan, _ = _layer_scan((4.0,))
    poses = np.zeros((len(scan.angles), 6), np.float32)
    poses[:, 0] = 0.003 * np.sin(np.deg2rad(scan.angles))
    poses[:, 3] = 0.3 * np.cos(np.deg2rad(scan.angles))
    detector = replace(scan.detector, center=(1.5, -0.4))
    _, _, geometry = build_geometry_from_dataset_metadata(
        {
            "grid": scan.grid.to_dict(),
            "detector": detector.to_dict(),
            "angles": scan.angles,
            "geometry_type": "lamino",
            "tilt": 35,
            "detector_roll": 7,
            "align_params": poses,
        },
        apply_poses=True,
    )
    scan = replace(scan, projections=np.zeros_like(scan.projections), geometry=geometry)
    estimate = tj.estimate_region(scan)
    assert estimate.info["status"] == "broad_uncertain"
    assert estimate.info["probes"] == []
    assert estimate.grid.vol_center == (0.0, 0.0, 0.0)
    assert estimate.scan.projections is scan.projections
    np.testing.assert_array_equal(estimate.scan.poses, scan.poses)
    np.testing.assert_array_equal(
        np.asarray(stack_view_poses(estimate.scan.geometry, len(scan.angles))),
        np.asarray(stack_view_poses(scan.geometry, len(scan.angles))),
    )
    assert estimate.scan.geometry.detector_roll == 7
    assert estimate.scan.detector == detector
    parallel = replace(scan, geometry=tj.ParallelGeometry(scan.grid, scan.detector, scan.angles))
    with pytest.raises(ValueError, match="laminography"):
        tj.estimate_region(parallel)


@pytest.mark.numerical
def test_shallow_region_coarsens_a_fixed_physical_survey(monkeypatch) -> None:
    scan, _ = _layer_scan((4.0,))
    scan = replace(
        scan,
        projections=np.zeros_like(scan.projections),
        geometry=replace(scan.geometry, tilt=0.1),
    )
    monkeypatch.setattr("tomojax._region.device_free_memory_bytes", lambda **kw: 2**20)
    monkeypatch.setattr(
        "tomojax._region.reconstruct_arrays",
        lambda method, geometry, grid, *args: (np.zeros((grid.nz, grid.ny, grid.nx)), {}),
    )
    estimate = tj.estimate_region(scan)
    grid = estimate.grid
    assert grid.nx * grid.ny * grid.nz <= 0.6 * 2**20 / 32
    assert estimate.info["status"] == "broad_uncertain"
    assert np.isfinite(estimate.info["focus"]).all()


@pytest.mark.numerical
def test_support_scoring_stays_on_the_budgeted_preview_grid(monkeypatch) -> None:
    import tomojax._region as region

    # A 35-degree scan whose budgeted preview is 16 x 16 x 8 voxels of 16: widening
    # it to its padded depth (8 x 16) used to rebuild a 24 x 24 x 8 scoring grid.
    angles = np.linspace(0, 360, 16, endpoint=False)
    geometry = tj.LaminographyGeometry(
        tj.Grid(32, 32, 32, 1, 1, 1), tj.Detector(32, 32, 1, 1), angles, tilt=35
    )
    scan = tj.Scan(np.zeros((16, 32, 32), np.float32), geometry)
    preview = tj.Grid(16, 16, 8, 16, 16, 16)
    z = (np.arange(8) - 3.5) * 16
    evidence = region._Evidence(preview, z, np.ones(8), 0.0, True, 32.0, 0.0)
    project, sizes = region.project, []

    def recorded(geometry, volume, *, grid):
        sizes.append(grid.nx * grid.ny * grid.nz)
        return project(geometry, volume, grid=grid)

    monkeypatch.setattr(region, "project", recorded)
    region._score_supports(scan, evidence)
    assert sizes and max(sizes) <= 16 * 16 * 8
