"""Scans converted to and from ASTRA Toolbox geometries project identically."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj

# check-public-imports: allow-private
from tomojax._data.geometry_meta import AugmentedGeometry
import tomojax.geometry

pytestmark = pytest.mark.numerical


def _volume(n: int) -> np.ndarray:
    c = (np.arange(n) - (n - 1) / 2) / (n / 6)
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    blob = np.exp(-((x - 0.6) ** 2 + (y + 0.3) ** 2 + (z - 0.4) ** 2) * 1.5)
    return (blob + 0.6 * np.exp(-((x + 0.9) ** 2 * 3 + (y - 0.5) ** 2 + z**2))).astype(np.float32)


def _posed_cone_scan(n: int = 24, views: int = 30) -> tj.Scan:
    grid = tj.Grid(n, n, n, 0.5, 0.5, 0.5, vol_center=(0.0, 0.0, 1.2))
    detector = tj.Detector(36, 30, 0.8, 0.7, (0.6, -0.4))
    beam = tj.ConeBeam(40.0, 64.0, axis_offset=0.7)
    angles = np.linspace(0.0, 360.0, views, endpoint=False)
    geometry = tj.ConeGeometry(grid, detector, angles, beam, axis_unit=(0.02, -0.01, 1.0))
    rng = np.random.default_rng(4)
    poses = np.concatenate(
        [np.deg2rad(rng.normal(0, 0.4, (views, 3))), rng.normal(0, 0.3, (views, 3))], axis=1
    )
    data = np.zeros((views, detector.nv, detector.nu), np.float32)
    posed = AugmentedGeometry(geometry, poses.astype(np.float32), "detector")
    return tj.Scan(data, posed)  # pyright: ignore[reportArgumentType]


def test_a_scan_converted_to_astra_and_back_projects_the_same():
    scan = _posed_cone_scan()
    volume = _volume(scan.grid.nx)
    before = np.asarray(tj.project(scan.geometry, volume))
    projections, proj_geom, vol_geom = tj.Scan(before, scan.geometry).to_astra()
    assert projections.shape == (scan.detector.nv, len(scan.angles), scan.detector.nu)

    again = tj.Scan.from_astra(projections, proj_geom, vol_geom)

    np.testing.assert_allclose(np.asarray(again.projections), before)
    after = np.asarray(tj.project(again.geometry, volume))
    assert np.linalg.norm(after - before) / np.linalg.norm(before) < 1e-4
    assert again.grid.nx == scan.grid.nx


def test_a_circular_astra_scan_needs_no_pose_corrections():
    angles = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    s, c = np.sin(angles), np.cos(angles)
    zero = np.zeros_like(angles)
    vectors = np.stack(
        [
            s * 50,
            -c * 50,
            zero,
            -s * 30,
            c * 30,
            zero,
            c * 0.9,
            s * 0.9,
            zero,
            zero,
            zero,
            zero + 0.8,
        ],
        axis=1,
    )
    proj_geom = {
        "type": "cone_vec",
        "DetectorRowCount": 20,
        "DetectorColCount": 28,
        "Vectors": vectors,
    }
    window = {f"WindowMin{a}": -6.0 for a in "XYZ"} | {f"WindowMax{a}": 6.0 for a in "XYZ"}
    vol_geom = {"GridColCount": 24, "GridRowCount": 24, "GridSliceCount": 24, "option": window}

    scan = tj.Scan.from_astra(np.zeros((20, 24, 28), np.float32), proj_geom, vol_geom)

    assert scan.poses is None
    # ASTRA turns the source; TomoJAX turns the object, the other way.
    turned = np.mod(scan.angles[0] - scan.angles, 360.0)
    np.testing.assert_allclose(turned, np.rad2deg(angles), atol=1e-6)
    assert scan.geometry.beam.source_to_axis == pytest.approx(50.0)  # pyright: ignore[reportAttributeAccessIssue]
    assert scan.geometry.beam.source_to_detector == pytest.approx(80.0)  # pyright: ignore[reportAttributeAccessIssue]
    assert scan.grid.vx == pytest.approx(0.5)


def test_a_detector_that_moves_relative_to_the_source_is_refused():
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    s, c = np.sin(angles), np.cos(angles)
    zero = np.zeros_like(angles)
    vectors = np.stack(
        [s * 50, -c * 50, zero, -s * 30, c * 30, angles, c, s, zero, zero, zero, zero + 1], axis=1
    )
    proj_geom = {
        "type": "cone_vec",
        "DetectorRowCount": 4,
        "DetectorColCount": 4,
        "Vectors": vectors,
    }
    vol_geom = {"GridColCount": 4, "GridRowCount": 4, "GridSliceCount": 4}
    with pytest.raises(ValueError, match="rigid source and detector"):
        tj.Scan.from_astra(np.zeros((4, 8, 4), np.float32), proj_geom, vol_geom | {"option": {}})


@pytest.mark.gpu
@pytest.mark.parametrize("mirrored", [False, True])
def test_from_astra_projects_like_astra(mirrored):
    astra = pytest.importorskip("astra")
    n, rows, cols, views = 32, 28, 40, 40
    vol_geom = astra.create_vol_geom(n, n, n)
    volume_zyx = np.transpose(_volume(n), (2, 1, 0)).copy()
    angles = np.linspace(0, 2 * np.pi, views, endpoint=False)
    vectors = astra.geom_2vec(
        astra.create_proj_geom("cone", 1.1, 1.0, rows, cols, angles, 70.0, 50.0)
    )["Vectors"]
    tilt = Rotation.from_euler("xyz", [1.5, -2.0, 0.7], degrees=True).as_matrix()
    rng = np.random.default_rng(0)
    for i in range(views):
        jitter = Rotation.from_rotvec(np.deg2rad(rng.normal(0, 0.3, 3))).as_matrix()
        for k in range(4):
            point = jitter @ tilt @ vectors[i, 3 * k : 3 * k + 3]
            vectors[i, 3 * k : 3 * k + 3] = point + (np.array([0.2, -0.3, 2.0]) if k < 2 else 0)
    if mirrored:
        vectors[:, 6:9] *= -1
    proj_geom = astra.create_proj_geom("cone_vec", rows, cols, vectors)
    volume_id = astra.data3d.create("-vol", vol_geom, volume_zyx)
    sino_id, data = astra.create_sino3d_gpu(volume_id, proj_geom, vol_geom)
    astra.data3d.delete([volume_id, sino_id])

    scan = tj.Scan.from_astra(data, proj_geom, vol_geom)

    ours = np.asarray(tj.project(scan.geometry, np.transpose(volume_zyx, (2, 1, 0))))
    assert np.linalg.norm(ours - np.asarray(scan.projections)) / np.linalg.norm(data) < 2e-3


@pytest.mark.gpu
def test_a_multi_orbit_astra_scan_becomes_segments_that_project_like_astra():
    astra = pytest.importorskip("astra")
    n, rows, cols, views = 32, 40, 40, 30
    vol_geom = astra.create_vol_geom(n, n, n)
    volume_zyx = np.transpose(_volume(n), (2, 1, 0)).copy()
    angles = np.linspace(0, 2 * np.pi, views, endpoint=False)
    circle = astra.create_proj_geom("cone", 1.0, 1.0, rows, cols, angles, 70.0, 50.0)
    orbits = []
    for height in (-6.0, 0.0, 6.0):  # the source moves; the detector stays
        vectors = astra.geom_2vec(circle)["Vectors"].copy()
        vectors[:, 2] += height
        orbits.append(vectors)
    proj_geom = astra.create_proj_geom("cone_vec", rows, cols, np.concatenate(orbits))
    volume_id = astra.data3d.create("-vol", vol_geom, volume_zyx)
    sino_id, data = astra.create_sino3d_gpu(volume_id, proj_geom, vol_geom)
    astra.data3d.delete([volume_id, sino_id])

    scan = tj.Scan.from_astra(data, proj_geom, vol_geom)

    assert isinstance(scan.geometry, tj.geometry.ConeSegments)
    assert len(scan.geometry.segments) == 3
    ours = np.asarray(tj.project(scan.geometry, np.transpose(volume_zyx, (2, 1, 0))))
    assert np.linalg.norm(ours - np.asarray(scan.projections)) / np.linalg.norm(data) < 2e-3
    back, proj_back, vol_back = scan.to_astra()
    np.testing.assert_allclose(proj_back["Vectors"], proj_geom["Vectors"], atol=1e-4)
    recon = tj.reconstruct(scan, "cgls", iterations=10)
    assert np.isfinite(np.asarray(recon.volume)).all()
    with pytest.raises(ValueError, match="one source-detector arrangement per segment"):
        tj.reconstruct(scan, "fbp")


def _two_orbit_scan() -> tj.Scan:
    grid = tj.Grid(10, 10, 10, 1.0, 1.0, 1.0)
    angles = np.linspace(0.0, 360.0, 8, endpoint=False)
    low = tj.ConeGeometry(grid, tj.Detector(16, 14, 1.0, 1.0), angles, tj.ConeBeam(30.0, 50.0))
    high = tj.ConeGeometry(
        grid, tj.Detector(16, 14, 1.0, 1.0, (0.0, 4.0)), angles, tj.ConeBeam(32.0, 50.0)
    )
    empty = np.zeros((8, 14, 16), np.float32)
    return tj.Scan.combine([tj.Scan(empty, low), tj.Scan(empty, high)])


def test_combined_scans_project_each_segment_with_its_own_arrangement():
    scan = _two_orbit_scan()
    volume = _volume(10)
    combined = np.asarray(tj.project(scan.geometry, volume))
    for k, segment in enumerate(scan.geometry.segments):  # pyright: ignore[reportAttributeAccessIssue]
        alone = np.asarray(tj.project(segment, volume))
        np.testing.assert_allclose(combined[8 * k : 8 * (k + 1)], alone, rtol=1e-5, atol=1e-6)
    assert len(scan.angles) == 16


def test_segmented_scans_save_and_load(tmp_path):
    scan = _two_orbit_scan()
    volume = _volume(10)
    scan = tj.Scan(np.asarray(tj.project(scan.geometry, volume)), scan.geometry)
    tj.save(tmp_path / "scan.nxs", scan)

    loaded = tj.load(tmp_path / "scan.nxs")

    assert isinstance(loaded.geometry, tj.geometry.ConeSegments)
    np.testing.assert_allclose(
        np.asarray(tj.project(loaded.geometry, volume)), np.asarray(scan.projections), atol=1e-5
    )
    recon = tj.reconstruct(loaded, "cgls", iterations=3)
    tj.save(tmp_path / "recon.nxs", recon)
    assert tj.load_reconstruction(tmp_path / "recon.nxs").volume.shape == (10, 10, 10)
