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
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
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


def _circle_vectors(du, detector=30.0, source=50.0, wobble=0.0) -> np.ndarray:
    angles = np.linspace(0.0, 2 * np.pi, 20, endpoint=False)
    s, c = np.sin(angles), np.cos(angles)
    zero = np.zeros_like(angles)
    columns = [s * source, -c * source, wobble + zero, -s * detector, c * detector, wobble + zero]
    return np.stack([*columns, c * du, s * du, zero, zero, zero, zero + du], axis=1)


def _astra_scan(vectors: np.ndarray) -> tj.Scan:
    proj_geom = {
        "type": "cone_vec",
        "DetectorRowCount": 8,
        "DetectorColCount": 8,
        "Vectors": vectors,
    }
    window = {f"WindowMin{a}": -4.0 for a in "XYZ"} | {f"WindowMax{a}": 4.0 for a in "XYZ"}
    vol_geom = {"GridColCount": 8, "GridRowCount": 8, "GridSliceCount": 8, "option": window}
    return tj.Scan.from_astra(np.zeros((8, len(vectors), 8), np.float32), proj_geom, vol_geom)


def test_astra_segments_share_one_pixel_pitch_despite_roundoff():
    # The pixel vectors' norms differ in the last bits between the two arrangements.
    second = np.arange(20) >= 10
    scan = _astra_scan(_circle_vectors(0.0748, np.where(second, 40.0, 30.0), wobble=7.0 * second))
    assert isinstance(scan.geometry, tj.geometry.ConeSegments)


def test_astra_wobble_that_is_magnified_beyond_a_hundredth_of_a_pixel_is_kept():
    # 5e-5 is below a hundredth of the 0.02 pitch but 0.1 pixel at 40x magnification.
    wobble = 5e-5 * np.where(np.arange(20) % 2, 1, -1)
    assert _astra_scan(_circle_vectors(0.02, 78.0, 2.0, wobble)).poses is not None
    assert _astra_scan(_circle_vectors(0.02, 78.0, 2.0)).poses is None


@pytest.mark.gpu
@pytest.mark.parametrize("mirrored", [False, True])
def test_from_astra_projects_like_astra(mirrored):
    astra = pytest.importorskip("astra")
    n, rows, cols, views = 32, 28, 40, 40
    vol_geom = astra.create_vol_geom(n, n, n)
    volume_zyx = _volume(n)
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

    ours = np.asarray(tj.project(scan.geometry, volume_zyx))
    assert np.linalg.norm(ours - np.asarray(scan.projections)) / np.linalg.norm(data) < 2e-3


@pytest.mark.gpu
def test_a_multi_orbit_astra_scan_becomes_segments_that_project_like_astra():
    astra = pytest.importorskip("astra")
    n, rows, cols, views = 32, 40, 40, 30
    vol_geom = astra.create_vol_geom(n, n, n)
    volume_zyx = _volume(n)
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
    ours = np.asarray(tj.project(scan.geometry, volume_zyx))
    assert np.linalg.norm(ours - np.asarray(scan.projections)) / np.linalg.norm(data) < 2e-3
    back, proj_back, vol_back = scan.to_astra()
    np.testing.assert_allclose(proj_back["Vectors"], proj_geom["Vectors"], atol=1e-4)
    recon = tj.reconstruct(scan, "cgls", iterations=10)
    assert np.isfinite(np.asarray(recon.volume)).all()
    # FDK weights each segment's own source-detector arrangement.
    fdk = tj.reconstruct(scan, "fbp")
    assert np.isfinite(np.asarray(fdk.volume)).all()


def _two_orbit_scan() -> tj.Scan:
    grid = tj.Grid(10, 10, 10, 1.0, 1.0, 1.0)
    angles = np.linspace(0.0, 360.0, 8, endpoint=False)
    low = tj.ConeGeometry(grid, tj.Detector(16, 14, 1.0, 1.0), angles, tj.ConeBeam(30.0, 50.0))
    high = tj.ConeGeometry(
        grid, tj.Detector(16, 14, 1.0, 1.0, (0.0, 4.0)), angles, tj.ConeBeam(32.0, 50.0)
    )
    empty = np.zeros((8, 14, 16), np.float32)
    return tj.Scan.combine([tj.Scan(empty, low), tj.Scan(empty, high)])


def test_astra_export_keeps_pose_metadata_on_host():
    import subprocess
    import sys

    code = """
import sys, numpy as np, tomojax as tj
for segmented in (False, True):
    grid, detector = tj.Grid(7, 5, 3, 1, 1, 1), tj.Detector(9, 7, 1, 1)
    scans = [tj.Scan(np.ones((3, 7, 9), np.float32), tj.ConeGeometry(
        grid, detector, [0, 45, 135], tj.ConeBeam(30, 50, source_height=h))) for h in (0, 3)]
    scan = tj.Scan.combine(scans) if segmented else scans[0]
    scan = scan.with_poses(np.full((len(scan.angles), 6), .001, np.float32))
    assert np.isfinite(scan.to_astra()[1]["Vectors"]).all()
assert "jax" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)


def test_combined_scans_project_each_segment_with_its_own_arrangement():
    scan = _two_orbit_scan()
    volume = _volume(10)
    combined = np.asarray(tj.project(scan.geometry, volume))
    for k, segment in enumerate(scan.geometry.segments):  # pyright: ignore[reportAttributeAccessIssue]
        alone = np.asarray(tj.project(segment, volume))
        np.testing.assert_allclose(combined[8 * k : 8 * (k + 1)], alone, rtol=1e-5, atol=1e-6)
    assert len(scan.angles) == 16


def test_a_posed_multi_orbit_scan_converts_to_astra_and_back_projects_the_same():
    scan = _two_orbit_scan()
    poses = np.zeros((16, 6), np.float32)
    poses[:, 3:] = np.random.default_rng(5).normal(0, 0.3, (16, 3))
    posed = AugmentedGeometry(scan.geometry, poses, "detector")
    volume = _volume(10)
    before = np.asarray(tj.project(posed, volume))

    projections, proj_geom, vol_geom = tj.Scan(before, posed).to_astra()  # pyright: ignore[reportArgumentType]

    again = tj.Scan.from_astra(projections, proj_geom, vol_geom)
    after = np.asarray(tj.project(again.geometry, volume))
    assert np.linalg.norm(after - before) / np.linalg.norm(before) < 1e-4


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


def test_independently_imported_orbits_keep_one_object_frame(tmp_path):
    vectors = [_circle_vectors(0.7, wobble=h) for h in (-2.5, 3.0)]
    combined = tj.Scan.combine([_astra_scan(v) for v in vectors])
    assert combined.grid.vol_center == (0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        combined.to_astra()[1]["Vectors"], np.concatenate(vectors), atol=1e-6
    )
    for scan in (combined, combined.selected(slice(0, 20))):
        vectors = scan.to_astra()[1]["Vectors"]
        zeros = np.zeros((len(scan.angles), 6), np.float32)
        restored = scan.with_poses(zeros).with_poses(None)
        np.testing.assert_allclose(restored.to_astra()[1]["Vectors"], vectors, atol=1e-6)
        tj.save(tmp_path / f"calibrated_{len(scan.angles)}.npz", restored)
        loaded = tj.load(tmp_path / f"calibrated_{len(scan.angles)}.npz")
        np.testing.assert_allclose(loaded.to_astra()[1]["Vectors"], vectors, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_positive_object_angles_project_one_phantom_in_all_three_libraries(monkeypatch):
    import importlib
    from pathlib import Path

    astra = pytest.importorskip("astra")
    tigre = pytest.importorskip("tigre")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    adapter = importlib.import_module("tigre_cone")
    grid = tj.Grid(32, 32, 32, 1.0, 1.0, 1.0)
    detector = tj.Detector(48, 40, 1.0, 1.0)
    geometry = tj.ConeGeometry(grid, detector, [0, 27, 90, 177, 271], tj.ConeBeam(80, 130))
    volume = _volume(32)
    empty = np.zeros((5, detector.nv, detector.nu), np.float32)
    _, projection_geometry, volume_geometry = tj.Scan(empty, geometry).to_astra()
    # The object turns +90 about z; the source in its fixed frame turns -90.
    np.testing.assert_allclose(
        projection_geometry["Vectors"][[0, 2], :3], [[0, -80, 0], [-80, 0, 0]], atol=1e-5
    )
    external_volume = np.ascontiguousarray(volume)
    volume_id = astra.data3d.create("-vol", volume_geometry, external_volume)
    sino_id = None
    try:
        sino_id, projected = astra.create_sino3d_gpu(
            volume_id, projection_geometry, volume_geometry
        )
        astra_projection = projected.transpose(1, 0, 2).copy()
    finally:
        astra.data3d.delete(volume_id)
        if sino_id is not None:
            astra.data3d.delete(sino_id)
    geo, angles = adapter.geometry(projection_geometry["Vectors"], grid=grid, detector=detector)
    projected = tigre.Ax(external_volume, geo, angles, projection_type="interpolated")
    # Finish external kernels before JAX claims an allocator pool: TIGRE refuses
    # an otherwise idle GPU when another allocator reserves half its memory.
    ours = np.asarray(tj.Projector(geometry)(volume))
    assert np.linalg.norm(ours - astra_projection) / np.linalg.norm(ours) < 2e-3
    assert np.linalg.norm(ours - projected) / np.linalg.norm(ours) < 1e-2
