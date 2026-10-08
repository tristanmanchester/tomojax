"""The workflow API: scans, projection, reconstruction, alignment and files."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

import tomojax as tj


def _phantom(n: int) -> np.ndarray:
    volume = np.zeros((n, n, n), np.float32)
    q = n // 4
    volume[q : 3 * q, q + 1 : 3 * q - 1, q - 1 : 3 * q + 1] = 1.0
    volume[n // 2 - 2 : n // 2 + 2, n // 2 - 2 : n // 2 + 2, n // 2 - 2 : n // 2 + 2] = 2.0
    return volume


def _geometries(
    n: int,
) -> dict[str, tj.ParallelGeometry | tj.LaminographyGeometry | tj.ConeGeometry]:
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    detector = tj.Detector(int(1.5 * n), int(1.5 * n), 1.0, 1.0)
    return {
        "parallel": tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 40, endpoint=False)),
        "lamino": tj.LaminographyGeometry(
            grid, detector, np.linspace(0, 360, 40, endpoint=False), tilt_deg=30
        ),
        "cone": tj.ConeGeometry(
            grid, detector, np.linspace(0, 360, 60, endpoint=False), tj.ConeBeam(3 * n, 4.5 * n)
        ),
    }


def test_import_is_lazy() -> None:
    code = "import sys, tomojax; print('jax' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_scan_rejects_projections_that_do_not_match_the_geometry() -> None:
    geometry = _geometries(8)["parallel"]
    with pytest.raises(ValueError, match="40 views of 12 rows x 12 columns"):
        tj.Scan(np.zeros((40, 12, 11), np.float32), geometry)


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_backproject_is_the_transpose_of_project(kind: str) -> None:
    geometry = _geometries(12)[kind]
    rng = np.random.default_rng(0)
    x = rng.random((12, 12, 12)).astype(np.float32)
    projections = tj.project(geometry, x)
    y = rng.random(projections.shape).astype(np.float32)
    lhs = np.vdot(np.asarray(projections, np.float64), y)
    rhs = np.vdot(x.astype(np.float64), np.asarray(tj.backproject(geometry, y), np.float64))
    assert abs(lhs - rhs) <= 1e-5 * abs(lhs)
    with pytest.raises(ValueError, match="does not match"):
        tj.project(geometry, x[:-1])


@pytest.mark.parametrize("kind", ["parallel", "cone"])
def test_reconstruct_runs_every_method(kind: str) -> None:
    n = 16
    geometry = _geometries(n)[kind]
    truth = _phantom(n)
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    errors = {}
    for method, options in (
        ("fbp", {}),
        ("cgls", {"iterations": 25}),
        ("fista", {"iterations": 25, "tv_weight": 1e-4, "nonnegative": True}),
        ("spdhg", {"iterations": 25, "tv_weight": 1e-4, "seed": 1}),
    ):
        recon = tj.reconstruct(scan, method, **options)  # type: ignore[arg-type]
        assert recon.method == method and recon.scan is scan and recon.grid == scan.grid
        volume = np.asarray(recon.volume)
        errors[method] = np.linalg.norm(volume - truth) / np.linalg.norm(truth)
    assert errors["cgls"] < 0.2 and errors["fbp"] < 0.4, errors
    with pytest.raises(ValueError, match="method 'cgls' does not take filter"):
        tj.reconstruct(scan, "cgls", filter="hann")
    with pytest.raises(ValueError, match="method must be one of"):
        tj.reconstruct(scan, "sirt")  # type: ignore[arg-type]


def test_reconstruct_on_another_grid() -> None:
    geometry = _geometries(16)["parallel"]
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom(16))), geometry)
    region = tj.Grid(8, 8, 16, 1.0, 1.0, 1.0)
    recon = tj.reconstruct(scan, grid=region)
    assert np.asarray(recon.volume).shape == (8, 8, 16) and recon.grid == region


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_scans_and_reconstructions_round_trip_through_files(kind: str, tmp_path: Path) -> None:
    geometry = _geometries(8)[kind]
    if isinstance(geometry, tj.ConeGeometry):
        geometry = tj.ConeGeometry(
            geometry.grid, geometry.detector, geometry.thetas_deg,
            tj.ConeBeam(24, 36, detector_roll_deg=0.4, axis_offset=0.7),
        )  # fmt: skip
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom(8))), geometry, name="part")
    tj.save(tmp_path / "scan.nxs", scan)
    loaded = tj.load(tmp_path / "scan.nxs")
    assert type(loaded.geometry) is type(geometry) and loaded.name == "part"
    assert loaded.grid == scan.grid and loaded.detector == scan.detector
    np.testing.assert_allclose(loaded.angles, scan.angles, atol=1e-5)
    np.testing.assert_array_equal(loaded.projections, scan.projections)
    if isinstance(geometry, tj.ConeGeometry):
        assert loaded.geometry.beam == geometry.beam  # pyright: ignore[reportAttributeAccessIssue]
    recon = tj.reconstruct(loaded)
    tj.save(tmp_path / "recon.nxs", recon)
    back = tj.load_reconstruction(tmp_path / "recon.nxs")
    np.testing.assert_allclose(back.volume, np.asarray(recon.volume), rtol=1e-6, atol=1e-6)
    assert back.method == "fbp"
    with pytest.raises(FileNotFoundError, match="no such file"):
        tj.load(tmp_path / "missing.nxs")


def test_alignment_modes_plan_their_schedules() -> None:
    from tomojax.alignment.api import alignment_plan

    grid = tj.Grid(128, 128, 128, 1.0, 1.0, 1.0)
    pose = alignment_plan("pose", grid)
    assert pose.pose_solver == "coupled" and pose.levels == (4, 2, 1)
    assert (
        pose.config.schedule == "lightning_pose"
        and pose.config.pose_translation_frame == "detector"
    )
    assert alignment_plan("cor", grid).config.schedule == "cor"
    assert alignment_plan("COR_then_pose", grid).config.schedule == "cor_then_pose"
    full = alignment_plan("full", grid, quality="reference", freeze=("dy",))
    assert full.config.schedule == "setup_safe" and full.levels == (4, 2, 1)
    assert full.config.align_profile == "tortoise" and full.config.freeze_dofs == ("dy",)
    with pytest.raises(ValueError, match="alignment mode must be one of"):
        alignment_plan("auto", grid)


@pytest.mark.gpu
def test_align_calibrates_the_axis_and_returns_a_corrected_scan(tmp_path: Path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    n = 32
    nominal = _geometries(n)["cone"]
    truth = tj.ConeGeometry(
        nominal.grid, nominal.detector, nominal.thetas_deg, tj.ConeBeam(96, 144, axis_offset=1.7)
    )
    scan = tj.Scan(np.asarray(tj.project(truth, _phantom(n))), nominal)
    result = tj.align(scan, mode="cor")
    assert abs(result.scan.geometry.beam.axis_offset - 1.7) < 0.1  # pyright: ignore[reportAttributeAccessIssue]
    assert result.poses.shape == (60, 6) and not result.poses.any()
    tj.save(tmp_path / "aligned.nxs", result)
    reloaded = tj.load(tmp_path / "aligned.nxs")
    assert reloaded.geometry.beam == result.scan.geometry.beam  # pyright: ignore[reportAttributeAccessIssue]


def test_aligning_a_loaded_scan_keeps_its_corrections(tmp_path: Path) -> None:
    n = 16
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, tj.Detector(n, n, 1.0, 1.0), np.linspace(0, 180, 20))
    data = np.array(tj.project(geometry, _phantom(n)))
    data[::2] = np.roll(data[::2], 1, axis=2)  # every other view one pixel across
    tj.save(tmp_path / "scan.nxs", tj.Scan(data, geometry))

    result = tj.align(tj.load(tmp_path / "scan.nxs"), levels=(1,))

    assert result.scan.poses is not None and np.abs(result.poses).max() > 0.5
    np.testing.assert_allclose(result.scan.poses, result.poses, atol=1e-6)


def test_binning_averages_pixels_and_keeps_the_detector_in_place():
    grid = tj.Grid(24, 24, 24, 1.0, 1.0, 1.0)
    detector = tj.Detector(41, 33, 0.5, 0.5, (0.3, -0.2))
    geometry = tj.ConeGeometry(
        grid, detector, np.linspace(0, 360, 24, endpoint=False), tj.ConeBeam(60.0, 90.0)
    )
    c = (np.arange(24) - 11.5) / 4
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    volume = np.exp(-(x**2 + y**2 + z**2)).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)

    binned = scan.binned(2)

    # The odd last column and row are dropped, shifting the centre by half a pixel.
    assert binned.projections.shape == (24, 16, 20)
    assert binned.detector.du == binned.detector.dv == 1.0
    assert binned.detector.det_center == pytest.approx((0.05, -0.45))
    direct = np.asarray(tj.project(binned.geometry, volume))
    assert np.linalg.norm(np.asarray(binned.projections) - direct) / np.linalg.norm(direct) < 0.01
    assert scan.binned(1) is scan
    with pytest.raises(ValueError, match="at least 1"):
        scan.binned(0)


def test_iterative_reconstruction_suggests_binning_a_detector_finer_than_the_grid() -> None:
    import warnings

    grid = tj.Grid(16, 16, 16, 1.0, 1.0, 1.0)
    fine = tj.Detector(40, 40, 0.4, 0.4)
    beam = tj.ConeBeam(64.0, 96.0)  # magnification 1.5: 0.27 pixels at the axis
    angles = np.linspace(0, 360, 12, endpoint=False)
    scan = tj.Scan(np.zeros((12, 40, 40), np.float32), tj.ConeGeometry(grid, fine, angles, beam))
    with pytest.warns(UserWarning, match=r"3\.8 times more finely.*binned\(3\)"):
        tj.reconstruct(scan, "cgls", iterations=1)
    matched = tj.Detector(24, 24, 1.5, 1.5)
    scan = tj.Scan(np.zeros((12, 24, 24), np.float32), tj.ConeGeometry(grid, matched, angles, beam))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tj.reconstruct(scan, "cgls", iterations=1)
        tj.reconstruct(
            tj.Scan(np.zeros((12, 40, 40), np.float32), tj.ConeGeometry(grid, fine, angles, beam)),
            "fbp",
        )
