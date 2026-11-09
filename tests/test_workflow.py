"""The workflow API: scans, projection, reconstruction, alignment and files."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

import h5py
import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Rings
from tomojax.geometry.api import view_frames


def test_trajectory_groups_survive_combine_select_crop_and_file(tmp_path: Path) -> None:
    grid, detector = tj.Grid(4, 4, 4, 1, 1, 1), tj.Detector(4, 4, 1, 1)
    parts = tuple(
        tj.ConeGeometry(grid, replace(detector, center=(i / 10, 0)), [i * 30], tj.ConeBeam(20, 40))
        for i in range(4)
    )
    geometry = tj.geometry.ConeSegments(parts, trajectory_lengths=(4,))
    scan = tj.Scan(np.ones((4, 4, 4), np.float32), geometry)
    scan = scan.with_poses(np.full((4, 6), 0.01, np.float32))
    combined = tj.Scan.combine([scan, scan, scan])
    assert combined.geometry.trajectory_lengths == (4, 4, 4)
    np.testing.assert_array_equal(
        view_frames(combined.geometry), np.tile(view_frames(scan.geometry), (3, 1))
    )
    selected = combined.selected([0, 2, 8, 10]).cropped(slice(1, 3), slice(1, 3))
    assert selected.geometry.trajectory_lengths == (2, 2)
    poses = np.zeros((4, 6), np.float32)
    poses[:, 3] = 0.1
    posed = selected.with_poses(poses)
    path = tmp_path / "trajectories.nxs"
    tj.save(path, posed)
    loaded = tj.load(path)
    assert loaded.geometry.trajectory_lengths == (2, 2)
    np.testing.assert_allclose(view_frames(loaded.geometry), view_frames(posed.geometry), atol=1e-6)
    np.testing.assert_array_equal(
        Rings(loaded.geometry).apply(loaded.projections),
        Rings(posed.geometry).apply(posed.projections),
    )


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
        tj.reconstruct(scan, "not-a-method")  # type: ignore[arg-type]


def test_reconstruct_takes_the_methods_configuration_under_its_keywords() -> None:
    from tomojax.recon import CGLSConfig, FistaConfig, fista_tv

    geometry = _geometries(16)["parallel"]
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom(16))), geometry)
    config = FistaConfig(regulariser="huber_tv", huber_delta=0.05, tv_weight=1e-3, iterations=4)
    recon = tj.reconstruct(scan, "fista", config=config)
    assert recon.info["config"] == config and recon.info["regulariser"] == "huber_tv"
    # A keyword replaces its field; the config's others stay.
    recon = tj.reconstruct(scan, "fista", config=config, iterations=6, nonnegative=True)
    expected = replace(config, iterations=6, nonnegative=True)
    assert recon.info["config"] == expected
    direct, direct_info = fista_tv(
        geometry, geometry.grid, geometry.detector, scan.projections, config=expected
    )
    assert recon.info["huber_delta"] == direct_info["huber_delta"]
    np.testing.assert_allclose(np.asarray(recon.volume), np.asarray(direct), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="method 'fista' takes a FistaConfig, not a CGLSConfig"):
        tj.reconstruct(scan, "fista", config=CGLSConfig())
    with pytest.raises(ValueError, match="method 'fbp' does not take warm_start"):
        tj.reconstruct(scan, "fbp", warm_start=True)


@pytest.mark.parametrize("method", ["fista", "spdhg"])
def test_nonnegative_is_honoured(method: str) -> None:
    geometry = _geometries(16)["parallel"]
    negative = -np.asarray(tj.project(geometry, _phantom(16)))  # a negative object's data
    scan = tj.Scan(negative, geometry)
    options = {"iterations": 10, "tv_weight": 1e-4}
    free = np.asarray(tj.reconstruct(scan, method, nonnegative=False, **options).volume)
    bounded = np.asarray(tj.reconstruct(scan, method, nonnegative=True, **options).volume)
    assert free.min() < -0.1 and bounded.min() >= 0.0


def test_reconstruct_on_another_grid() -> None:
    geometry = _geometries(16)["parallel"]
    scan = tj.Scan(np.asarray(tj.project(geometry, _phantom(16))), geometry)
    region = tj.Grid(8, 8, 16, 1.0, 1.0, 1.0)
    recon = tj.reconstruct(scan, grid=region)
    assert np.asarray(recon.volume).shape == (16, 8, 8) and recon.grid == region


@pytest.mark.parametrize("kind", ["parallel", "lamino", "cone"])
def test_scans_and_reconstructions_round_trip_through_files(kind: str, tmp_path: Path) -> None:
    geometry = _geometries(8)[kind]
    if isinstance(geometry, tj.ConeGeometry):
        geometry = tj.ConeGeometry(
            geometry.grid, geometry.detector, geometry.angles,
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
    recon = tj.reconstruct(loaded, "fbp")
    tj.save(tmp_path / "recon.nxs", recon)
    with h5py.File(tmp_path / "recon.nxs") as handle:
        assert handle["entry/processing/tomojax/volume"].attrs["long_name"] == "volume"
    back = tj.load_reconstruction(tmp_path / "recon.nxs")
    np.testing.assert_allclose(back.volume, np.asarray(recon.volume), rtol=1e-6, atol=1e-6)
    assert back.method == "fbp" and back.grid == recon.grid
    assert back.info == {"config": asdict(recon.info["config"])}  # the configuration's fields
    with pytest.raises(FileNotFoundError, match="no such file"):
        tj.load(tmp_path / "missing.nxs")


@pytest.mark.parametrize("suffix", [".nxs", ".npz"])
def test_selected_loaded_scans_save_only_their_own_view_labels(tmp_path: Path, suffix: str) -> None:
    geometry = _geometries(8)["parallel"]
    source = tmp_path / "source.nxs"
    tj.save(source, tj.Scan(np.zeros((40, 12, 12), np.float32), geometry))
    with h5py.File(source, "a") as handle:
        handle["entry/instrument/detector/image_key"][...] = 0
    selected = tj.load(source).selected([2, 5, 11])
    target = tmp_path / f"selected{suffix}"
    tj.save(target, selected)
    loaded = tj.load(target)
    np.testing.assert_array_equal(loaded.angles, selected.angles)
    assert loaded.projections.shape == selected.projections.shape


def test_the_least_magnified_segment_decides_the_binning_suggestion() -> None:
    grid = tj.Grid(16, 16, 16, 1.0, 1.0, 1.0)
    fine = tj.Detector(40, 40, 0.4, 0.4)
    angles = np.linspace(0, 360, 6, endpoint=False)
    less = tj.ConeGeometry(grid, fine, angles, tj.ConeBeam(64.0, 96.0))  # magnification 1.5
    more = tj.ConeGeometry(grid, fine, angles, tj.ConeBeam(32.0, 96.0))  # magnification 3
    for segments in ((less, more), (more, less)):
        scan = tj.Scan(np.zeros((12, 40, 40), np.float32), tj.geometry.ConeSegments(segments))
        with pytest.warns(UserWarning, match=r"3\.8 times more finely"):
            tj.reconstruct(scan, "cgls", iterations=1)


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
    assert binned.detector.center == pytest.approx((0.05, -0.45))
    direct = np.asarray(tj.project(binned.geometry, volume))
    assert np.linalg.norm(np.asarray(binned.projections) - direct) / np.linalg.norm(direct) < 0.01
    assert scan.binned(1) is scan
    with pytest.raises(ValueError, match="at least 1"):
        scan.binned(0)
    arrays = (scan.projections, np.asfortranarray(scan.projections), scan.projections[:, ::-1])
    view_strided = np.ascontiguousarray(scan.projections.swapaxes(0, 1)).swapaxes(0, 1)
    for data in (*arrays, view_strided, scan.projections[:, :, ::-1]):
        for factor in range(2, 9):
            rows, cols = data.shape[1] // factor, data.shape[2] // factor
            actual = replace(scan, projections=data).binned(factor)
            expected = (
                data[:, : rows * factor, : cols * factor]
                .reshape(24, rows, factor, cols, factor)
                .mean(axis=(2, 4))
            )
            np.testing.assert_array_equal(actual.projections, expected)


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


@pytest.mark.parametrize("method", ["fbp", "fourier"])
def test_reconstruction_fills_non_cubic_zyx_output_without_replacing_it(tmp_path, method):
    grid, detector = tj.Grid(7, 5, 3, 1, 1, 1), tj.Detector(9, 3, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 24, endpoint=False))
    scan = tj.Scan(np.zeros((24, 3, 9), np.float32), geometry)
    out = np.lib.format.open_memmap(
        tmp_path / "volume.npy", mode="w+", dtype=np.float32, shape=(3, 5, 7)
    )
    result = tj.reconstruct(scan, method, out=out)
    assert result.volume is out and result.volume.flags.c_contiguous
    np.testing.assert_array_equal(result.volume, 0)


def test_single_sinogram_scan_promotes_rows_without_replacing_geometry():
    grid, detector = tj.Grid(7, 5, 1, 0.7, 0.8, 0.9), tj.Detector(9, 1, 0.7, 0.9)
    geometry = tj.ParallelGeometry(grid, detector, [0.0, 90.0])
    for data in (np.arange(18, dtype=np.float32).reshape(2, 9), jax.numpy.ones((2, 9))):
        scan = tj.Scan(data, geometry)
        assert scan.geometry is geometry and scan.projections.shape == (2, 1, 9)
        np.testing.assert_array_equal(scan.projections[:, 0], data)


def test_single_slice_operator_output_feeds_scan_and_reconstruction():
    grid, detector = tj.Grid(9, 7, 1, 1.0, 1.0, 1.0), tj.Detector(15, 1, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 24, endpoint=False))
    truth = np.zeros((7, 9), np.float32)
    truth[2:5, 2:6] = 1
    sinogram = tj.project(geometry, truth)
    assert sinogram.shape == (24, 15)
    scan = tj.Scan(sinogram, geometry)
    actual = tj.reconstruct(scan, "cgls", iterations=20).volume
    assert actual.shape == (1, 7, 9)
    assert np.linalg.norm(np.asarray(actual)[0] - truth) / np.linalg.norm(truth) < 0.06
