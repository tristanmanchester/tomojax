"""The workflow API: scans, projection, reconstruction, alignment and files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest

import tomojax as tj

if TYPE_CHECKING:
    from tomojax.alignment import AlignConfig


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
    assert np.asarray(recon.volume).shape == (8, 8, 16) and recon.grid == region


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
    recon = tj.reconstruct(loaded)
    tj.save(tmp_path / "recon.nxs", recon)
    back = tj.load_reconstruction(tmp_path / "recon.nxs")
    np.testing.assert_allclose(back.volume, np.asarray(recon.volume), rtol=1e-6, atol=1e-6)
    assert back.method == "fbp" and back.info == recon.info and back.grid == recon.grid
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
    assert full.config.quality == "reference" and full.config.freeze == ("dy",)
    with pytest.raises(ValueError, match="alignment mode must be one of"):
        alignment_plan("auto", grid)


@pytest.mark.gpu
def test_align_calibrates_the_axis_and_returns_a_corrected_scan(tmp_path: Path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    n = 32
    nominal = _geometries(n)["cone"]
    truth = tj.ConeGeometry(
        nominal.grid, nominal.detector, nominal.angles, tj.ConeBeam(96, 144, axis_offset=1.7)
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
    tj.save(tmp_path / "aligned.nxs", result.scan)
    np.testing.assert_allclose(tj.load(tmp_path / "aligned.nxs").poses, result.poses, atol=1e-6)
    assert tj.load(tmp_path / "aligned.nxs", poses=False).poses is None


def _shifted_scan(n: int = 12) -> tj.Scan:
    """A parallel scan whose every other view moved one pixel across."""
    grid = tj.Grid(n, n, n, 1.0, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, tj.Detector(n, n, 1.0, 1.0), np.linspace(0, 180, 16))
    data = np.array(tj.project(geometry, _phantom(n)))
    data[::2] = np.roll(data[::2], 1, axis=2)
    return tj.Scan(data, geometry)


def _few_iterations(scan: tj.Scan) -> AlignConfig:
    """The default pose alignment's settings, stopped after four outer iterations."""
    from tomojax.alignment.api import alignment_plan

    plan = alignment_plan("pose", scan.grid)
    return replace(plan.config, outer_iterations=4, early_stop=False)


def test_align_on_another_grid() -> None:
    scan = _shifted_scan()
    region = tj.Grid(10, 10, 8, 1.0, 1.0, 1.0)

    result = tj.align(scan, grid=region, levels=(1,), config=_few_iterations(scan))

    assert np.asarray(result.volume).shape == (10, 10, 8)
    assert result.scan.grid == region
    assert tj.reconstruct(result.scan).grid == region


def test_an_interrupted_alignment_resumes_from_its_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = _shifted_scan()
    options = {"levels": (1,), "config": _few_iterations(scan)}
    uninterrupted = tj.align(scan, **options)  # pyright: ignore[reportArgumentType]

    class Interrupted(Exception):
        pass

    from tomojax.alignment.api import load_alignment_checkpoint, save_alignment_checkpoint

    def save_then_stop(*args: object, **kwargs: object) -> None:
        save_alignment_checkpoint(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        raise Interrupted

    path = tmp_path / "align.ckpt"
    with monkeypatch.context() as patch:
        patch.setattr("tomojax.alignment.io.resume.save_alignment_checkpoint", save_then_stop)
        with pytest.raises(Interrupted):
            tj.align(scan, checkpoint=path, **options)  # pyright: ignore[reportArgumentType]
    # Stopped at the first checkpoint: the proposal stage and one solver iteration.
    progress = load_alignment_checkpoint(path).metadata
    assert progress["global_outer_iterations_completed"] == 2 and not progress["level_complete"]

    resumed = tj.align(scan, checkpoint=path, **options)  # pyright: ignore[reportArgumentType]

    np.testing.assert_allclose(resumed.poses, uninterrupted.poses, atol=1e-5)
    np.testing.assert_allclose(resumed.volume, uninterrupted.volume, atol=1e-5)
    np.testing.assert_allclose(resumed.info["loss"], uninterrupted.info["loss"], rtol=1e-5)
    assert load_alignment_checkpoint(path).metadata["run_complete"]
    finished = tj.align(scan, checkpoint=path, **options)  # pyright: ignore[reportArgumentType]
    np.testing.assert_allclose(finished.poses, uninterrupted.poses, atol=1e-5)


def test_alignment_refuses_the_checkpoint_of_another_alignment(tmp_path: Path) -> None:
    scan = _shifted_scan()
    config = _few_iterations(scan)
    path = tmp_path / "align.ckpt"
    tj.align(scan, levels=(1,), config=config, checkpoint=path)
    saved = path.read_bytes()

    with pytest.raises(ValueError, match=r"config differs in freeze .*choose another checkpoint"):
        tj.align(scan, levels=(1,), config=replace(config, freeze=("dx",)), checkpoint=path)
    other = replace(scan, projections=np.asarray(scan.projections)[:, :, ::-1])
    with pytest.raises(ValueError, match="fingerprint of the projections and geometry"):
        tj.align(other, levels=(1,), config=config, checkpoint=path)
    with pytest.raises(ValueError, match="reconstruction grid"):
        tj.align(
            scan,
            levels=(1,),
            config=config,
            grid=tj.Grid(10, 10, 8, 1.0, 1.0, 1.0),
            checkpoint=path,
        )
    assert path.read_bytes() == saved


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


def test_setup_alignment_refuses_scans_that_carry_poses() -> None:
    geometry = _geometries(12)["parallel"]
    posed = tj.align(tj.Scan(np.asarray(tj.project(geometry, _phantom(12))), geometry), levels=(1,))
    for mode in ("cor", "full"):
        with pytest.raises(ValueError, match="mode='pose'"):
            tj.align(posed.scan, mode=mode)


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
