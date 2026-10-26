"""Owned raw-to-reconstruction slabs preserve their physical geometry and provenance."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import RejectViews, Stripes
from tomojax.geometry import ConeSegments, grid_volume_origin
from tomojax.recon import FBPConfig, FourierConfig, MLEMConfig, SIRTConfig, api

CASES = [
    ("fourier", FourierConfig(backend="numpy", slices_per_batch=4)),
    ("fourier", FourierConfig(backend="auto")),
    ("fbp", FBPConfig(backend="jax")),
    pytest.param(
        "fourier", FourierConfig(backend="cupy", slices_per_batch=4), marks=pytest.mark.gpu
    ),
    pytest.param("fbp", FBPConfig(backend="cuda"), marks=pytest.mark.gpu),
]


@pytest.mark.parametrize(("method", "cfg"), CASES)
def test_raw_slabs_match_corrected_scan_and_remain_reconstructible(tmp_path, method, cfg):
    assert "reconstruct_slabs" in api.__all__
    rng = np.random.default_rng(27)
    grid = tj.Grid(17, 19, 11, 0.9, 1.1, 0.6, vol_origin=(-7, -8, -3))
    detector = tj.Detector(23, 9, 1, 0.8, center=(0.7, 0.3))
    geometry = tj.ParallelGeometry(
        replace(grid, vol_origin=(-8, -9, -4)), detector, np.linspace(180, 0, 25)
    )
    raw = rng.uniform(200, 900, (25, 9, 23)).astype(">f4")
    flat = rng.normal(1000, 2, (4, 9, 23)).astype(np.float32)
    dark = np.ones((2, 9, 23), np.float32)
    path = tmp_path / "counts.hdf"
    with h5py.File(path, "w") as handle:
        handle["data"] = raw
    frames = tj.load_frames(path, geometry=geometry, angles=geometry.angles, flats=flat, darks=dark)
    frames = replace(frames, flat_positions=np.array([0, 0, 25, 25])).cropped(
        slice(1, 8), slice(None)
    )
    step = Stripes(5, varying=True)
    expected = tj.reconstruct(frames.corrected(step), method, grid=grid, config=cfg).volume
    with tj.reconstruct_slabs(frames, method, grid=grid, corrections=[step], config=cfg) as slabs:
        collected = list(slabs)
    actual = np.concatenate([item.volume for _, item in collected], axis=0)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    for section, item in collected:
        origin = grid_volume_origin(item.grid)
        np.testing.assert_allclose(origin, (-7, -8, -3 + section.start * 0.6))
        rows = slice(*item.info["source_rows"])
        independent = frames.cropped(rows, slice(None)).corrected(step)
        np.testing.assert_allclose(item.scan.projections, independent.projections, rtol=0, atol=0)
        assert item.scan.corrections == independent.corrections
        again = tj.reconstruct(item.scan, method, config=cfg)
        np.testing.assert_allclose(again.volume, item.volume, rtol=3e-5, atol=2e-6)
    tj.save(tmp_path / "slab.npz", collected[0][1])
    reloaded = tj.load_reconstruction(tmp_path / "slab.npz")
    np.testing.assert_array_equal(reloaded.volume, collected[0][1].volume)


@pytest.mark.parametrize(("method", "cfg"), CASES)
def test_slab_context_closes_on_early_exit_and_rejects_view_selection(method, cfg):
    grid = tj.Grid(17, 15, 9, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(23, 9, 1, 1), np.linspace(0, 180, 25))
    frames = tj.Frames(np.full((25, 9, 23), 500, ">f4"), geometry, white_level=1000)
    with tj.reconstruct_slabs(frames, method, config=cfg) as slabs:
        _, retained = next(slabs)
        saved = retained.volume.copy()
    with pytest.raises(StopIteration):
        next(slabs)
    np.testing.assert_array_equal(retained.volume, saved)
    np.testing.assert_allclose(retained.scan.projections, np.log(2), rtol=1e-6)
    with (
        pytest.raises(ValueError, match="without view selection"),
        tj.reconstruct_slabs(frames, method, corrections=[RejectViews()], config=cfg),
    ):
        pass


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("method", ["sirt", "mlem"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_iterative_slabs_preserve_updates_masks_and_ownership(method, backend, warm_start):
    grid = tj.Grid(13, 15, 11, 0.8, 1.1, 0.5, vol_center=(0.3, -0.5, 2.0))
    detector = tj.Detector(17, 11, 1, 0.5, center=(0.6, 2.0))
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 13))
    raw = np.random.default_rng(3).uniform(100, 800, (13, 11, 17)).astype(np.float32)
    frames = tj.Frames(raw, geometry, white_level=1000)
    support = np.ones((11, 15, 13), bool)
    support[::2, :, 0] = False
    settings = dict(iterations=2, subsets=3, support=support, backend=backend)
    cfg = (
        SIRTConfig(lower_bound=0, upper_bound=0.5, **settings)
        if method == "sirt"
        else MLEMConfig(**settings)
    )
    expected = tj.reconstruct(frames.corrected(), method, config=cfg, warm_start=warm_start)
    with tj.reconstruct_slabs(frames, method, config=cfg, warm_start=warm_start) as slabs:
        collected = list(slabs)
    actual = np.concatenate([item.volume for _, item in collected], axis=0)
    np.testing.assert_allclose(actual, expected.volume, rtol=2e-5, atol=1e-6)
    assert len(collected) > 1
    assert np.count_nonzero(actual[~support]) == 0
    for section, item in collected:
        np.testing.assert_array_equal(
            item.scan.projections, frames.corrected().projections[:, section]
        )
        np.testing.assert_allclose(grid_volume_origin(item.grid)[2], -0.5 + section.start * 0.5)
        assert item.info["warm_start"] == warm_start
        assert item.info["config"].support.shape == item.volume.shape
    with tj.reconstruct_slabs(frames, method, config=cfg) as slabs:
        _, retained = next(slabs)
        saved = retained.volume.copy()
    with pytest.raises(StopIteration):
        next(slabs)
    np.testing.assert_array_equal(retained.volume, saved)


def test_iterative_slabs_refuse_coupled_planes():
    grid = tj.Grid(9, 9, 3, 1, 1, 1)
    geometry = tj.ParallelGeometry(grid, tj.Detector(13, 3, 1, 1), np.linspace(0, 180, 9))
    frames = tj.Frames(np.full((9, 3, 13), 500, np.float32), geometry, white_level=1000)
    for cfg, target in [
        (SIRTConfig(projector_model="ray"), grid),
        (SIRTConfig(), replace(grid, vz=0.5)),
    ]:
        with (
            tj.reconstruct_slabs(frames, "sirt", config=cfg, grid=target) as slabs,
            pytest.raises(ValueError, match="independent axial Joseph planes"),
        ):
            next(slabs)


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cone_raw_slabs_preserve_acquisition_weights_and_owned_rows(backend):
    class Windowed:
        def __init__(self, data, requests):
            self.data, self.requests = data, requests
            self.shape, self.dtype = data.shape, data.dtype

        def window(self, rows, cols):
            return Windowed(self.data[:, rows, cols], self.requests)

        def __getitem__(self, views):
            out = self.data[views]
            self.requests.append(out.shape)
            return out

        def __array__(self, *args, **kwargs):
            raise AssertionError("Raw input must be read in bounded windows")

    grid = tj.Grid(7, 9, 37, 0.8, 1.0, 0.3, vol_center=(0.2, -0.1, 0.1))
    detector = tj.Detector(23, 65, 1.0, 0.5, (0.7, -0.3))
    geometry = tj.ConeGeometry(
        grid,
        detector,
        np.linspace(0, 360, 16, endpoint=False),
        tj.ConeBeam(30, 45, detector_roll_deg=1.5),
    )
    geometry = ConeSegments(
        (
            replace(geometry, angles=geometry.angles[:8]),
            replace(
                geometry, angles=geometry.angles[8:], detector=replace(detector, center=(1.1, 0.2))
            ),
        )
    )
    exposures, flat_exposures = np.linspace(0.7, 1.6, 16), np.array([0.3, 1.1, 0.5])
    raw = np.random.default_rng(91).uniform(200, 900, (16, 65, 23))
    raw = (11 + raw * exposures[:, None, None]).astype(">f4")
    flats = np.broadcast_to(
        (11 + np.array([1000, 1100, 900]) * flat_exposures)[:, None, None], (3, 65, 23)
    ).copy()
    eager = tj.Frames(
        raw,
        geometry,
        flats=flats,
        darks=np.full((1, 65, 23), 11),
        flat_positions=np.array([0, 7, 16]),
        view_positions=np.linspace(0.2, 15.8, 16),
        exposure_times=exposures,
        flat_exposure_times=flat_exposures,
    )
    requests = []
    frames = replace(eager, counts=Windowed(raw, requests))
    cfg, step = FBPConfig(backend=backend), Stripes(5, varying=True)
    expected = tj.reconstruct(eager.corrected(step), "fbp", config=cfg).volume
    with tj.reconstruct_slabs(frames, "fbp", config=cfg, corrections=[step]) as slabs:
        collected = list(slabs)
    assert len(collected) == 2
    assert max(shape[1] for shape in requests) < detector.nv
    actual = np.concatenate([r.volume for _, r in collected], axis=0)
    if backend == "jax":
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    else:
        # CUDA normalizes half textures to each row window, as in fdk_host.
        assert np.linalg.norm(actual - expected) < 1e-3 * np.linalg.norm(expected)
    for section, recon in collected:
        rows = slice(*recon.info["source_rows"])
        independent = eager.cropped(rows, slice(None)).corrected(step)
        assert isinstance(recon.scan.projections, np.ndarray)
        np.testing.assert_allclose(recon.scan.projections, independent.projections, atol=1e-6)
        assert recon.scan.corrections == independent.corrections
        np.testing.assert_allclose(
            grid_volume_origin(recon.grid)[2],
            grid_volume_origin(grid)[2] + section.start * grid.vz,
        )


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cone_raw_slab_early_close_keeps_completed_result(backend):
    grid, detector = tj.Grid(5, 5, 35, 1, 1, 0.2), tj.Detector(9, 21, 1, 1)
    geometry = tj.ConeGeometry(
        grid, detector, np.linspace(0, 360, 8, endpoint=False), tj.ConeBeam(30, 45)
    )
    frames = tj.Frames(np.full((8, 21, 9), 500, np.uint16), geometry, white_level=1000)
    with tj.reconstruct_slabs(frames, "fbp", config=FBPConfig(backend=backend)) as slabs:
        _, retained = next(slabs)
        saved = retained.volume.copy()
    with pytest.raises(StopIteration):
        next(slabs)
    np.testing.assert_array_equal(retained.volume, saved)
    np.testing.assert_allclose(retained.scan.projections, np.log(2), atol=1e-6)
