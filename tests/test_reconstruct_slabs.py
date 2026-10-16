"""Owned raw-to-reconstruction slabs preserve their physical geometry and provenance."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import RejectViews, Stripes
from tomojax.geometry import grid_volume_origin
from tomojax.recon import FBPConfig, FourierConfig, api

CASES = [
    ("fourier", FourierConfig(backend="numpy", slices_per_batch=4)),
    ("fourier", FourierConfig(backend="auto")),
    ("fbp", FBPConfig(backprojector="jax")),
    pytest.param(
        "fourier", FourierConfig(backend="cupy", slices_per_batch=4), marks=pytest.mark.gpu
    ),
    pytest.param("fbp", FBPConfig(backprojector="pallas"), marks=pytest.mark.gpu),
]


@pytest.mark.parametrize(("method", "cfg"), CASES)
def test_raw_slabs_match_corrected_scan_and_remain_reconstructible(tmp_path, method, cfg):
    assert f"{method}_slabs" in api.__all__
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
    actual = np.concatenate([item.volume for _, item in collected], axis=2)
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
