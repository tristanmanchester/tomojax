from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import tomojax as tj
from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.io import ProjectionDataset, load_dataset, save_dataset
from tomojax.recon import FDKConfig, FISTAConfig, api, fbp
from tomojax.recon.api import clear_filter_caches, default_fbp_scale
from tomojax.recon.fdk import fdk, view_weights
from tomojax.recon.filters import get_fbp_filter_np


def _tiny_geometry() -> tuple[Grid, Detector, ParallelGeometry]:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.linspace(0.0, 180.0, 4, endpoint=False, dtype=np.float32),
    )
    return grid, detector, geometry


def test_default_fbp_scale_uses_positive_view_count() -> None:
    assert default_fbp_scale(4) == pytest.approx(np.pi / 4.0)
    with pytest.raises(ValueError, match="n_views must be positive"):
        default_fbp_scale(0)


def test_clear_filter_caches_rebuilds_read_only_filters() -> None:
    clear_filter_caches()
    first = get_fbp_filter_np("ramp", 4, 1.0, "float32")
    second = get_fbp_filter_np("ramp", 4, 1.0, "float32")
    assert first is second

    clear_filter_caches()
    rebuilt = get_fbp_filter_np("ramp", 4, 1.0, "float32")

    assert rebuilt is not first
    assert not rebuilt.flags.writeable


def test_reconstruction_volume_roundtrips_through_dataset_contract(tmp_path: Path) -> None:
    path = tmp_path / "recon.nxs"
    volume = np.arange(4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4)
    dataset = ProjectionDataset(
        projections=np.ones((2, 2, 4), dtype=np.float32),
        angles=np.asarray([0.0, 90.0], dtype=np.float32),
        volume=volume,
        detector=Detector(nu=4, nv=2, du=1.0, dv=1.0),
        grid=Grid(nx=4, ny=4, nz=2, vx=1.0, vy=1.0, vz=1.0),
        sample_name="roundtrip",
    )

    save_dataset(path, dataset)
    loaded = load_dataset(path)

    assert loaded.volume is not None
    np.testing.assert_allclose(loaded.volume, volume)
    assert loaded.grid == dataset.grid


@dataclass
class _CorrectedGeometry:
    """Independent pose wrapper, also applicable around a whole segmented scan."""

    base: object
    poses: np.ndarray
    frame: str
    detector_roll: float = 0.0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def pose_matrix(self, i):
        nominal = np.asarray(self.base.pose_matrix(i), dtype=np.float64)
        update = np.eye(4)
        update[:3, :3] = Rotation.from_rotvec(self.poses[i, :3]).as_matrix()
        update[:3, 3] = self.poses[i, 3:]
        result = nominal @ update
        if self.frame == "detector":
            result[:3, 3] = nominal[:3, 3] + update[:3, 3]
        return tuple(map(tuple, result))


def _segments(counts=(4, 7)):
    grid = tj.Grid(5, 4, 3, 0.8, 1.1, 1.3)
    first = tj.Detector(12, 10, 0.9, 0.8, (0.2, -0.3))
    second = replace(first, center=(-0.7, 0.5))
    parts = (
        tj.ConeGeometry(
            grid, first, np.linspace(0, 360, counts[0], endpoint=False), tj.ConeBeam(20, 36)
        ),
        tj.ConeGeometry(
            grid,
            second,
            np.linspace(13, 373, counts[1], endpoint=False),
            tj.ConeBeam(26, 45, detector_roll=2.0),
        ),
    )
    return tj.geometry.ConeSegments(parts)


@pytest.mark.parametrize("method", ["cgls", "fista"])
def test_public_warm_start_retains_arrangements_then_applies_the_solver_constraints(
    monkeypatch, method
):
    segments = _segments()
    count = len(segments.angles)
    # Caller may supply a binned/cropped detector separately from the geometry.
    detector = tj.Detector(6, 5, 1.8, 1.6, (0.1, -0.4))
    projections = np.arange(count * 5 * 6, dtype=np.float32).reshape(count, 5, 6)
    captured = []

    def filter_part(part, grid, own_detector, data, *, config, det_grid=None):
        assert grid == segments.grid and det_grid is None
        captured.append((part, own_detector, data.copy(), config))
        return jnp.full((grid.nz, grid.ny, grid.nx), -4.0 if len(captured) == 1 else 1.0)

    def solve(_geometry, _grid, _detector, _data, *, init_x, **_kwargs):
        return init_x, {"initial_image_test": True}

    monkeypatch.setattr(importlib.import_module("tomojax.recon.fdk"), "fdk", filter_part)
    monkeypatch.setattr(
        importlib.import_module("tomojax.recon"),
        method,
        solve,
    )
    config = api.method_config(method)
    if method != "cgls":
        config = replace(config, lower_bound=0.0)
    volume, info = api.reconstruct_arrays(
        method, segments, segments.grid, detector, projections, config=config, warm_start=True
    )
    expected = -4.0
    np.testing.assert_allclose(volume, expected if method == "cgls" else 0.0, atol=1e-7)
    assert info["warm_start"] and info["initial_image_test"]
    assert len(captured) == 1
    part, own_detector, data, cfg = captured[0]
    assert part is segments and own_detector == detector and cfg == FDKConfig()
    np.testing.assert_array_equal(data, projections)


@pytest.mark.parametrize("frame", ["object", "detector"])
def test_segmented_start_matches_individual_fdk_with_all_pose_corrections(monkeypatch, frame):
    segments = _segments((12, 12))
    rng = np.random.default_rng(5)
    poses = rng.normal(0, 0.03, (24, 6)).astype(np.float32)
    geometry = _CorrectedGeometry(segments, poses, frame)
    data = rng.random((24, 10, 12)).astype(np.float32)

    # Compare like for like: CUDA FDK interpolates with half-float textures.
    def initialize(*args, config):
        return fdk(*args, config=replace(config, backend="jax"))

    monkeypatch.setattr(importlib.import_module("tomojax.recon.fdk"), "fdk", initialize)
    config = FISTAConfig(iterations=0, lipschitz=1.0, tv_weight=0.0, lower_bound=None)
    actual, _ = api.reconstruct_arrays(
        "fista", geometry, segments.grid, segments.detector, data, config=config, warm_start=True
    )
    weights = view_weights(geometry, segments.detector, len(data))
    offset = 0
    expected = np.zeros((3, 4, 5), np.float32)
    for segment in segments.segments:
        n = len(segment.angles)
        own = _CorrectedGeometry(segment, poses[offset : offset + n], frame)
        # Isolate the poses, not a different acquisition: each panel shares
        # redundancy with the full scan's conjugate panels. Separate FDKs
        # therefore need the same angular measure, not a per-turn average.
        local = view_weights(own, segment.detector, n)
        measured = data[offset : offset + n] * (weights[offset : offset + n] / local)[:, None, :]
        image = fdk(
            own,
            segments.grid,
            segment.detector,
            measured,
            config=FDKConfig(backend="jax"),
        )
        expected += np.asarray(image)
        offset += n
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    ("case", "views", "message"),
    [
        ("short_arc", 8, "FDK short scans need at least"),
        ("rolled", 11, "canonical detector grid"),
        ("wrong_views", 10, "warm_start projections"),
    ],
)
def test_segmented_warm_start_rejects_invalid_scans(case, views, message):
    geometry = _segments()
    if case == "short_arc":
        geometry = tj.geometry.ConeSegments(
            tuple(replace(segment, angles=[0, 30, 60, 90]) for segment in geometry.segments)
        )
    elif case == "rolled":
        geometry = _CorrectedGeometry(geometry, np.zeros((11, 6)), "object", detector_roll=1.0)
    with pytest.raises(ValueError, match=message):
        api.reconstruct_arrays(
            "fista",
            geometry,
            geometry.grid,
            geometry.detector,
            np.zeros((views, 10, 12), np.float32),
            warm_start=True,
        )


def test_fbp_halves_view_batches_after_a_cufft_scratch_failure(monkeypatch):
    module = importlib.import_module("tomojax.recon.fbp")
    grid, detector, geometry = _tiny_geometry()
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs["batch_size"])
        raise jax.errors.JaxRuntimeError(
            "INTERNAL: RET_CHECK failure (fft_thunk.cc:201) fft_plan != nullptr "
            "Failed to create cuFFT batched plan with scratch allocator"
        )

    monkeypatch.setattr(module, "_run_fbp_streamed", fail)
    with pytest.raises(jax.errors.JaxRuntimeError, match="Failed to create cuFFT"):
        fbp(geometry, grid, detector, jnp.ones((4, 4, 4)))
    assert calls == [4, 2, 1]


def test_quicklook_transfers_only_the_central_slice_of_a_device_volume():
    from tomojax.recon.quicklook import extract_central_slice

    class Volume:
        shape, ndim = (5, 6, 7), 3

        def __init__(self):
            self.selections = []

        def __array__(self, dtype=None, copy=None):
            raise AssertionError("downloaded the whole volume")

        def __getitem__(self, selection):
            self.selections.append(selection)
            return np.arange(42, dtype=np.float32).reshape(6, 7)

    volume = Volume()
    assert extract_central_slice(volume).shape == (6, 7)
    assert volume.selections == [2]


@pytest.mark.numerical
@pytest.mark.parametrize("method", ["mlem", "sirt"])
def test_iterative_axial_slabs_bound_the_warm_initializer(monkeypatch, method):
    grid = Grid(8, 9, 5, 0.8, 1.1, 1.3, vol_origin=(1.3, -2.7, 2.5 - 4 * 1.3 / 2))
    detector = Detector(12, 5, 0.8, 1.3, (0.2, 2.5))
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 180, 31, endpoint=False))
    truth = np.random.default_rng(17).uniform(size=(5, 9, 8)).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, truth)), geometry)
    support = np.ones((5, 9, 8), bool)
    support[:, :, 0] = False
    settings = (
        {"support": support[:1]}
        if method == "mlem"
        else {
            "support": support,
            "lower_bound": 0,
            "upper_bound": 2,
        }
    )
    cfg = api.method_config(method, iterations=2, subsets=3, **settings)
    expected = np.asarray(tj.reconstruct(scan, method, config=cfg, warm_start=True).volume)
    per_row = 4 * (8 * 9 * 6 + 31 * 12 * 2)
    monkeypatch.setattr(tj.backends, "device_free_memory_bytes", lambda: 5 * per_row)
    actual = tj.reconstruct(scan, method, config=cfg, warm_start=True)
    assert actual.info["host_slabs"]
    assert isinstance(actual.volume, np.ndarray)
    np.testing.assert_allclose(actual.volume, expected, rtol=2e-5, atol=2e-6)
