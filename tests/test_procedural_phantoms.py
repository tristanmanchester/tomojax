"""Procedural documentation phantoms build quickly, deterministically and in physical units."""

from collections.abc import Callable
import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))

Builder = Callable[..., tuple[np.ndarray, dict[str, Any]]]
NAMES = ["porous_rock", "battery", "lattice_part"]
pytestmark = pytest.mark.slow


def _builder(name: str) -> Builder:
    return getattr(importlib.import_module(name), name)


@pytest.mark.parametrize("name", NAMES)
def test_phantom_is_deterministic_bounded_and_scale_free(name: str) -> None:
    build = _builder(name)
    volume, meta = build(24, seed=3)
    again, _ = build(24, seed=3)
    coarse, coarse_meta = build(32, seed=3)

    assert volume.shape == (24, 24, 24)
    assert volume.dtype == np.float32 and volume.flags.c_contiguous
    np.testing.assert_array_equal(volume, again)
    assert meta["seed"] == 3
    assert meta["voxel_mm"] == pytest.approx(meta["fov_mm"] / 24)
    mu = np.array(list(meta["materials"].values()))
    assert np.all(np.isfinite(volume))
    assert volume.min() >= 0
    assert volume.max() <= mu.max() * (1 + 1e-5)
    # Several materials appear, so the volume is not a single-valued mask.
    assert np.count_nonzero(np.histogram(volume[volume > 0], bins=64)[0]) > 8
    # The object is fixed in mm: a coarser grid integrates to the same total.
    total = volume.sum(dtype=np.float64) * meta["voxel_mm"] ** 3
    coarse_total = coarse.sum(dtype=np.float64) * coarse_meta["voxel_mm"] ** 3
    assert coarse_total == pytest.approx(total, rel=0.03)


def test_procedural_voxel_coordinates_are_zyx():
    from _procedural import Grid, sample_where, voxelise

    def labels(x, y, z):
        return (x > 0).astype(int) + 2 * (y > 0) + 4 * (z > 0), np.ones_like(x)

    grid = Grid(4, 4)
    volume = voxelise(labels, grid, np.arange(8), fine=1)
    z, y, x = np.indices((4, 4, 4))
    np.testing.assert_array_equal(volume, (x >= 2) + 2 * (y >= 2) + 4 * (z >= 2))
    mask = (x == 1) | (z == 3)
    values, _ = sample_where(labels, grid, np.arange(8), mask, fine=1)
    np.testing.assert_array_equal(values, volume[mask])


def test_mesh_occupancy_uses_non_cubic_zyx_shape():
    from mesh_voxels import occupancy

    vertices = np.array([[x, y, z] for z in (-0.4, 0.4) for y in (-0.8, 0.8) for x in (-1.2, 1.2)])
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 7, 5],
            [4, 6, 7],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    volume = occupancy(vertices[faces], (3, 5, 7), (0.5, 0.5, 0.5), (-1.5, -1, -0.5))
    assert volume.shape == (3, 5, 7) and volume.flags.c_contiguous
    assert volume[1, 2, 3] == 1 and not volume[:, :, 0].any()


def test_projection_benchmark_phantom_support_and_slice_are_zyx(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    import fista_projection_benchmark as benchmark

    shape = (7, 9, 13)
    rng = np.random.default_rng(81)
    assert benchmark.random_phantom(shape, rng).shape == shape
    support = benchmark.random_support(shape, rng)
    assert support.shape == shape and support[3, 4, 6] == 1
    volume = np.zeros(shape, np.float32)
    volume[2] = 1
    assert benchmark.most_informative_z_slice(volume) == 2
