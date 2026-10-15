"""Owned detector-row FBP agrees with a full physical reconstruction."""

from contextlib import closing
from dataclasses import replace

import jax
import numpy as np
import pytest

from tomojax.geometry import Detector, Grid, ParallelGeometry
from tomojax.recon.api import FBPConfig, fbp, fbp_slabs

BACKENDS = ["jax", pytest.param("pallas", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("kind", ["integer", "strided", "fractional", "outside", "one_row"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_owned_fbp_slabs_keep_measured_rows_and_physical_geometry(kind, backend):
    grid = Grid(9, 7, 19, 0.8, 1.2, 0.7, vol_center=(1.2, -0.4, 0.31))
    detector = Detector(9, 17, 0.9, 1.0, (0.23, -0.19))
    if kind == "integer":
        grid = replace(grid, vz=1.0, vol_center=(1.2, -0.4, 0.0))
        detector = replace(detector, nv=19, center=(0.23, 0.0))
    elif kind == "strided":
        grid = replace(grid, vz=2.0, vol_center=(1.2, -0.4, 0.0))
        detector = replace(detector, nv=39, center=(0.23, 0.0))
    elif kind == "outside":
        grid = replace(grid, vz=3.0, vol_origin=(-1.7, -2.1, -25.0))
    elif kind == "one_row":
        detector = replace(detector, nv=1)
    geometry = ParallelGeometry(grid, detector, np.linspace(0, 179, 73))
    data = np.random.default_rng(132).normal(size=(73, detector.nv, detector.nu)).astype(np.float32)
    cfg = FBPConfig(backprojector=backend, filter="hann", scale=0.41)
    expected = np.asarray(fbp(geometry, grid, detector, data, config=cfg))

    def read(first, stop):
        return jax.device_put(data[:, first:stop].copy()), (first, stop)

    with closing(fbp_slabs(geometry, grid, detector, read, config=cfg)) as output:
        collected = list(output)
    actual = np.concatenate([item.volume for item in collected], axis=2)
    np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=5e-5)
    assert collected[-1].stop == grid.nz
    assert max(item.stop - item.start for item in collected) <= 8
    for item in collected:
        assert item.context == (item.rows.start, item.rows.stop)
        np.testing.assert_array_equal(item.projections, data[:, item.rows])
        assert item.volume.dtype == np.float32 and item.volume.flags.f_contiguous


@pytest.mark.parametrize("backend", BACKENDS)
def test_owned_fbp_slabs_survive_early_context_closure(backend):
    grid = Grid(11, 9, 23, 1, 1, 1)
    detector = Detector(17, 23, 1, 1)
    geometry = ParallelGeometry(grid, detector, [0, 41, 97, 154])

    def read(first, stop):
        return jax.numpy.full((4, stop - first, 17), first + 1, dtype="float32"), None

    with closing(
        fbp_slabs(geometry, grid, detector, read, config=FBPConfig(backprojector=backend))
    ) as output:
        first = next(output)
        saved = first.volume.copy()
        next(output)
    with pytest.raises(StopIteration):
        next(output)
    np.testing.assert_array_equal(first.volume, saved)
    np.testing.assert_array_equal(
        first.projections, np.ones((4, first.rows.stop - first.rows.start, 17), np.float32)
    )
