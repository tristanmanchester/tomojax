"""Automatic raw-frame batches account for padded FFT arrays before reading data."""

from __future__ import annotations

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Paganin, Zingers


class ConstantViews:
    """Lazy per-view constant fields, logging which data the correction reads."""

    dtype = np.dtype(np.float32)

    def __init__(self, rows: int, cols: int) -> None:
        self.shape = (65, rows, cols)
        self.levels = np.linspace(400.0, 800.0, self.shape[0], dtype=np.float32)
        self.requests: list[int] = []

    def __getitem__(self, views: slice) -> np.ndarray:
        levels = self.levels[views]
        self.requests.append(len(levels))
        return np.broadcast_to(levels[:, None, None], (len(levels), *self.shape[1:]))


@pytest.mark.parametrize("shape", [(256, 256), (128, 512)])
@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("batch", [None, 11])
def test_phase_retrieval_limits_automatic_reads_and_preserves_explicit_batches(
    shape: tuple[int, int], phase: bool, batch: int | None
) -> None:
    rows, cols = shape
    counts = ConstantViews(rows, cols)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), np.arange(65)
    )
    steps = [Paganin(300.0, 0.001, 20.0, 1e-6)] if phase else []
    actual = tj.Frames(counts, geometry, white_level=1000.0).corrected(*steps, batch_views=batch)

    if batch is not None:
        assert max(counts.requests) == batch
    elif phase:
        # The small unpadded input fits one batch; its padded FFT arrays do not.
        assert max(counts.requests) < counts.shape[0]
    else:
        assert counts.requests == [counts.shape[0]]
    assert sum(counts.requests) == counts.shape[0]
    expected = -np.log(counts.levels / 1000.0)[:, None, None]
    np.testing.assert_allclose(
        actual.projections, np.broadcast_to(expected, counts.shape), rtol=2e-6, atol=2e-7
    )
    np.testing.assert_array_equal(actual.angles, np.arange(65))
    assert next(record for record in actual.corrections if record.name == "log").found == {
        "nonpositive_transmission": 0,
        "nonfinite_set_to_zero": 0,
    }


@pytest.mark.numerical
@pytest.mark.parametrize("shape", [(512, 512), (256, 1024)])
@pytest.mark.parametrize("size", [3, 5, 7])
@pytest.mark.parametrize("batch", [None, 11])
def test_zinger_workspace_bounds_automatic_reads_without_changing_pixels(shape, size, batch):
    rows, cols = shape
    counts = ConstantViews(rows, cols)
    geometry = tj.ParallelGeometry(
        tj.Grid(8, 8, 2, 1.0, 1.0, 1.0), tj.Detector(cols, rows, 1.0, 1.0), np.arange(65)
    )
    actual = tj.Frames(counts, geometry, white_level=1000.0).corrected(
        Zingers(size=size), batch_views=batch
    )
    if batch is None:
        # The unpadded inputs fit together; the padded sorting scratch does not.
        assert max(counts.requests) < counts.shape[0]
    else:
        assert max(counts.requests) == batch
    assert sum(counts.requests) == counts.shape[0]
    expected = -np.log(counts.levels / 1000.0)[:, None, None]
    np.testing.assert_allclose(
        actual.projections, np.broadcast_to(expected, counts.shape), rtol=2e-6, atol=2e-7
    )
    np.testing.assert_array_equal(actual.angles, np.arange(65))
    assert actual.corrections[1].name == "zingers"
    assert actual.corrections[1].settings == {"threshold": 0.1, "size": size}


@pytest.mark.numerical
def test_zinger_batch_sizes_preserve_independent_gaussian_reconstruction():
    views, nu, nv = 65, 65, 17
    du, dv, mu, sigma, sigma_z = 0.8, 1.2, 0.025, 3.0, 2.0
    x0, y0, z0 = 1.2, -1.7, 0.9
    angles = np.linspace(0.0, 180.0, views, endpoint=False, dtype=np.float32)
    theta = np.deg2rad(angles.astype(float))
    u = (np.arange(nu) - (nu - 1) / 2) * du
    v = (np.arange(nv) - (nv - 1) / 2) * dv
    shift = x0 * np.cos(theta) - y0 * np.sin(theta)
    p = (
        mu
        * np.sqrt(2 * np.pi)
        * sigma
        * np.exp(-0.5 * ((u[None, :] - shift[:, None]) / sigma) ** 2)[:, None, :]
        * np.exp(-0.5 * ((v - z0) / sigma_z) ** 2)[None, :, None]
    )
    counts = (1000 * np.exp(-p)).astype(np.float32)
    counts[np.arange(views), np.arange(views) % nv, 3 + np.arange(views) % 59] += 500
    grid = tj.Grid(49, 49, nv, du, du, dv)
    geometry = tj.ParallelGeometry(grid, tj.Detector(nu, nv, du, dv), angles)
    frames = tj.Frames(counts, geometry, white_level=1000.0)
    x = (np.arange(49) - 24) * du
    truth = mu * np.exp(
        -0.5 * ((x[:, None, None] - x0) ** 2 + (x[None, :, None] - y0) ** 2) / sigma**2
        - 0.5 * ((v[None, None, :] - z0) / sigma_z) ** 2
    )
    full = frames.corrected(Zingers())
    chunked = frames.corrected(Zingers(), batch_views=7)
    np.testing.assert_array_equal(full.projections, chunked.projections)
    a = np.asarray(tj.reconstruct(full, "fbp").volume)
    b = np.asarray(tj.reconstruct(chunked, "fbp").volume)
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a - truth) / np.linalg.norm(truth) < 0.05
