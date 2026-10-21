import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Rings

N = 256


def _sinogram(turn: float, views: int, offset: float = 0.0) -> np.ndarray:
    """Line integrals of a few Gaussian blobs, (views, columns), axis at column 127.5 - offset."""
    angles = np.deg2rad(np.arange(views) * turn / views)[:, None]
    u = np.arange(N) - (N - 1) / 2 + offset
    q = np.zeros((views, N))
    for x, y, r, h in ((20, 5, 25, 0.6), (-45, 30, 12, 0.4), (10, -60, 18, 0.5), (70, 40, 9, 0.8)):
        q += h * np.exp(-(((u - x * np.cos(angles) - y * np.sin(angles)) / r) ** 2))
    return q


def _errors(q: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.02, N)
    b = rng.normal(0, 0.05, N)
    return q + a + b * -np.expm1(-q) + rng.normal(0, 0.003, q.shape)


def _rms_offset(q: np.ndarray, truth: np.ndarray) -> float:
    """RMS over columns of the error's view-mean, after removing its smooth trend."""
    e = (q - truth).mean(0)
    return float(np.sqrt(np.mean((e - np.convolve(e, np.ones(31) / 31, "same"))[20:-20] ** 2)))


def _geometry(turn: float, views: int, offset: float = 0.0) -> tj.ParallelGeometry:
    detector = tj.Detector(N, 1, 1.0, 1.0, center=(offset, 0.0))
    angles = [i * turn / views for i in range(views)]
    return tj.ParallelGeometry(tj.Grid(N, N, 1, 1.0, 1.0, 1.0), detector, angles)


@pytest.mark.parametrize("turn", [180.0, 360.0])
def test_removes_column_errors(turn: float) -> None:
    clean = _sinogram(turn, 180)
    measured = _errors(clean)
    step = Rings(_geometry(turn, 180))
    assert step.mirror_axis == (None if turn == 180.0 else (N - 1) / 2)
    fixed = np.asarray(step.apply(jnp.asarray(measured[:, None], jnp.float32)))[:, 0]
    assert _rms_offset(fixed, clean) < 0.4 * _rms_offset(measured, clean)


def test_detector_reversal_and_flat_data() -> None:
    q = jnp.asarray(_errors(_sinogram(360.0, 180, offset=0.5))[:, None], jnp.float32)
    a = Rings(_geometry(360.0, 180, offset=0.5)).apply(q)
    b = Rings(_geometry(360.0, 180, offset=-0.5)).apply(q[:, :, ::-1])[:, :, ::-1]
    assert float(jnp.abs(a - b).max()) < 1e-5
    flat = jnp.full((90, 2, N), 0.3, jnp.float32)
    assert float(jnp.abs(Rings().apply(flat) - flat).max()) < 1e-6


def test_mirror_only_on_a_stationary_parallel_full_turn() -> None:
    assert Rings(_geometry(360.0, 180, offset=0.3)).mirror_axis == 127.0
    assert Rings(_geometry(362.0, 181)).mirror_axis == 127.5  # last view repeats the first
    assert Rings(_geometry(540.0, 270)).mirror_axis is None
    assert Rings(_geometry(360.0, 180)) != Rings(_geometry(360.0, 180, offset=1.0))
    assert Rings(_geometry(360.0, 180)).record().settings == {"mirror_axis": 127.5}
    with pytest.raises(ValueError, match="detector columns"):
        Rings(_geometry(360.0, 180)).apply(jnp.zeros((180, 1, N // 2), jnp.float32))


def test_scan_corrected_records_the_step() -> None:
    detector = tj.Detector(N, 1, 1.0, 1.0)
    geometry = tj.ParallelGeometry(
        tj.Grid(N, N, 1, 1.0, 1.0, 1.0), detector, [float(a) for a in range(180)]
    )
    scan = tj.Scan(_errors(_sinogram(180.0, 180))[:, None].astype(np.float32), geometry)
    out = scan.corrected(Rings())
    assert [c.name for c in out.corrections][-1] == "rings"
