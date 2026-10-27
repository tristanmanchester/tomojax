from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import Rings
from tomojax.geometry import ConeSegments
from tomojax.geometry.api import view_frames

N = 256


@pytest.mark.parametrize("groups", [(32,), (16, 16)])
def test_calibration_subdivision_preserves_acquisition_trajectories(groups) -> None:
    base = _geometry(180.0, 32)
    parts = tuple(
        tj.ConeGeometry(
            base.grid,
            replace(base.detector, center=(shift, 0)),
            base.angles[start : start + 16],
            tj.ConeBeam(100.0, 200.0),
        )
        for start, shift in [(0, 0.0), (16, 0.5)]
    )
    coarse = ConeSegments(parts, trajectory_lengths=groups)
    fine = ConeSegments(
        tuple(replace(part, angles=(angle,)) for part in parts for angle in part.angles),
        trajectory_lengths=groups,
    )
    np.testing.assert_array_equal(view_frames(fine), view_frames(coarse))
    data = jnp.asarray(_errors(_sinogram(180.0, 32))[:, None], jnp.float32)
    np.testing.assert_array_equal(Rings(fine).apply(data), Rings(coarse).apply(data))


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


@pytest.mark.parametrize("geometry", [None, _geometry(180.0, 180)])
def test_unsorted_fit_preserves_smooth_ring_free_data(geometry: object) -> None:
    clean = _sinogram(180.0, 180).astype(np.float32)
    fixed = np.asarray(Rings(geometry).apply(jnp.asarray(clean[:, None])))[:, 0]
    assert np.linalg.norm(fixed - clean) < 1e-5 * np.linalg.norm(clean)


@pytest.mark.parametrize("shell", [False, True])
def test_full_turn_prior_preserves_clean_scene_contrasts(shell: bool) -> None:
    u = np.arange(N) - (N - 1) / 2
    radii = np.sqrt(np.maximum(np.array([90, 75])[:, None] ** 2 - u * u, 0))
    clean = np.tile(0.8 * (radii[0] - radii[1]) / 15, (180, 1)) if shell else _sinogram(360, 180)
    fixed = np.asarray(Rings(_geometry(360, 180)).apply(jnp.asarray(clean[:, None], jnp.float32)))
    assert np.linalg.norm(fixed[:, 0] - clean) < 2e-4 * np.linalg.norm(clean)


@pytest.mark.parametrize("segmented", [False, True])
def test_irregular_view_order_and_separate_trajectories(segmented: bool) -> None:
    rng = np.random.default_rng(19)
    selected = np.sort(rng.choice(180, 100, replace=False))
    angles = selected.astype(float)
    base = _geometry(180.0, 180)
    q = _errors(_sinogram(180.0, 180)[selected]).astype(np.float32)

    def geometry(a: np.ndarray) -> object:
        if not segmented:
            return tj.ParallelGeometry(base.grid, base.detector, a)
        return ConeSegments(
            tuple(
                tj.ConeGeometry(
                    base.grid, base.detector, a, tj.ConeBeam(100.0, 200.0, source_height=h)
                )
                for h in (0.0, 5.0)
            )
        )

    order = rng.permutation(len(angles))
    if segmented:
        q = np.concatenate([q, 0.7 * q])
        order = np.concatenate([order, order + len(angles)])
    fixed = np.asarray(Rings(geometry(angles)).apply(jnp.asarray(q[:, None])))
    shuffled = Rings(geometry(angles[order[: len(angles)]])).apply(jnp.asarray(q[order, None]))
    np.testing.assert_array_equal(fixed, np.asarray(shuffled)[np.argsort(order)])


def test_unsorted_geometry_view_count_is_checked() -> None:
    with pytest.raises(ValueError, match="views"):
        Rings(_geometry(180.0, 180)).apply(jnp.zeros((179, 1, N), jnp.float32))
    with pytest.raises(ValueError, match="two views"):
        Rings(_geometry(180.0, 1))
    with pytest.raises(ValueError, match="two views"):
        Rings().apply(jnp.zeros((1, 1, N), jnp.float32))


@pytest.mark.parametrize("closing", [False, True])
def test_full_turn_fit_preserves_view_order_including_closing_phase(closing: bool) -> None:
    q = _errors(_sinogram(360, 180)).astype(np.float32)
    if closing:
        q = np.concatenate([q, q[:1]])
    angles = 2.0 * np.arange(len(q))
    base = _geometry(360, 180)
    order = np.random.default_rng(481).permutation(len(q))
    geometry = tj.ParallelGeometry(base.grid, base.detector, angles)
    permuted = tj.ParallelGeometry(base.grid, base.detector, angles[order])
    fixed = np.asarray(Rings(geometry).apply(jnp.asarray(q[:, None])))
    shuffled = np.asarray(Rings(permuted).apply(jnp.asarray(q[order, None])))
    np.testing.assert_array_equal(shuffled[np.argsort(order)], fixed)


@pytest.mark.parametrize("origin", [-40.0, 0.0, 20.0])
def test_wrapped_closing_phase_pairs_unequal_observations(origin: float) -> None:
    clean = _sinogram(360, 180)
    q = _errors(np.concatenate([clean, clean[:1]])).astype(np.float32)
    angles = origin + 2.0 * np.arange(len(q))
    base = _geometry(360, 180)
    geometry = tj.ParallelGeometry(base.grid, base.detector, angles)
    wrapped = tj.ParallelGeometry(base.grid, base.detector, angles % 360)
    assert Rings(geometry).mirror_axis == Rings(wrapped).mirror_axis == 127.5
    fixed = np.asarray(Rings(geometry).apply(jnp.asarray(q[:, None])))
    np.testing.assert_array_equal(Rings(wrapped).apply(jnp.asarray(q[:, None])), fixed)
    swapped = np.arange(len(q))
    swapped[[0, -1]] = swapped[[-1, 0]]
    changed = np.asarray(Rings(wrapped).apply(jnp.asarray(q[swapped, None])))
    np.testing.assert_array_equal(changed[swapped], fixed)


@pytest.mark.parametrize("angles", [np.arange(-40, 41, 4), np.array([-135, -45, 45, 135])])
def test_angular_phase_wrap_does_not_change_correction(angles: np.ndarray) -> None:
    clean = _sinogram(360.0, 360)[angles.astype(int) % 360]
    measured = jnp.asarray(_errors(clean)[:, None], jnp.float32)
    base = _geometry(180.0, 180)

    def corrected(a: np.ndarray, data: jnp.ndarray) -> np.ndarray:
        geometry = tj.ParallelGeometry(base.grid, base.detector, a)
        return np.asarray(Rings(geometry).apply(data))

    fixed = corrected(angles, measured)
    np.testing.assert_array_equal(fixed, corrected(angles % 360, measured))
    order = np.random.default_rng(17).permutation(len(angles))
    shuffled = corrected((angles % 360)[order], measured[order])
    np.testing.assert_array_equal(fixed, shuffled[np.argsort(order)])
