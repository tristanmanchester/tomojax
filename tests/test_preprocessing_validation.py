"""Preprocessing refuses ambiguous inputs before reading frames or changing geometry."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening
from tomojax.io import ProjectionDataset, save_dataset

from ._helpers import tiny_detector, tiny_grid

pytestmark = pytest.mark.surface


class _UnreadFrames:
    shape = (6, 4, 4)
    dtype = np.dtype(np.float32)

    def __getitem__(self, key):
        raise AssertionError("invalid settings must fail before reading any counts")


def _scan(*, segmented=False):
    grid, detector = tiny_grid(nx=4, ny=4, nz=4), tiny_detector(nu=4, nv=4)
    data = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    if not segmented:
        return tj.Scan(data, tj.ParallelGeometry(grid, detector, np.arange(6) * 30.0))
    return tj.Scan.combine(
        [
            tj.Scan(
                data[3 * k : 3 * k + 3],
                tj.ConeGeometry(grid, detector, [10.0, 50.0, 100.0], tj.ConeBeam(20, 30 + k)),
            )
            for k in range(2)
        ]
    )


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        slice(None, None, -1),
        slice(4, 0, -2),
        slice(2, 2),
        [],
        np.zeros(6, dtype=bool),
        [0.9, 3.9],
        [0.0, 3.0],
        [[0, 3]],
        ["0", "3"],
        [0, 0],
        [3, 0],
        [-1, 3],
        [0, 6],
        np.asarray([0, 2**64 - 1], dtype=np.uint64),
    ],
)
def test_selections_need_nonempty_increasing_integer_indices(segmented, selection):
    scan = _scan(segmented=segmented)
    frames = tj.Frames(_UnreadFrames(), scan.geometry, white_level=8.0)
    for source in (scan, frames):
        with pytest.raises(ValueError, match="views"):
            source.selected(selection)


@pytest.mark.parametrize("selection", [slice(1, None, 2), [1, 3, 5], np.arange(6) % 2 == 1])
def test_valid_selections_keep_the_same_views_and_orbits(selection):
    scan = _scan(segmented=True)
    selected = scan.selected(selection)
    kept = np.asarray([1, 3, 5])
    np.testing.assert_array_equal(selected.projections, scan.projections[kept])
    np.testing.assert_array_equal(selected.angles, scan.angles[kept])
    for new, old in enumerate(kept):
        np.testing.assert_allclose(
            selected.geometry.pose_for_view(new), scan.geometry.pose_for_view(old)
        )
        before, after = scan.geometry.rays_for_view(old), selected.geometry.rays_for_view(new)
        for ray_before, ray_after in zip(before, after, strict=True):
            np.testing.assert_allclose(ray_after(1, 2), ray_before(1, 2))


@pytest.mark.parametrize("batch", [0, -1, 1.5, 2.0, True, np.nan, np.inf, "2"])
def test_invalid_batch_sizes_fail_before_reading_counts(batch):
    scan = _scan()
    frames = tj.Frames(_UnreadFrames(), scan.geometry, white_level=8.0)
    with pytest.raises(ValueError, match="batch_views"):
        frames.corrected(batch_views=batch)
    with pytest.raises(ValueError, match="batch_views"):
        scan.corrected(BeamHardening(), batch_views=batch)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_epsilon_must_be_positive_and_finite_before_reading_counts(epsilon):
    frames = tj.Frames(_UnreadFrames(), _scan().geometry, white_level=8.0)
    with pytest.raises(ValueError, match="epsilon"):
        frames.corrected(epsilon=epsilon)


@pytest.mark.parametrize("level", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_white_level_must_be_positive_and_finite_before_reading_counts(level):
    frames = tj.Frames(_UnreadFrames(), _scan().geometry, white_level=level)
    with pytest.raises(ValueError, match="white_level"):
        frames.corrected()


@pytest.mark.parametrize("positions", [np.nan, [0.0], [[0.0, 6.0]], [0.0, np.nan], [0.0, np.inf]])
def test_flat_positions_need_one_finite_position_per_flat(positions):
    frames = tj.Frames(
        _UnreadFrames(),
        _scan().geometry,
        flats=np.full((2, 4, 4), 8.0, np.float32),
        flat_positions=positions,
    )
    with pytest.raises(ValueError, match="flat_positions"):
        frames.corrected()


@pytest.mark.parametrize("positions", [np.full(6, np.nan), np.full(6, np.inf)])
def test_view_positions_must_be_finite_before_reading_counts(positions):
    with pytest.raises(ValueError, match="view_positions"):
        tj.Frames(
            _UnreadFrames(), _scan().geometry, white_level=8.0, view_positions=positions
        ).corrected()


@pytest.mark.parametrize("field", ["flats", "darks"])
def test_empty_calibration_stacks_are_refused_before_reading_counts(field):
    frames = tj.Frames(_UnreadFrames(), _scan().geometry, white_level=8.0)
    frames = replace(frames, **{field: np.empty((0, 4, 4), np.float32)})
    with pytest.raises(ValueError, match=field):
        frames.corrected()


@pytest.mark.numerical
def test_posed_multi_orbit_selection_keeps_the_original_forward_and_transpose(tmp_path):
    scan = _scan(segmented=True)
    path = tmp_path / "scan.npz"
    tj.save(path, scan)
    from tomojax.io import load_dataset

    record = load_dataset(path)
    assert isinstance(record, ProjectionDataset)
    poses = np.arange(36, dtype=np.float32).reshape(6, 6) / 1000
    record.align_params = poses
    record.align_gauge = {"pose_translation_frame": "detector"}
    save_dataset(path, record)
    scan = tj.load(path)
    kept = np.asarray([1, 3, 5])
    selected = scan.selected(kept)
    np.testing.assert_allclose(selected.poses, scan.poses[kept])
    volume = np.random.default_rng(4).uniform(size=(4, 4, 4)).astype(np.float32)
    whole = np.asarray(tj.project(scan.geometry, volume))
    np.testing.assert_allclose(
        tj.project(selected.geometry, volume), whole[kept], rtol=2e-6, atol=2e-6
    )
    data = np.random.default_rng(5).uniform(size=selected.projections.shape).astype(np.float32)
    padded = np.zeros_like(scan.projections)
    padded[kept] = data
    np.testing.assert_allclose(
        tj.backproject(selected.geometry, data),
        tj.backproject(scan.geometry, padded),
        rtol=2e-6,
        atol=2e-6,
    )


@pytest.mark.parametrize("batch", [None, 1, np.int64(2), 20])
def test_valid_corrections_match_independent_flat_interpolation_and_log(batch):
    scan = _scan()
    views = np.asarray([-2.0, 0.5, 1.5, 4.5, 6.0, 8.0])
    # Unsorted flat sets and repeated positions are averaged before interpolation.
    flats = np.stack([np.full((4, 4), v, np.float32) for v in (21, 9, 13)])
    flat = np.interp(views, [0, 6], [11, 21])
    transmission = np.linspace(0.2, 0.8, 6, dtype=np.float32)
    counts = np.broadcast_to((1 + transmission * (flat - 1))[:, None, None], (6, 4, 4)).copy()
    frames = tj.Frames(
        counts,
        scan.geometry,
        flats=flats,
        darks=np.ones((2, 4, 4), np.float32),
        flat_positions=np.asarray([6.0, 0.0, 0.0]),
        view_positions=views,
    )
    corrected = frames.corrected(BeamHardening((1.0, 0.05)), batch_views=batch)
    p = -np.log(transmission)
    reference = np.broadcast_to((p + 0.05 * p**2)[:, None, None], (6, 4, 4))
    np.testing.assert_allclose(corrected.projections, reference, rtol=2e-6)
    assert corrected.corrections[0].settings["flat_sets"] == 2
