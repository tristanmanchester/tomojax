"""Integrated counts with unequal exposures share one count-rate illumination model."""

import numpy as np

import tomojax as tj
from tomojax.corrections import correct_frames, corrected_row_source


def test_exposure_dark_and_temporal_interpolation_match_physical_counts():
    dark = np.arange(24).reshape(4, 6) + 7.0
    rate = np.arange(24).reshape(4, 6) + 100.0
    flat_times = np.array([1.0, 3.0, 2.0, 4.0])
    positions = np.array([0.0, 0.0, 10.0, 10.0])
    flats = dark + flat_times[:, None, None] * rate * np.array([1, 1, 2, 2])[:, None, None]
    view_times, at = np.array([2.0, 5.0, 3.0]), np.array([0.0, 4.0, 10.0])
    truth = np.array([0.2, 0.5, 0.8])[:, None, None] + np.zeros((3, 4, 6))
    counts = dark + view_times[:, None, None] * rate * (1 + at[:, None, None] / 10) * np.exp(-truth)
    settings = dict(
        flats=flats,
        flat_positions=positions,
        darks=dark[None],
        white_level=None,
        view_positions=at,
        exposure_times=view_times,
        flat_exposure_times=flat_times,
        steps=(),
        epsilon=1e-6,
    )
    eager = correct_frames(counts, batch_views=1, **settings)
    np.testing.assert_allclose(eager.projections, truth, atol=2e-7)
    with corrected_row_source(counts, **settings) as source:
        np.testing.assert_allclose(source(1, 3).projections, truth[:, 1:3], atol=2e-7)
    geometry = tj.ParallelGeometry(tj.Grid(6, 6, 4, 1, 1, 1), tj.Detector(6, 4, 1, 1), [0, 60, 120])
    frames = tj.Frames(
        counts,
        geometry,
        flats=flats,
        darks=dark[None],
        flat_positions=positions,
        view_positions=at,
        exposure_times=view_times,
        flat_exposure_times=flat_times,
    )
    selected = frames.selected([0, 2]).cropped(slice(1, 3), slice(2, 5)).corrected()
    np.testing.assert_allclose(selected.projections, truth[[0, 2], 1:3, 2:5], atol=2e-7)
