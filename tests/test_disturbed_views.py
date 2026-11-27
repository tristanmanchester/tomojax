"""Measured detector disturbances and an explicitly held-out alignment interval."""

from dataclasses import replace

import numpy as np
import pytest

import tomojax as tj
from tomojax.alignment import AlignConfig
from tomojax.alignment.api import disturbed_views


@pytest.mark.slow
def test_excluded_views_do_not_affect_fit_and_receive_interpolated_pose():
    sim = tj.simulate("shepp", size=12, n_views=24, geometry="parallel").scan
    held = np.arange(9, 14)
    keep = np.ones(24, bool)
    keep[held] = False
    position = np.arange(24, dtype=float) ** 1.1
    scan = replace(sim, view_positions=position)
    config = AlignConfig(levels=(1,), iterations=2, refine_iterations=0)
    ref = tj.align(scan.selected(keep), config=config)
    values = np.array(scan.projections)
    values[held] += 2
    fit = tj.align(
        replace(scan, projections=values), config=replace(config, exclude_views=((9, 14),))
    )
    np.testing.assert_array_equal(fit.image, ref.image)
    np.testing.assert_array_equal(fit.poses[keep], ref.poses)
    for column in (3, 4):
        np.testing.assert_allclose(
            fit.poses[held, column], np.interp(position[held], position[keep], ref.poses[:, column])
        )
    np.testing.assert_array_equal(fit.scan.projections, values)
    assert fit.info["disturbed_views"]["score"][9:14] == [None] * 5
    assert fit.info["fit_views"] == np.flatnonzero(keep).tolist()


def test_row_common_shadow_report_is_distinct_from_specimen_slip():
    n, rows, cols = 120, 36, 48
    angles = np.linspace(0, 180, n, endpoint=False)
    a = np.deg2rad(angles)[:, None, None]
    v, u = np.mgrid[:rows, :cols]
    # A translated, rotating specimen with distinct top and bottom structure.
    x = u - cols / 2 - 3 * np.sin(a) - 2 * (a > 1.1)
    z = v - rows / 2 + np.cos(a)
    data = np.exp(-((x - 3 * np.cos(a)) ** 2 / 65 + (z + 2 * np.sin(a)) ** 2 / 32))
    data += np.random.default_rng(17).normal(0, 0.004, data.shape)
    geometry = tj.ParallelGeometry(
        tj.Grid(32, 32, 24, 1, 1, 1), tj.Detector(cols, rows, 1, 1), angles
    )
    clean = disturbed_views(data.astype("f4"), geometry)
    assert clean["ranges"] == []
    affected = np.arange(61, 70)
    shadow = 0.24 * np.exp(-(((u[0] - np.linspace(8, 32, len(affected))[:, None]) / 4) ** 2))
    data[affected] += shadow[:, None, :]
    report = disturbed_views(data.astype("f4"), geometry)
    assert report["ranges"] == [[61, 70]]
    assert report["angles"] == [[angles[61], angles[69]]]
    assert "crosses the field of view" in report["message"]
    assert "Inspect" in report["message"]

    # With the entire upper band masked, the lower band alone supplies no evidence.
    data[:, : rows // 2] = np.nan
    masked = disturbed_views(data.astype("f4"), geometry)
    assert masked["ranges"] == [] and masked["score"] == [None] * n
    assert "Too few" in masked["message"]


@pytest.mark.slow
def test_pose_interpolation_stops_at_trajectory_boundaries():
    scan = tj.simulate("shepp", size=12, n_views=24, translation_sigma=0.4).scan
    scan = replace(scan, view_positions=np.tile(np.arange(12), 2))
    cfg = AlignConfig(exclude_views=((9, 15),), iterations=1, levels=(1,), refine_iterations=0)
    result = tj.align(scan, config=cfg)
    # The left trajectory ends at 8. The next starts at 15.
    np.testing.assert_array_equal(result.poses[9:12], np.repeat(result.poses[8:9], 3, axis=0))
    np.testing.assert_array_equal(result.poses[12:15], np.repeat(result.poses[15:16], 3, axis=0))
    with pytest.raises(ValueError, match="one per trajectory"):
        tj.align(scan, config=replace(cfg, exclude_views=((0, 12),)))


@pytest.mark.parametrize("intervals", [None, 0, ((True, 4),), ((2, 2),), ((2.5, 3),), (4,)])
def test_invalid_exclusion_ranges_fail_at_configuration(intervals):
    with pytest.raises(ValueError, match="exclude_views"):
        AlignConfig(exclude_views=intervals)
