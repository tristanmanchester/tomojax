"""Views keep their place in acquisition time, and the motion prior follows it.

A gap left by rejected or unselected views is charged as the steps it hides, and a
restart (a combined scan) breaks the chain. The oracle for the gap is
marginalization: a random walk observed at fewer times has, at those times, the
precision of the full chain with the missing times integrated out.
"""

import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint._pose import MotionPrior, ViewOrder

# check-public-imports: allow-private
from tomojax.alignment._joint._report import isolated_views
from tomojax.corrections import RejectViews


def _laplacian(n, positions=None):
    """The prior's first-difference precision on n views, at unit robust weight."""
    order = ViewOrder(n, positions)
    prior = MotionPrior(n, np.zeros((n, 0)), knot_spacing=(10_000,), delta=1.0, order=order)
    w = prior.weights(np.zeros(n), 1.0)
    return np.stack([np.asarray(prior._laplacian(np.eye(n)[:, j], w)) for j in range(n)], axis=1)


def test_a_gap_costs_what_the_missing_views_would_have():
    full = _laplacian(14)
    removed = np.r_[4:7, 10]  # an interior block of three and a single view
    kept = np.setdiff1d(np.arange(14), removed)
    marginal = full[np.ix_(kept, kept)] - full[np.ix_(kept, removed)] @ np.linalg.solve(
        full[np.ix_(removed, removed)], full[np.ix_(removed, kept)]
    )
    np.testing.assert_allclose(_laplacian(len(kept), kept + 0.5), marginal, atol=1e-12)


def test_view_order_links_come_from_when_views_were_taken():
    # Interlaced: angles jump about, but the views are stored in time order.
    order = ViewOrder(8, np.arange(8) + 0.5)
    assert order.linked.all() and np.array_equal(order.steps, np.ones(7))
    # Every third view: still one step apart.
    order = ViewOrder(5, 3 * np.arange(5) + 0.5)
    assert np.array_equal(order.steps, np.ones(4))
    # A rejected block of three views between positions 2.5 and 6.5.
    order = ViewOrder(5, [0.5, 1.5, 2.5, 6.5, 7.5])
    assert np.array_equal(order.steps, [1, 1, 4, 1]) and order.linked.all()
    # Two acquisitions combined: no link across the restart.
    order = ViewOrder(6, [0.5, 1.5, 2.5, 0.5, 1.5, 2.5])
    assert order.linked.tolist() == [True, True, False, True, True]
    assert order.segments == [(0, 3), (3, 6)]


def test_positions_survive_selection_rejection_combining_and_saving(tmp_path):
    grid, detector = tj.Grid(16, 16, 8, 1, 1, 1), tj.Detector(16, 8, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, 24, endpoint=False))
    volume = np.zeros((8, 16, 16), np.float32)
    volume[:, 2:14, 2:14] = 1
    noise = np.random.default_rng(0).normal(0, 0.05, (24, 8, 16)).astype(np.float32)
    projections = np.array(tj.project(geometry, volume)) + noise
    projections[9] *= 0.2  # a beam drop RejectViews removes
    scan = tj.Scan(projections, geometry)
    assert scan.view_positions is None

    corrected = scan.corrected(RejectViews())
    assert corrected.corrections[-1].found["rejected"] == [9]
    np.testing.assert_array_equal(corrected.view_positions, np.r_[0:9, 10:24] + 0.5)
    chosen = corrected.selected(slice(6, 12, 2))
    np.testing.assert_array_equal(chosen.view_positions, [6.5, 8.5, 11.5])
    np.testing.assert_array_equal(chosen.binned(2).view_positions, chosen.view_positions)

    # Combining acquisitions (cone orbits) restarts each one's positions.
    cone = tj.ConeGeometry(
        grid, detector, np.linspace(0, 360, 6, endpoint=False), tj.ConeBeam(40, 60)
    )
    orbit = tj.Scan(np.zeros((6, 8, 16), np.float32), cone)
    combined = tj.Scan.combine([orbit, orbit.selected(slice(0, 4))])
    np.testing.assert_array_equal(combined.view_positions, np.r_[np.arange(6), np.arange(4)] + 0.5)
    assert ViewOrder(10, combined.view_positions).segments == [(0, 6), (6, 10)]

    tj.save(tmp_path / "chosen.npz", chosen)
    loaded = tj.load(tmp_path / "chosen.npz").view_positions
    np.testing.assert_array_equal(loaded, chosen.view_positions)
    with pytest.raises(ValueError, match="one finite position for each view"):
        tj.Scan(projections, geometry, view_positions=np.arange(23.0))


def test_a_view_next_to_a_break_is_not_judged_against_another_acquisition():
    values = np.random.default_rng(1).normal(0, 0.01, (20, 1))
    values[[6, 12]] += 1.0  # the same isolated spike, mid-run and first after a restart
    restart = np.r_[np.arange(12), np.arange(8)] + 0.5
    assert isolated_views(values) == [6, 12]
    assert isolated_views(values, ViewOrder(20, restart)) == [6]


@pytest.mark.slow
def test_a_regular_selection_aligns_exactly_as_consecutive_views():
    from bench.fit_alignment import make_case

    scan, _, _, _ = make_case("translations", geometry="parallel", size=24, views=40, grid_size=12)
    plain = tj.align(scan, fit="translations").poses
    every_third = tj.Scan(scan.projections, scan.geometry, view_positions=3 * np.arange(40) + 0.5)
    np.testing.assert_array_equal(tj.align(every_third, fit="translations").poses, plain)


def test_a_slip_inside_a_gap_is_recovered_at_its_true_size():
    from bench.fit_alignment import make_case
    from scipy import ndimage

    scan, _, _, meta = make_case(
        "translations", geometry="parallel", size=24, views=60, grid_size=12, solid=True
    )
    true = np.asarray(meta["true_scan"].poses, np.float64)[:, 3] / scan.detector.du
    projections = np.array(scan.projections)
    slip = 1.5  # detector px from view 33 on: a lab-x move of the whole object
    for i in range(33, 60):
        projections[i] = ndimage.shift(projections[i], (0, slip), order=3, mode="nearest")
    true[33:] += slip
    kept = np.r_[0:27, 39:60]  # the slip happens while views 27-38 are missing
    fitted = tj.align(tj.Scan(projections, scan.geometry).selected(kept), fit="translations")
    error = np.asarray(fitted.poses, np.float64)[:, 3] / scan.detector.du - true[kept]
    theta = np.deg2rad(np.asarray(scan.angles))[kept]
    gauge = np.column_stack([np.cos(theta), np.sin(theta), np.ones(len(kept))])
    error -= gauge @ np.linalg.lstsq(gauge, error, rcond=None)[0]
    assert abs(error[27] - error[26]) < 0.05 * slip
