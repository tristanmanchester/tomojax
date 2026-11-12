"""RejectViews drops the views of an abrupt event and keeps the object's own change, however fast."""

import numpy as np

import tomojax as tj
from tomojax.corrections import RejectViews


def _scan(events):
    """A full-turn parallel scan of a small cylinder in a container, 360 views, with noise.

    The cylinder covers about a sixth of the detector and the container's wall
    the middle of it, as for a pebble in a tube: most pixels see the container.
    """
    grid, detector = tj.Grid(48, 48, 4, 1, 1, 1), tj.Detector(64, 4, 1, 1)
    geometry = tj.ParallelGeometry(grid, detector, np.arange(360.0))
    y, x = np.mgrid[:48, :48] - 23.5
    volume = np.broadcast_to(((x**2 + y**2) < 16).astype(np.float32) * 0.05, (4, 48, 48))
    projections = np.array(tj.project(geometry, np.ascontiguousarray(volume)))
    projections += np.random.default_rng(3).normal(0, 0.005, projections.shape).astype(np.float32)
    projections += 0.1 * WALL
    events(projections)
    return tj.Scan(projections, geometry)


WALL = np.exp(-0.5 * ((np.arange(64) - 32) / 25) ** 2)  # the container wall's footprint


def _rejected(scan):
    return scan.corrected(RejectViews()).corrections[-1].found["rejected"]


def test_a_beam_loss_is_rejected_whole_and_a_fast_real_change_is_kept():
    def events(p):
        p[100:130] += np.log(2)  # 30 frames at half the flux
        p[250] += np.log(4)  # one frame at a quarter
        for start in (40, 220):  # the wall turns edge-on, and again half a turn later
            rise = np.sin(np.pi * np.arange(50) / 50) ** 2
            p[start : start + 50] += 0.4 * rise[:, None, None] * WALL

    assert _rejected(_scan(events)) == [*range(100, 130), 250]


def test_a_failed_view_at_either_end_and_nothing_on_a_clean_scan():
    def first(p):
        p[0] += 2.0

    def last(p):
        p[-1] += 2.0

    assert _rejected(_scan(first)) == [0]
    assert _rejected(_scan(last)) == [359]
    assert _rejected(_scan(lambda p: None)) == []


def test_a_level_change_that_never_returns_keeps_its_views():
    def events(p):
        p[200:] += 0.05  # the flux settles lower for the rest of the scan

    assert _rejected(_scan(events)) == []
