"""The motion prior's weight takes its error scale from the views' own scatter.

Each view of these scans is multiplied by its own smooth fringe (random orientation
and phase, half a cycle to a cycle and a half across the detector): error that differs
from view to view and is not motion. The fit must not turn it into motion.
"""

import numpy as np
import pytest
from scipy.interpolate import LSQUnivariateSpline

import tomojax as tj

VIEWS, DELTA_PX = 90, 0.02


def _scan(fringe, step):
    n, nz = 32, 24
    angles = np.linspace(0, 360, VIEWS, endpoint=False)
    geometry = tj.ParallelGeometry(tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(40, 28, 1, 1), angles)
    z, y, x = np.mgrid[:nz, :n, :n]
    rng = np.random.default_rng(3)
    volume = np.zeros((nz, n, n), np.float32)
    for _ in range(8):
        c, r = rng.uniform([6, 8, 8], [nz - 6, n - 8, n - 8]), rng.uniform(2.5, 5)
        volume[((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2) <= r * r] += rng.uniform(
            0.5, 1.2
        )
    table = np.zeros((VIEWS, 6))
    table[VIEWS // 3 :, 3] = step
    empty = tj.Scan(np.zeros((VIEWS, 28, 40), np.float32), geometry)
    data = np.array(tj.project(empty.with_poses(table).geometry, volume), np.float64)
    v, u = np.mgrid[:28, :40]
    draws = np.random.default_rng(11)
    for k in range(VIEWS):
        theta, phase, cycles = (
            draws.uniform(0, np.pi),
            draws.uniform(0, 2 * np.pi),
            draws.uniform(0.5, 1.5),
        )
        wave = np.sin(
            2 * np.pi * cycles * (u / 40 * np.cos(theta) + v / 28 * np.sin(theta)) + phase
        )
        data[k] *= 1 + fringe * wave
    data += np.random.default_rng(7).normal(0, 0.01, data.shape)
    return tj.align(tj.Scan(data.astype(np.float32), geometry), fit="translations")


def _invented_jumps(result):
    """View-to-view jumps of the fitted u shift above delta, a smooth drift removed (no step)."""
    shift = np.asarray(result.poses)[:, 3]
    views = np.arange(VIEWS)
    drift = LSQUnivariateSpline(views, shift, np.arange(10, VIEWS - 10, 10), k=3)(views)
    return int(np.sum(np.abs(np.diff(shift - drift)) > DELTA_PX))


@pytest.mark.numerical
@pytest.mark.slow
def test_error_that_is_not_motion_raises_the_weight_and_is_not_followed():
    # The photon-scaled weight followed the fringe: 12 invented jumps above delta at 2%
    # and 55 at 5% (20 of them above 0.05 px), with the same weight as on the clean scan.
    # The scatter-derived weight rises with the fringe (about 0.7, 23 and 58 times the
    # photon weight here) and the fit invents none.
    scale = {}
    for fringe in (0.0, 0.02, 0.05):
        result = _scan(fringe, 0.0)
        scale[fringe] = result.info["levels"][-1]["translation_noise"]
        assert _invented_jumps(result) == 0, fringe
    assert scale[0.0] < scale[0.02] < scale[0.05]
    assert scale[0.05] > 30 * scale[0.0]


@pytest.mark.numerical
@pytest.mark.slow
def test_a_real_step_is_followed_and_the_tail_taxes_it_where_the_weight_is_large():
    # A 1.5 px slip, 75 delta. On the clean scan it is followed within 5%. Under the 5%
    # fringe the weight is about fifty times the photon one, and the Student-t's force
    # on a jump of J >> delta, 2 w delta / J, still exceeds the photon threshold: the
    # step is followed about a fifth short (1.22 px). That is the tail's tax, an open
    # defect (v1 open items, "The motion prior taxes real steps where the weight is
    # large"); this bound records it and should tighten when that item is fixed.
    for fringe, tolerance in ((0.0, 0.05), (0.05, 0.25)):
        shift = np.asarray(_scan(fringe, 1.5).poses)[:, 3]
        fitted = shift[VIEWS // 3] - shift[VIEWS // 3 - 1]
        assert abs(fitted - 1.5) < tolerance * 1.5, (fringe, fitted)
