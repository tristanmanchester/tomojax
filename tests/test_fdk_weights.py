"""FDK quadrature must depend on physical angles, not their turn labels."""

from __future__ import annotations

import numpy as np
import pytest

from tomojax.geometry import ConeBeam, ConeGeometry, Detector, Grid
from tomojax.recon.fdk import view_weights


def geometry(angles):
    return ConeGeometry(Grid(4, 4, 4, 1, 1, 1), Detector(8, 8, 1, 1), angles, ConeBeam(12, 20))


@pytest.mark.parametrize("arc", [210, 360])
@pytest.mark.parametrize("labels", ["wrapped", "mixed_turns", "negative_turns", "shuffled"])
def test_angular_weights_ignore_equivalent_angle_labels(arc, labels):
    angles = np.linspace(270, 270 + arc, 72, endpoint=arc < 360)
    reference = geometry(angles)
    expected = view_weights(reference, reference.detector, len(angles))
    order = np.arange(len(angles))
    if labels == "wrapped":
        angles = angles % 360
    elif labels == "mixed_turns":
        angles = angles + 360 * (order % 3)
    elif labels == "negative_turns":
        angles = angles - 720
    else:
        order = np.random.default_rng(7).permutation(order)
        angles = (angles % 360)[order]
    actual = view_weights(geometry(angles), reference.detector, len(angles))
    assert np.isfinite(actual).all() and np.all(actual >= 0)
    np.testing.assert_allclose(actual, expected[order], rtol=1e-11, atol=1e-13)


def test_irregular_full_turn_weights_match_circular_midpoint_quadrature():
    # A centred detector gives half each view's angular Voronoi-cell width.
    angles = np.array([0, 60, 150, 260])
    scan = geometry(angles)
    weights = view_weights(scan, scan.detector, len(angles))
    expected = np.deg2rad([40, 37.5, 50, 52.5])
    np.testing.assert_allclose(weights, np.broadcast_to(expected[:, None], weights.shape))
    np.testing.assert_allclose(weights.sum(axis=0), np.pi)


@pytest.mark.parametrize("arc", [210, 360])
def test_repeated_angles_share_measure_even_across_turn_boundaries(arc):
    angles = np.linspace(270, 270 + arc, 72, endpoint=arc < 360)
    scan = geometry(angles)
    expected = view_weights(scan, scan.detector, len(angles))
    repeated = np.concatenate([angles, angles % 360, angles - 720])
    order = np.random.default_rng(8).permutation(len(repeated))
    weights = view_weights(geometry(repeated[order]), scan.detector, len(repeated))
    np.testing.assert_allclose(weights, np.tile(expected / 3, (3, 1))[order], atol=1e-13)


@pytest.mark.parametrize("start", [0, 270, -450])
def test_wrapping_a_too_short_scan_does_not_make_a_full_turn(start):
    angles = np.linspace(start, start + 150, 40) % 360
    scan = geometry(angles)
    with pytest.raises(ValueError, match="short scans"):
        view_weights(scan, scan.detector, len(angles))


@pytest.mark.parametrize("angles", [[0, np.nan, 240], [0, 120, np.inf], [[0], [120], [240]]])
def test_fdk_weights_refuse_malformed_or_nonfinite_angles(angles):
    scan = geometry(angles)
    with pytest.raises(ValueError, match="finite rotation angle"):
        view_weights(scan, scan.detector, len(angles))


@pytest.mark.parametrize("angles", [[0], [0, 360, 720]])
def test_fdk_weights_require_distinct_angles(angles):
    scan = geometry(angles)
    with pytest.raises(ValueError, match="distinct rotation angles"):
        view_weights(scan, scan.detector, len(angles))


@pytest.mark.parametrize("arc", [240, 300, 350])
def test_overscan_conjugate_rays_partition_one(arc):
    # +/-5-degree fan columns and half-degree views have exact acquired
    # conjugates at beta + 180 + 2*gamma, independent of the weight formula.
    step = 0.5
    angles = np.arange(0, arc, step)
    detector = Detector(3, 3, 120 * np.tan(np.deg2rad(5)), 1)
    scan = ConeGeometry(Grid(3, 3, 3, 1, 1, 1), detector, angles, ConeBeam(60, 120))
    weights = view_weights(scan, detector, len(angles)) / np.deg2rad(step)
    assert np.all(weights[1:] > 0)
    for column, fan in enumerate([-5, 0, 5]):
        shift = int((180 + 2 * fan) / step)
        np.testing.assert_allclose(
            weights[:-shift, column] + weights[shift:, 2 - column], 1, atol=2e-12
        )


def test_changing_and_repeated_panels_share_actual_conjugate_rays():
    from tomojax.geometry import ConeSegments

    # A five-degree fan has an exact conjugate 190 degrees later. Some panels
    # do not contain -5 degrees; repetitions differ between angular nodes.
    step = 10
    du = 120 * np.tan(np.deg2rad(5))
    parts, labels, offsets = [], [], []
    for angle in range(0, 360, step):
        for offset in [2] if angle % 30 == 0 else [0, 2]:
            parts.append(
                ConeGeometry(
                    Grid(3, 3, 3, 1, 1, 1),
                    Detector(5, 3, du, 1, (offset * du, 0)),
                    [angle],
                    ConeBeam(60, 120),
                )
            )
            labels.append(angle)
            offsets.append(offset)
    scan = ConeSegments(tuple(parts))
    weights = view_weights(scan, scan.detector, len(parts)) / np.deg2rad(step)
    for angle in range(0, 360, step):
        total = sum(
            weights[i, 3 - offset]
            for i, (a, offset) in enumerate(zip(labels, offsets, strict=True))
            if a == angle
        )
        total += sum(
            weights[i, 1 - offset]
            for i, (a, offset) in enumerate(zip(labels, offsets, strict=True))
            if a == (angle + 190) % 360 and offset <= 1
        )
        np.testing.assert_allclose(total, 1, atol=2e-12)
