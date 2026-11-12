"""The facts tj.align reports and the views it flags, on cases built by hand."""

import numpy as np
from scipy.spatial.transform import Rotation

# check-public-imports: allow-private
from tomojax.alignment._joint._report import axial, build, isolated_views, unusual_views


def _drift(views, columns=2):
    """A smooth correction with small view-to-view scatter, in working pixels."""
    t = np.linspace(0, 1, views)
    rng = np.random.default_rng(4)
    smooth = np.stack([0.6 * np.sin(2 * np.pi * t + k) for k in range(columns)], 1)
    return smooth + rng.normal(0, 0.003, (views, columns))


def test_a_spike_is_isolated_in_either_direction_and_column():
    steps = _drift(60)
    steps[17, 0] += 0.8
    steps[41, 1] -= 0.5
    assert isolated_views(steps) == [17, 41]


def test_a_slip_is_a_step_and_not_isolated():
    steps = _drift(60)
    steps[25:, 0] += 0.8  # a slip: the view leaves one neighbour, not both
    steps[40:, 1] -= 1.2
    assert isolated_views(steps) == []


def test_a_jump_below_the_working_pixel_floor_is_not_flagged():
    steps = np.zeros((30, 2))  # no scatter: any jump exceeds five robust SDs
    steps[11, 0] = 0.05
    steps[19, 1] = 0.2
    assert isolated_views(steps) == [19]


def test_a_view_the_image_explains_badly_is_unusual():
    # Unusual means a residual RMS three times the neighbours': energy 9x.
    energy = 1 + np.random.default_rng(8).normal(0, 0.02, 80)
    energy[[9, 33, 63]] = [12.0, 7.0, 9.5]  # 7x is below the bar
    assert unusual_views(energy) == [9, 63]


def test_unusual_is_judged_against_neighbours_not_the_scan_angle_trend():
    # Residual energy varies smoothly with angle by a factor of three; only the
    # frames that differ from their neighbours by the bar are unusual.
    angle = np.linspace(0, 4 * np.pi, 200)
    energy = (2 + np.cos(angle)) * (1 + np.random.default_rng(9).normal(0, 0.02, 200))
    energy[[40, 151]] *= 12.0
    energy[90] *= 3.0  # a modest outlier, as clean data shows at grid-aligned angles
    assert unusual_views(energy) == [40, 151]


def test_report_states_sizes_in_native_pixels_and_flags_in_working_pixels():
    views, native, binning = 90, (2.0, 3.0), 2
    working = _drift(views)
    working[30, 0] += 1.0
    table = np.zeros((views, 6))
    # Detector translations: dx in units of du, dz in units of dv.
    table[:, 3] = working[:, 0] * binning * native[0]
    table[:, 4] = working[:, 1] * binning * native[1]
    levels = [{"steps": [{}] * 3, "status": "pose_steps_below_tolerance"}]
    offsets = np.column_stack([np.linspace(-0.1, 0.2, views), np.zeros(views)])
    energy = np.ones(views)
    energy[50] = 10.0
    report = build(
        table=table,
        local=(3, 4),
        fit=("translations",),
        axial=True,
        corrections={},
        native=native,
        binning=binning,
        levels=levels,
        refinement={"status": "step_budget_reached"},
        offsets=offsets,
        energy=energy,
        spacing=10,
    )
    assert report["isolated_views"] == [30]
    assert report["unusual_views"] == [50]
    assert report["levels"] == [{"steps": 3, "ended": "pose_steps_below_tolerance"}]
    dx = report["corrections"]["per_view"]["dx"]
    assert dx["max"] == np.abs(working[:, 0] * binning).max()
    # The spline carries the drift (RMS 0.6 / sqrt(2) working px); the view-to-view
    # part is the spike, 1 working px in one of 90 views, and the scatter.
    np.testing.assert_allclose(dx["smooth_rms"], 0.6 / np.sqrt(2) * binning, rtol=0.05)
    np.testing.assert_allclose(dx["view_to_view_rms"], binning / np.sqrt(views), rtol=0.2)
    np.testing.assert_allclose(report["offsets"]["offset_range"], [-0.1, 0.2])


def test_axial_means_the_rotation_axis_is_perpendicular_to_the_beam():
    angles = np.deg2rad(np.arange(0, 360, 30))
    poses = np.repeat(np.eye(4)[None], len(angles), axis=0)
    beam = (0.0, 1.0, 0.0)
    for tilt, expected in ((0.0, True), (2.0, True), (30.0, False)):
        axis = Rotation.from_euler("x", tilt, degrees=True).apply([0.0, 0.0, 1.0])
        poses[:, :3, :3] = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
        assert axial(poses, beam) is expected


def test_rotations_on_an_axial_geometry_carry_the_held_out_advisory():
    views = 40
    common = {
        "table": np.zeros((views, 6)),
        "local": (0, 1, 2, 3, 4),
        "corrections": {},
        "native": (1.0, 1.0),
        "binning": 1,
        "levels": [],
        "refinement": {},
        "offsets": np.zeros((views, 2)),
        "energy": np.ones(views),
        "spacing": 10,
    }
    with_rotations = build(fit=("translations", "rotations"), axial=True, **common)
    assert len(with_rotations["advisories"]) == 1
    assert "0.14-0.77 px" in with_rotations["advisories"][0]
    assert build(fit=("translations", "rotations"), axial=False, **common)["advisories"] == []
    assert build(fit=("translations",), axial=True, **common)["advisories"] == []
