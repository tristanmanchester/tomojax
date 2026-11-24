"""Physical motion components and the exact tier: units, frames, composition and acceptance."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "phantoms"))
import exact_tier
import motion

# check-public-imports: allow-private

ANGLES = np.linspace(0.0, 360.0, 72, endpoint=False)


def axial_nominal(angles):
    """Parallel axial nominal poses: the object turns about its own z, which is lab z."""
    out = np.tile(np.eye(4), (len(angles), 1, 1))
    for i, a in enumerate(np.deg2rad(angles)):
        out[i, :3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    return out


def lab_point(poses, point):
    return np.einsum("nij,j->ni", poses[:, :3, :3], point) + poses[:, :3, 3]


def test_a_slip_is_an_object_frame_step_that_turns_with_the_sample():
    nominal = axial_nominal(ANGLES)
    shift = (5.0, 0.0, 0.0)  # micrometres along object x
    slipped = motion.poses(
        motion.Motion(slips=(motion.Slip(30, shift),)), nominal, ANGLES, np.zeros(3)
    )
    still = motion.poses(motion.Motion(), nominal, ANGLES, np.zeros(3))
    moved = lab_point(slipped["poses"], np.zeros(3)) - lab_point(still["poses"], np.zeros(3))
    np.testing.assert_allclose(moved[:30], 0.0, atol=1e-12)
    # After the slip the 5 um object-frame shift turns with the sample: lab x is 5 um cos(angle).
    theta = np.deg2rad(ANGLES[30:])
    np.testing.assert_allclose(moved[30:, 0], 5e-3 * np.cos(theta), atol=1e-12)
    np.testing.assert_allclose(moved[30:, 1], 5e-3 * np.sin(theta), atol=1e-12)


def test_a_tip_turns_the_sample_about_its_center():
    center = np.array([0.2, -0.1, 0.4])
    tip = motion.Slip(0, (0.0, 0.0, 0.0), tip_rad=1e-3, tip_axis=(0.0, 1.0, 0.0))
    transform = tip.transform(center)
    np.testing.assert_allclose(transform[:3, :3] @ center + transform[:3, 3], center, atol=1e-15)
    edge = center + np.array([0.0, 0.0, 0.5])
    moved = transform[:3, :3] @ edge + transform[:3, 3] - edge
    np.testing.assert_allclose(np.linalg.norm(moved), 0.5e-3, rtol=1e-6)


def test_stage_tilt_reaches_the_sample_as_tilt_times_height():
    nominal = axial_nominal(ANGLES)
    m = motion.Motion(tilt_urad=((10.0,), ()), tilt_phase=((0.0,), ()), sample_height_mm=100.0)
    lab = lab_point(motion.poses(m, nominal, ANGLES, np.zeros(3))["poses"], np.zeros(3))
    # Tilt about lab x by 10 urad cos(angle) at 100 mm above the bearing moves the sample in lab y.
    np.testing.assert_allclose(
        lab[:, 1], -100.0 * 10e-6 * np.cos(np.deg2rad(ANGLES)), rtol=1e-6, atol=1e-12
    )


def test_recorded_components_compose_to_the_poses():
    nominal = axial_nominal(ANGLES)
    ranges = {
        "center_um": [1, 5],
        "axis_tilt": [0.1, 0.5],
        "axis_roll_deg": [0.0, 0.2],
        "harmonics": 3,
        "radial_um": [0.5, 2],
        "axial_um": [0.5, 2],
        "tilt_urad": [4, 12],
        "asynchronous_fraction": [0.1, 0.2],
        "drift_um_per_hour": [1, 5],
        "drift_curvature": [-0.5, 0.5],
        "encoder_arcsec": [1, 5],
        "scan_minutes": [10, 60],
        "sample_height_mm": [20, 100],
        "slips": [2, 2],
        "slip_um": [1, 10],
        "tip": "geometric",
    }
    m = motion.draw(np.random.default_rng(7), ranges, views=len(ANGLES), sample_size_mm=1.0)
    out = motion.poses(m, nominal, ANGLES, np.zeros(3))
    encoder = np.tile(np.eye(4), (len(ANGLES), 1, 1))
    c, s = np.cos(out["encoder_rad"]), np.sin(out["encoder_rad"])
    encoder[:, 0, 0], encoder[:, 0, 1], encoder[:, 1, 0], encoder[:, 1, 1] = c, -s, s, c
    np.testing.assert_allclose(
        out["stage"] @ nominal @ encoder @ out["sample"], out["poses"], atol=1e-14
    )
    assert len(m.slips) == 2
    for slip in m.slips:
        size_um = np.linalg.norm(slip.shift_um)
        np.testing.assert_allclose(abs(slip.tip_rad), size_um * 1e-3 / 1.0, rtol=1e-12)
        assert slip.tip_axis[2] == 0.0


def test_stage_motion_is_smooth_and_only_slips_step():
    nominal = axial_nominal(ANGLES)
    ranges = {
        "harmonics": 3,
        "radial_um": [2, 2],
        "axial_um": [2, 2],
        "drift_um_per_hour": [5, 5],
        "scan_minutes": [30, 30],
    }
    m = motion.draw(np.random.default_rng(3), ranges, views=len(ANGLES), sample_size_mm=1.0)
    lab = lab_point(motion.poses(m, nominal, ANGLES, np.zeros(3))["poses"], np.zeros(3))
    # Three harmonics of at most 2 um each change by at most 2 pi k / n per view.
    step = np.abs(np.diff(lab, axis=0)).max()
    assert step < 2e-3 * 2 * np.pi * (1 + 2 / 2 + 3 / 3) / len(ANGLES) * 1.01 + 5e-3 * 0.5 / len(
        ANGLES
    )


def test_draws_repeat_by_seed_and_files_get_their_own_noise():
    ranges = {"radial_um": [0.5, 2], "slips": [0, 3], "slip_um": [1, 10]}
    a = motion.draw(np.random.default_rng(11), ranges, views=72, sample_size_mm=1.0)
    b = motion.draw(np.random.default_rng(11), ranges, views=72, sample_size_mm=1.0)
    c = motion.draw(np.random.default_rng(12), ranges, views=72, sample_size_mm=1.0)
    assert a == b and a != c
    seeds = {
        exact_tier.noise_seed(0, row, photons)
        for row in ("watch/parallel", "rock/parallel")
        for photons in (1e3, 1e4)
    }
    assert len(seeds) == 4


def _landmark_px(poses, true):
    """RMS detector-pixel error of 27 points after the best rigid gauge (1 mm pixels)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    points = np.stack(np.meshgrid(*[np.linspace(-7, 7, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)

    def pixels(p, pts):
        return (np.einsum("nij,kj->nki", p[:, :3, :3], pts) + p[:, None, :3, 3])[..., [0, 2]]

    def residual(z):
        moved = points @ Rotation.from_rotvec(z[:3]).as_matrix().T + z[3:]
        return (pixels(poses, moved) - pixels(true, points)).ravel()

    fitted = least_squares(residual, np.zeros(6)).x
    return float(np.sqrt(np.mean(residual(fitted) ** 2) * 2))


def test_gauge_free_truth_has_no_component_along_the_fits_rigid_translations():
    nominal = axial_nominal(ANGLES)
    m = motion.Motion(
        center_offset_um=300.0,
        radial_um=((400.0,), (300.0,)),
        radial_phase=((0.3,), (1.1,)),
        axial_um=(250.0,),
        axial_phase=(0.5,),
    )
    raw = motion.poses(m, nominal, ANGLES, np.zeros(3))["poses"]
    free = exact_tier.gauge_free(raw, nominal, pitch=(1.0, 1.0), lever_px=12.0)
    table = exact_tier.corrections(nominal, free)
    state = np.stack([table[:, 3], table[:, 4]], -1).reshape(-1)
    tangents = np.stack(
        [np.stack([nominal[:, 0, k], nominal[:, 2, k]], -1).reshape(-1) for k in range(3)], -1
    )
    raw_table = exact_tier.corrections(nominal, raw)
    raw_state = np.stack([raw_table[:, 3], raw_table[:, 4]], -1).reshape(-1)
    # Zero to the least-squares solve's precision, relative to the raw truth's component.
    assert np.abs(tangents.T @ state).max() < 1e-8 * np.abs(tangents.T @ raw_state).max()
    # The same object moved rigidly: the rotations are untouched and the shift is common.
    np.testing.assert_allclose(free[:, :3, :3], raw[:, :3, :3], atol=1e-15)
    assert _landmark_px(free, raw) < 1e-9


def _exact_tier_ends(labels=("raw", "gauge-free"), config=None):
    """Landmark error of tj.align from nominal against the raw and the gauge-free truth."""
    import tomojax as tj
    from tomojax.geometry import stack_view_poses

    grid = tj.Grid(24, 24, 24, 1.0, 1.0, 1.0)
    detector = tj.Detector(32, 32, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, ANGLES)
    nominal = np.asarray(stack_view_poses(geometry, len(ANGLES)), np.float64)
    z, y, x = np.mgrid[:24, :24, :24] - 11.5
    truth = np.exp(-((x - 3) ** 2 + y**2 + (z + 2) ** 2) / 20) + 0.6 * (np.abs(x + 4) < 3) * (
        np.abs(y - 2) < 4
    ) * (np.abs(z) < 6)
    m = motion.Motion(
        center_offset_um=300.0,
        radial_um=((0.0, 400.0), (0.0, 300.0)),
        radial_phase=((0.0, 0.3), (0.0, 1.1)),
        axial_um=(250.0,),
        axial_phase=(0.5,),
        slips=(motion.Slip(40, (300.0, -200.0, 150.0)),),
    )
    raw = motion.poses(m, nominal, ANGLES, np.zeros(3))["poses"]
    truths = {
        "raw": raw,
        "gauge-free": exact_tier.gauge_free(raw, nominal, pitch=(1.0, 1.0), lever_px=12.0),
    }
    ends = {}
    for label in labels:
        true = truths[label]
        scan = tj.Scan(exact_tier.projections(grid, detector, truth, true), geometry)
        result = tj.align(scan, fit="translations", config=config)
        ends[label] = _landmark_px(
            np.asarray(stack_view_poses(result.scan.geometry, len(ANGLES)), np.float64), true
        )
    return ends


@pytest.mark.numerical
@pytest.mark.slow
@pytest.mark.parametrize("label", ["gauge-free", "raw"])
def test_the_exact_tiers_objective_has_its_minimum_at_the_gauge_free_truth(label):
    # The exact tier's data are the fit's own model, so the objective's minimum is the truth
    # held in the fit's gauge. Run long with the stop rule off, the fit reaches it from a
    # 0.5 px start; a truth held at another sub-voxel position than the fit's gauge cannot be
    # reached, which is what this tier must exclude. Anything that moves the minimum (a
    # pedestal, a filter's free band) fails here. The two fits are separate tests so they
    # can run side by side.
    from tomojax.alignment import AlignConfig

    end = _exact_tier_ends((label,), config=AlignConfig(iterations=100, tolerance=0.0))[label]
    assert end < 2e-4 if label == "gauge-free" else end > 5e-3


@pytest.mark.numerical
def test_the_default_call_on_exact_data_ends_within_the_stop_rules_tolerance():
    # The stop rule ends a level once two successive pose proposals are under its tolerance;
    # on exact data the default call's remaining error must be under that same tolerance.
    from tomojax.alignment import AlignConfig

    assert _exact_tier_ends(("gauge-free",))["gauge-free"] < AlignConfig().tolerance


@pytest.mark.numerical
def test_exact_tier_linear_kernel_is_the_projector_and_cubic_differs():
    import tomojax as tj
    from tomojax.geometry import stack_view_poses

    grid = tj.Grid(16, 16, 16, 1.0, 1.0, 1.0)
    detector = tj.Detector(20, 20, 1.0, 1.0)
    geometry = tj.ParallelGeometry(grid, detector, ANGLES[::4])
    nominal = np.asarray(stack_view_poses(geometry, len(ANGLES[::4])), np.float64)
    truth = np.random.default_rng(5).random((16, 16, 16)).astype(np.float32)
    poses = nominal.copy()
    poses[:, 0, 3] += 0.37  # a sub-pixel shift, where the two kernels disagree
    scan = tj.Scan(np.zeros((len(poses), 20, 20), np.float32), geometry)
    reference = tj.project(scan.with_poses(exact_tier.corrections(nominal, poses)).geometry, truth)
    linear = exact_tier.projections(grid, detector, truth, poses)
    cubic = exact_tier.projections(grid, detector, truth, poses, interpolation="cubic")
    np.testing.assert_allclose(linear, np.asarray(reference), rtol=1e-5, atol=1e-5)
    assert np.abs(cubic - linear).max() > 1e-2
