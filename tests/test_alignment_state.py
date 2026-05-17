from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align.model.dof_specs import ActiveParameterView, optimizer_step_stats
from tomojax.align.model.dofs import normalize_bounds
from tomojax.align.model.state import AlignmentState, PoseState, SetupGeometryState
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry
from tomojax.geometry import axis_unit_from_rotations


def _state() -> AlignmentState:
    params = jnp.asarray(
        [
            [0.01, 0.02, 0.03, 1.0, 2.0],
            [0.11, 0.12, 0.13, 3.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    setup = SetupGeometryState.from_degrees(
        det_u_px=-4.0,
        det_v_px=2.0,
        detector_roll_deg=5.0,
        axis_rot_x_deg=6.0,
        axis_rot_y_deg=-7.0,
    )
    return AlignmentState(setup=setup, pose=PoseState(params))


def test_setup_dof_packs_one_value_and_leaves_pose_frozen():
    state = _state()
    view = ActiveParameterView.from_dofs(("det_u_px",))

    packed = view.pack(state)
    updated = view.unpack(state, packed + jnp.asarray([1.0], dtype=jnp.float32))

    assert packed.shape == (1,)
    assert float(packed[0]) == pytest.approx(-4.0)
    assert float(updated.setup.det_u_px) == pytest.approx(-3.0)
    np.testing.assert_allclose(np.asarray(updated.pose.params5), np.asarray(state.pose.params5))


def test_pose_dofs_pack_columns_and_leave_setup_frozen():
    state = _state()
    view = ActiveParameterView.from_dofs(("alpha", "dx"))

    packed = view.pack(state)
    updated = view.unpack(state, packed + 1.0)

    expected = np.asarray([0.01 / 1e-3, 0.11 / 1e-3, 1.0 / 1e-1, 3.0 / 1e-1])
    np.testing.assert_allclose(np.asarray(packed), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(updated.pose.params5[:, 0]),
        np.asarray(state.pose.params5[:, 0] + 1e-3),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(updated.pose.params5[:, 3]),
        np.asarray(state.pose.params5[:, 3] + 1e-1),
        rtol=1e-6,
    )
    assert float(updated.setup.det_u_px) == pytest.approx(float(state.setup.det_u_px))


def test_mixed_active_view_uses_stable_registry_order():
    view = ActiveParameterView.from_dofs(("detector_roll_deg", "phi", "det_u_px"))

    assert view.dofs == ("phi", "det_u_px", "detector_roll_deg")
    assert view.active_pose_dofs == ("phi",)
    assert view.active_setup_dofs == ("det_u_px", "detector_roll_deg")


def test_degree_facing_setup_metadata_uses_radian_optimizer_state():
    state = AlignmentState(
        setup=SetupGeometryState.from_degrees(detector_roll_deg=5.0, axis_rot_x_deg=-3.0),
        pose=PoseState.zeros(1),
    )
    view = ActiveParameterView.from_dofs(("detector_roll_deg",))

    packed = view.pack(state)
    restored = view.unpack(state, jnp.asarray([6.0], dtype=jnp.float32))
    calibration = restored.to_calibration_state(active_dofs=view.dofs)
    variables = calibration.variables_by_name()

    assert float(state.setup.detector_roll_rad) == pytest.approx(np.deg2rad(5.0))
    assert float(packed[0]) == pytest.approx(5.0, rel=1e-6)
    assert variables["detector_roll_deg"].value == pytest.approx(6.0, abs=1e-5)
    assert variables["detector_roll_deg"].status == "estimated"
    assert variables["axis_rot_x_deg"].status == "frozen"


def test_calibration_state_exports_axis_unit_from_axis_rotations():
    setup = SetupGeometryState.from_degrees(
        axis_rot_x_deg=6.0,
        axis_rot_y_deg=-7.0,
        nominal_axis_unit=(0.0, 0.0, 1.0),
    )
    state = AlignmentState(setup=setup, pose=PoseState.zeros(1))

    calibration = state.to_calibration_state(active_dofs=("axis_rot_x_deg", "axis_rot_y_deg"))
    variables = calibration.variables_by_name()
    expected_axis = axis_unit_from_rotations(
        (0.0, 0.0, 1.0),
        axis_rot_x_deg=6.0,
        axis_rot_y_deg=-7.0,
    )

    np.testing.assert_allclose(
        variables["axis_unit_lab"].value,
        np.asarray(expected_axis),
        rtol=1e-6,
        atol=1e-6,
    )
    assert variables["axis_unit_lab"].status == "derived"


def test_tilt_alias_resolves_through_existing_geometry_rules():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="z",
    )

    view = ActiveParameterView.from_dofs(("tilt_deg",), geometry=geometry)

    assert view.dofs == ("axis_rot_y_deg",)


def test_tilt_bounds_constrain_laminography_z_active_axis():
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0)
    geometry = LaminographyGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 45.0],
        tilt_deg=30.0,
        tilt_about="z",
    )
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(1))
    view = ActiveParameterView.from_dofs(("tilt_deg",), geometry=geometry)
    bounds = normalize_bounds({"tilt_deg": (-2.0, 2.0)})

    lower, upper = view.bounds_whitened(state, bounds=bounds)

    assert view.dofs == ("axis_rot_y_deg",)
    np.testing.assert_allclose(np.asarray(lower), np.asarray([-2.0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(upper), np.asarray([2.0]), rtol=1e-6)


def test_tilt_bounds_reject_ambiguous_concrete_axis_view():
    state = AlignmentState(setup=SetupGeometryState(), pose=PoseState.zeros(1))
    view = ActiveParameterView.from_dofs(("axis_rot_x_deg", "axis_rot_y_deg"))
    bounds = normalize_bounds({"tilt_deg": (-2.0, 2.0)})

    with pytest.raises(ValueError, match="Ambiguous tilt_deg alignment bound"):
        view.bounds_whitened(state, bounds=bounds)


def test_optimizer_stats_report_whitened_and_native_motion():
    before = _state()
    view = ActiveParameterView.from_dofs(("det_u_px", "detector_roll_deg"))
    after = view.unpack(before, view.pack(before) + jnp.asarray([2.0, 3.0], dtype=jnp.float32))

    stats = optimizer_step_stats(view=view, before=before, after=after)

    assert stats["step_norm_whitened"] == pytest.approx(float(np.hypot(2.0, 3.0)))
    native = stats["step_by_dof_native_units"]
    assert native["det_u_px"] == pytest.approx(2.0)
    assert native["detector_roll_deg"] == pytest.approx(np.deg2rad(3.0), rel=1e-6)


def test_alignment_state_is_a_jax_pytree():
    state = _state()

    doubled = jax.tree_util.tree_map(
        lambda leaf: leaf * 2.0 if leaf is not None else None,
        state,
    )

    assert float(doubled.setup.det_u_px) == pytest.approx(-8.0)
    np.testing.assert_allclose(np.asarray(doubled.pose.params5), np.asarray(state.pose.params5 * 2))
