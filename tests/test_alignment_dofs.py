from __future__ import annotations

import pytest

# check-public-imports: allow-private
from tomojax.align._geometry.geometry_blocks import normalize_geometry_dofs

# check-public-imports: allow-private
from tomojax.align._model.dof_specs import ActiveParameterView, dof_spec

# check-public-imports: allow-private
from tomojax.align._model.dofs import (
    GEOMETRY_DOF_NAMES,
    normalize_alignment_dofs,
    normalize_bounds,
    resolve_scoped_alignment_dofs,
)

# check-public-imports: allow-private
from tomojax.align._model.schedules import resolve_alignment_schedule
from tomojax.align.api import AlignConfig


def test_setup_dofs_use_canonical_axis_names() -> None:
    assert "tilt_deg" not in GEOMETRY_DOF_NAMES
    assert normalize_alignment_dofs("axis_rot_x_deg,axis_rot_y_deg") == (
        "axis_rot_x_deg",
        "axis_rot_y_deg",
    )
    assert normalize_geometry_dofs(("det_u_px", "axis_rot_x_deg")) == (
        "det_u_px",
        "axis_rot_x_deg",
    )


def test_tilt_deg_is_not_a_supported_dof() -> None:
    for call in (
        lambda: normalize_alignment_dofs("tilt_deg"),
        lambda: normalize_geometry_dofs(("tilt_deg",)),
        lambda: normalize_bounds("tilt_deg=-1:1"),
        lambda: ActiveParameterView.from_dofs(("tilt_deg",)),
        lambda: dof_spec("tilt_deg"),
    ):
        with pytest.raises(ValueError, match="tilt_deg"):
            call()


def test_setup_geometry_is_selected_by_optimise_dofs() -> None:
    resolved = resolve_scoped_alignment_dofs(optimise_dofs=("det_u_px", "axis_rot_x_deg"))

    assert resolved.active_pose_dofs == ()
    assert resolved.active_geometry_dofs == ("det_u_px", "axis_rot_x_deg")
    assert resolved.active_dofs == ("det_u_px", "axis_rot_x_deg")


def test_geometry_dofs_is_not_a_public_setup_input() -> None:
    with pytest.raises(TypeError, match="geometry_dofs"):
        AlignConfig(geometry_dofs=("det_u_px",))  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="geometry_dofs"):
        resolve_alignment_schedule(geometry_dofs=("det_u_px",))  # type: ignore[call-arg]
