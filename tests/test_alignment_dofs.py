from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from tomojax.alignment.api import (
    AlignConfig,
    GaugePolicyError,
    alignment_params_payload,
    dof_spec,
    normalize_alignment_dofs,
    normalize_bounds,
    normalize_geometry_dofs,
    resolve_alignment_schedule,
)
from tomojax.cli.align import build_parser


def test_setup_dofs_use_canonical_axis_names() -> None:
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
        lambda: dof_spec("tilt_deg"),
    ):
        with pytest.raises(ValueError, match="tilt_deg"):
            _ = call()


def test_setup_geometry_is_selected_by_optimise_dofs() -> None:
    resolved = resolve_alignment_schedule(optimise_dofs=("det_u_px", "axis_rot_x_deg"))

    assert resolved.active_pose_dofs == ()
    assert resolved.active_geometry_dofs == ("det_u_px", "axis_rot_x_deg")
    assert resolved.active_dofs == ("det_u_px", "axis_rot_x_deg")


def test_geometry_dofs_is_not_a_public_setup_input() -> None:
    with pytest.raises(TypeError, match="geometry_dofs"):
        AlignConfig(geometry_dofs=("det_u_px",))  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="geometry_dofs"):
        resolve_alignment_schedule(geometry_dofs=("det_u_px",))  # type: ignore[call-arg]


def test_cli_alignment_defaults_to_per_view_pose() -> None:
    parser = build_parser()
    args = parser.parse_args(["input.nxs", "-o", "aligned.nxs"])

    assert args.mode == "pose"


def test_align_config_defaults_to_per_view_pose_model() -> None:
    assert AlignConfig().pose_model == "per_view"


@pytest.mark.parametrize("quality", ["fast", "reference"])
def test_alignment_qualities_size_reconstruction_batches_automatically(
    quality: Literal["fast", "reference"],
) -> None:
    import importlib

    from tomojax.geometry import Detector, Grid

    # check-public-imports: allow-private
    loop = importlib.import_module("tomojax.alignment._pose._pose_loop")
    cfg = AlignConfig(quality=quality)
    assert cfg.views_per_batch == 0
    grid, detector = Grid(8, 8, 8, 1.0, 1.0, 1.0), Detector(8, 8, 1.0, 1.0)
    resolved = loop._with_resolved_views_per_batch(cfg, n_views=12, grid=grid, detector=detector)
    assert 1 <= resolved.views_per_batch <= 12
    assert cfg.views_per_batch == 0
    cfg.views_per_batch = 3
    kept = loop._with_resolved_views_per_batch(cfg, n_views=12, grid=grid, detector=detector)
    assert kept is cfg and kept.views_per_batch == 3


def test_direct_mixed_dofs_explain_gauge_policy() -> None:
    with pytest.raises(GaugePolicyError, match='gauge_policy = "anchor_mean" in a --config file'):
        resolve_alignment_schedule(
            optimise_dofs=("alpha", "det_u_px"),
            gauge_policy="reject",
        )


def test_alignment_params_export_unwraps_object_dtype_scalars() -> None:
    payload = alignment_params_payload(
        np.zeros((1, 5), dtype=np.float32),
        du=1.0,
        dv=1.0,
        gauge_metadata={
            "mode": np.array("mean_translation", dtype=object),
            "note": np.array(None, dtype=object),
        },
    )

    assert payload["gauge_fix"] == {"mode": "mean_translation", "note": None}


def test_coupled_pose_config_matches_the_cli_pose_solver() -> None:
    from tomojax.alignment.api import L2LossSpec, coupled_pose_config

    cfg = coupled_pose_config(outer_iterations=7)
    assert (cfg.gn_coupling, cfg.gn_joint_solver, cfg.ray_integrator) == (
        "joint",
        "pose_eliminated",
        "joseph",
    )
    assert isinstance(cfg.loss, L2LossSpec)
    assert (cfg.tv_weight, cfg.gather_dtype, cfg.outer_iterations) == (0.0, "fp32", 7)
