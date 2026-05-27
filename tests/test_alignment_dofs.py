from __future__ import annotations

from pathlib import Path

import numpy as np
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
from tomojax.align._model.schedules import GaugePolicyError, resolve_alignment_schedule
from tomojax.align.api import AlignConfig, save_alignment_checkpoint

# check-public-imports: allow-private
from tomojax.align.io.params_export import alignment_params_payload

# check-public-imports: allow-private
from tomojax.cli._align_checkpoint import checkpoint_metadata

# check-public-imports: allow-private
from tomojax.cli._align_command import build_parser

# check-public-imports: allow-private
from tomojax.cli._align_plan import build_align_cli_run_plan
from tomojax.geometry import Detector, Grid
from tomojax.io import ProjectionDataset


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
            _ = call()


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


def test_cli_geometry_dofs_route_to_multires(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--data", "input.nxs", "--optimise-dofs", "det_u_px"])
    dataset = ProjectionDataset(
        projections=np.zeros((3, 4, 5), dtype=np.float32),
        angles_deg=np.asarray([0.0, 90.0, 180.0], dtype=np.float32),
        detector=Detector(nu=5, nv=4, du=1.0, dv=1.0),
        grid=Grid(nx=5, ny=5, nz=4, vx=1.0, vy=1.0, vz=1.0),
    )

    def load_dataset(_: object) -> ProjectionDataset:
        return dataset

    monkeypatch.setattr("tomojax.cli._align_plan.load_projection_payload", load_dataset)

    plan = build_align_cli_run_plan(
        parser,
        args,
        {"explicit_cli_keys": [], "config_file_values": {}},
    )

    assert plan.run_levels == [1]
    assert plan.cfg.optimise_dofs == ("det_u_px",)


def test_cli_resume_restores_geometry_dofs_from_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = build_parser()
    checkpoint_path = tmp_path / "align.ckpt"
    dataset = ProjectionDataset(
        projections=np.zeros((3, 4, 5), dtype=np.float32),
        angles_deg=np.asarray([0.0, 90.0, 180.0], dtype=np.float32),
        detector=Detector(nu=5, nv=4, du=1.0, dv=1.0),
        grid=Grid(nx=5, ny=5, nz=4, vx=1.0, vy=1.0, vz=1.0),
    )

    def load_dataset(_: object) -> ProjectionDataset:
        return dataset

    monkeypatch.setattr("tomojax.cli._align_plan.load_projection_payload", load_dataset)
    initial_args = parser.parse_args(
        [
            "--data",
            "input.nxs",
            "--checkpoint",
            str(checkpoint_path),
            "--optimise-dofs",
            "det_u_px",
        ]
    )
    initial_plan = build_align_cli_run_plan(
        parser,
        initial_args,
        {"explicit_cli_keys": [], "config_file_values": {}},
    )
    save_alignment_checkpoint(
        checkpoint_path,
        x=np.zeros((5, 5, 4), dtype=np.float32),
        params5=np.zeros((3, 5), dtype=np.float32),
        metadata=checkpoint_metadata(
            meta=initial_plan.meta,
            projections=initial_plan.projections,
            cfg=initial_plan.cfg,
            command=initial_plan.command,
            recon_grid=initial_plan.recon_grid,
            detector=initial_plan.detector,
            state_grid=initial_plan.recon_grid,
            state_detector=initial_plan.detector,
            gather_dtype=initial_plan.gather_dtype,
            levels=initial_plan.run_levels,
            level_index=0,
            level_factor=1,
            completed_outer_iters_in_level=0,
            global_outer_iters_completed=0,
            prev_factor=None,
            L_prev=None,
            small_impr_streak=0,
            elapsed_offset=0.0,
            level_complete=False,
            run_complete=False,
            schedule_metadata=initial_plan.schedule_metadata,
        ),
    )

    resume_args = parser.parse_args(["--data", "input.nxs", "--resume", str(checkpoint_path)])
    resume_plan = build_align_cli_run_plan(
        parser,
        resume_args,
        {"explicit_cli_keys": [], "config_file_values": {}, "effective_options": vars(resume_args)},
    )

    assert resume_plan.run_levels == [1]
    assert resume_plan.cfg.optimise_dofs == ("det_u_px",)


def test_cli_pose_only_dofs_stay_single_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = build_parser()
    args = parser.parse_args(["--data", "input.nxs", "--optimise-dofs", "dx"])
    dataset = ProjectionDataset(
        projections=np.zeros((3, 4, 5), dtype=np.float32),
        angles_deg=np.asarray([0.0, 90.0, 180.0], dtype=np.float32),
        detector=Detector(nu=5, nv=4, du=1.0, dv=1.0),
        grid=Grid(nx=5, ny=5, nz=4, vx=1.0, vy=1.0, vz=1.0),
    )

    def load_dataset(_: object) -> ProjectionDataset:
        return dataset

    monkeypatch.setattr("tomojax.cli._align_plan.load_projection_payload", load_dataset)

    plan = build_align_cli_run_plan(
        parser,
        args,
        {"explicit_cli_keys": [], "config_file_values": {}},
    )

    assert plan.run_levels is None


def test_cli_alignment_defaults_to_per_view_pose() -> None:
    parser = build_parser()
    args = parser.parse_args(["--data", "input.nxs", "--out", "aligned.nxs"])

    assert args.mode == "pose"
    assert args.pose_model == "per_view"


def test_align_config_defaults_to_per_view_pose_model() -> None:
    assert AlignConfig().pose_model == "per_view"


def test_direct_mixed_dofs_explain_gauge_policy() -> None:
    with pytest.raises(GaugePolicyError, match="--gauge-policy anchor_mean"):
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
