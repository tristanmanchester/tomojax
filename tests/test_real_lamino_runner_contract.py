# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "run_real_lamino_reference_regression.py"
)
_REPORT_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "summarize_real_lamino_report.py"
)
_STAGED_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "run_real_lamino_staged.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_real_lamino_reference_regression",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_REPORT_SPEC = importlib.util.spec_from_file_location(
    "summarize_real_lamino_report",
    _REPORT_SCRIPT_PATH,
)
assert _REPORT_SPEC is not None
assert _REPORT_SPEC.loader is not None
report_runner = importlib.util.module_from_spec(_REPORT_SPEC)
_REPORT_SPEC.loader.exec_module(report_runner)

_STAGED_SPEC = importlib.util.spec_from_file_location(
    "run_real_lamino_staged",
    _STAGED_SCRIPT_PATH,
)
assert _STAGED_SPEC is not None
assert _STAGED_SPEC.loader is not None
staged_runner = importlib.util.module_from_spec(_STAGED_SPEC)
_STAGED_SPEC.loader.exec_module(staged_runner)


def test_runner_input_argument_is_required(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--out", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        runner._parse_args()

    assert exc_info.value.code == 2


def test_runner_expected_projection_shape_argument_is_configurable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--expected-projection-shape",
            "3x4x5",
        ],
    )

    args = runner._parse_args()

    assert args.input == "input.nxs"
    assert args.expected_projection_shape == (3, 4, 5)


def test_runner_defaults_to_explicit_lightning_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
        ],
    )

    args = runner._parse_args()
    cfg = runner._make_cfg(args, active_pose=("phi",))

    assert args.align_profile == "lightning"
    assert args.projector_backend == "pallas"
    assert args.regulariser == "huber_tv"
    assert args.quality_tier == "fast"
    assert args.fallback_policy == "fallback"
    assert args.pose_model == "spline"
    assert args.knot_spacing == 8
    assert args.pose_degree == 3
    assert args.views_per_batch == 0
    assert cfg.align_profile == "lightning"
    assert cfg.projector_backend == "pallas"
    assert cfg.regulariser == "huber_tv"
    assert cfg.quality_tier == "fast"
    assert cfg.fallback_policy == "fallback"
    assert cfg.fold_rigid_detector_grid is True
    assert cfg.pose_model == "spline"
    assert cfg.knot_spacing == 8
    assert cfg.degree == 3


def test_runner_accepts_real_pose_model_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--pose-model",
            "polynomial",
            "--knot-spacing",
            "5",
            "--pose-degree",
            "2",
        ],
    )

    args = runner._parse_args()
    cfg = runner._make_cfg(args, active_pose=("phi",))

    assert cfg.pose_model == "polynomial"
    assert cfg.knot_spacing == 5
    assert cfg.degree == 2


def test_runner_accepts_tortoise_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--align-profile",
            "tortoise",
            "--projector-backend",
            "jax",
            "--regulariser",
            "tv",
            "--quality-tier",
            "reference",
            "--views-per-batch",
            "1",
            "--gather-dtype",
            "fp32",
        ],
    )

    args = runner._parse_args()
    cfg = runner._make_cfg(args, active_pose=("phi",))

    assert cfg.align_profile == "tortoise"
    assert cfg.projector_backend == "jax"
    assert cfg.regulariser == "tv"
    assert cfg.quality_tier == "reference"
    assert cfg.gather_dtype == "fp32"
    assert cfg.views_per_batch == 1


def test_runner_smoke_reduces_real_data_workload(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--outer-iters",
            "8",
            "--recon-iters",
            "40",
            "--tv-prox-iters",
            "16",
            "--smoke",
        ],
    )

    args = runner._parse_args()

    assert args.levels_setup == [8]
    assert args.levels_phi == [8]
    assert args.levels_dx_dz == [8]
    assert args.levels_polish == [8]
    assert args.slab_nz == 48
    assert args.outer_iters == 1
    assert args.recon_iters == 3
    assert args.tv_prox_iters == 2
    assert args.views_per_batch == 16


def test_runner_can_request_canonical_detector_grid_diagnostic(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--canonical-det-grid",
        ],
    )

    args = runner._parse_args()

    assert args.canonical_det_grid is True


def test_status_stage_update_clears_stale_message(tmp_path) -> None:
    path = tmp_path / "status.json"

    runner._status(path, state="running", stage="00_baseline", message="baseline_fbp")
    runner._status(path, state="running", stage="01_setup_geometry")

    payload = json.loads(path.read_text())
    assert payload["stage"] == "01_setup_geometry"
    assert "message" not in payload


def test_validate_loaded_input_accepts_derived_shape_without_expected_contract() -> None:
    projections = np.zeros((3, 4, 5), dtype=np.float32)
    thetas = np.arange(3, dtype=np.float32)

    got_projections, got_thetas = runner._validate_loaded_input(
        projections,
        thetas,
        expected_projection_shape=None,
    )

    assert got_projections.shape == (3, 4, 5)
    assert got_thetas.shape == (3,)


def test_validate_loaded_input_rejects_configured_shape_mismatch() -> None:
    projections = np.zeros((3, 4, 5), dtype=np.float32)
    thetas = np.arange(3, dtype=np.float32)

    with pytest.raises(ValueError, match=r"expected \(3, 4, 6\).*--expected-projection-shape"):
        runner._validate_loaded_input(
            projections,
            thetas,
            expected_projection_shape=(3, 4, 6),
        )


def test_validate_loaded_input_rejects_theta_count_mismatch() -> None:
    projections = np.zeros((3, 4, 5), dtype=np.float32)
    thetas = np.arange(2, dtype=np.float32)

    with pytest.raises(ValueError, match="theta count must match projections n_views"):
        runner._validate_loaded_input(
            projections,
            thetas,
            expected_projection_shape=None,
        )


def test_real_lamino_summary_uses_final_vs_cor_only_quality_contract(tmp_path) -> None:
    run_dir = _write_minimal_real_lamino_run(tmp_path)

    summary = report_runner.build_real_lamino_report(run_dir, out_dir=tmp_path / "report")

    assert summary["schema"] == "tomojax.real_lamino_staged_report.v2"
    assert summary["success"]["passed"] is True
    assert summary["quality_basis"]["truth_metrics"] == "not_applicable_real_data"
    assert summary["reconstruction_comparison"]["loss_improvement_abs"] == 20.0
    assert (tmp_path / "report" / "real_lamino_summary.json").exists()
    assert (tmp_path / "report" / "real_lamino_residual_trace.csv").exists()
    assert (tmp_path / "report" / "real_lamino_geometry_trace.json").exists()
    assert (tmp_path / "report" / "publication" / "before_orthos.png").exists()
    assert (tmp_path / "report" / "publication" / "cor_only_orthos.png").exists()
    assert (tmp_path / "report" / "publication" / "full_orthos.png").exists()


def test_real_lamino_summary_can_require_quality_success(tmp_path) -> None:
    run_dir = _write_minimal_real_lamino_run(tmp_path, final_last_loss=130.0)

    with pytest.raises(RuntimeError, match="did not improve"):
        report_runner.build_real_lamino_report(
            run_dir,
            out_dir=tmp_path / "report",
            require_success=True,
        )


def test_staged_smoke_reduces_to_det_u_and_cor_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--outer-iters",
            "8",
            "--recon-iters",
            "40",
            "--tv-prox-iters",
            "16",
            "--smoke",
        ],
    )

    args = staged_runner._parse_args()

    assert args.levels_setup == [8]
    assert args.outer_iters == 1
    assert args.recon_iters == 3
    assert args.tv_prox_iters == 2
    assert args.views_per_batch == 1
    assert args.bin_factor == 4


def test_staged_public_help_uses_clean_profile_names(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        staged_runner._parse_args()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "staged-lamino" in captured.out
    assert "reference-regression" in captured.out
    assert "diagnostic-fast" in captured.out
    assert "staged_lamino" not in captured.out
    assert "reference_regression_audit" not in captured.out
    assert "--v1-parity-real-lamino" not in captured.out
    for forbidden in ("mvp", "v1", "parity", "cor_mvp", "full_mvp", "smoke"):
        assert forbidden not in captured.out.lower()


def test_staged_accepts_explicit_binned_smoke_shape(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--bin-factor",
            "2",
            "--smoke-shape",
            "16x64x64",
        ],
    )

    args = staged_runner._parse_args()

    assert args.bin_factor == 2
    assert args.smoke_shape == (16, 64, 64)


def test_staged_accepts_final_candidate_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--final-candidate-policy",
            "last_valid",
        ],
    )

    args = staged_runner._parse_args()

    assert args.final_candidate_policy == "last_valid"


def test_staged_reference_regression_profile_forces_reference_contract(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--profile",
            "reference-regression",
            "--pose-bounds-profile",
            "reference_conservative",
            "--final-candidate-policy",
            "all",
            "--outer-iters",
            "1",
        ],
    )

    args = staged_runner._parse_args()

    assert args.full_staged is True
    assert args.pose_model == "per_view"
    assert args.pose_bounds_profile == "wide"
    assert args.final_candidate_policy == "last_valid"
    assert args.fold_rigid_detector_grid is False
    assert args.outer_iters == 8
    assert args.levels_phi == [4, 2, 1]
    cfg = runner._make_cfg(args, active_pose=("phi",))
    assert cfg.fold_rigid_detector_grid is False
    assert staged_runner._pose_phi_bounds(args) == "phi=-0.0872665:0.0872665"
    assert staged_runner._pose_dx_dz_bounds(args) == "dx=-16:16,dz=-16:16"


def test_staged_staged_lamino_profile_forces_winning_contract(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--profile",
            "staged-lamino",
            "--pose-model",
            "spline",
            "--final-candidate-policy",
            "all",
            "--outer-iters",
            "1",
        ],
    )

    args = staged_runner._parse_args()

    assert args.profile == "staged-lamino"
    assert args.reference_regression is False
    assert args.full_staged is True
    assert args.pose_model == "per_view"
    assert args.pose_bounds_profile == "wide"
    assert args.final_candidate_policy == "last_valid"
    assert args.fold_rigid_detector_grid is False
    assert args.outer_iters == 8
    assert args.levels_setup == [8, 4, 2]
    assert args.levels_phi == [4, 2, 1]
    assert args.levels_dx_dz == [4, 2, 1]
    assert args.levels_polish == [2, 1]
    cfg = runner._make_cfg(args, active_pose=("phi",))
    assert cfg.fold_rigid_detector_grid is False
    assert staged_runner._pose_phi_bounds(args) == "phi=-0.0872665:0.0872665"
    assert staged_runner._pose_dx_dz_bounds(args) == "dx=-16:16,dz=-16:16"


def test_staged_legacy_reference_flag_is_hidden_profile_alias(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--v1-parity-real-lamino",
        ],
    )

    args = staged_runner._parse_args()

    assert args.profile == "reference-regression"
    assert args.reference_regression is True


def test_reference_regression_level_outer_counts_replay_reference_stage_summary(
    monkeypatch,
    tmp_path,
) -> None:
    reference_run = tmp_path / "reference_run"
    summary_path = reference_run / "01_setup_geometry" / "03_axis_direction" / "stage_summary.csv"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        "stage,level_factor,outer_iter,geometry_loss_before,geometry_loss_after\n"
        "01_setup_geometry/03_axis_direction,8,1,2.0,1.5\n"
        "01_setup_geometry/03_axis_direction,8,2,1.5,1.4\n"
        "01_setup_geometry/03_axis_direction,4,1,4.0,3.0\n",
        encoding="utf-8",
    )
    reference_report = reference_run / "real_lamino_report" / "real_lamino_summary.json"
    reference_report.parent.mkdir(parents=True)
    _write_json(reference_report, {"success": {"passed": True}})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path / "out"),
            "--profile",
            "reference-regression",
            "--reference-report",
            str(reference_report),
        ],
    )

    args = staged_runner._parse_args()

    assert staged_runner._reference_regression_level_outer_counts(
        args,
        stage_name="01_setup_geometry/03_axis_direction",
    ) == {8: 2, 4: 1}


def test_staged_diagnostic_fast_profile_uses_bounded_smoke(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--profile",
            "diagnostic-fast",
            "--recon-iters",
            "40",
        ],
    )

    args = staged_runner._parse_args()

    assert args.profile == "diagnostic-fast"
    assert args.full_staged is True
    assert args.smoke is True
    assert args.bin_factor == 4
    assert args.recon_iters == 3
    assert args.final_candidate_policy == "last_valid"


def test_staged_accepts_real_pose_model_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
            "--pose-model",
            "spline",
            "--knot-spacing",
            "6",
            "--pose-degree",
            "2",
        ],
    )

    args = staged_runner._parse_args()

    assert args.pose_model == "spline"
    assert args.knot_spacing == 6
    assert args.pose_degree == 2


def test_v2_binned_fixture_scales_geometry_and_records_provenance() -> None:
    args = SimpleNamespace(
        slab_nz=4,
        slab_center_z=4,
        preview_z=4,
        stack_z_range="2:6",
        bin_factor=2,
        smoke_shape=None,
    )
    raw = np.arange(5 * 8 * 8, dtype=np.float32).reshape(5, 8, 8)
    thetas = np.linspace(0.0, 180.0, 5, endpoint=False, dtype=np.float32)

    working_raw, working_thetas, geometry_inputs, provenance = (
        staged_runner._prepare_binned_fixture(
            args,
            native=runner,
            raw_projections=raw,
            thetas=thetas,
        )
    )

    grid = geometry_inputs["grid"]
    detector = geometry_inputs["detector"]
    assert working_raw.shape == (5, 4, 4)
    assert working_thetas.shape == (5,)
    assert grid.nx == 4
    assert grid.nz == 2
    assert grid.vx == 2.0
    assert detector.nu == 4
    assert detector.nv == 4
    assert detector.du == 2.0
    assert geometry_inputs["full_nz"] == 8
    assert args.slab_nz == 2
    assert args.preview_z == 4
    assert args.stack_z_range == "2:6"
    assert provenance["enabled"] is True
    assert provenance["effective_bin_factor"] == 2
    assert provenance["original_projection_shape"] == [5, 8, 8]
    assert provenance["working_projection_shape"] == [5, 4, 4]
    assert provenance["coordinate_full_nz"] == 8
    assert provenance["working_detector_nz"] == 4
    assert provenance["detector_shift_bound_scale"] == 0.5


def test_v2_binned_fixture_smoke_shape_subselects_views_and_raises_factor() -> None:
    args = SimpleNamespace(
        slab_nz=8,
        slab_center_z=8,
        preview_z=8,
        stack_z_range="6:10",
        bin_factor=1,
        smoke_shape=(3, 4, 4),
    )
    raw = np.zeros((7, 16, 16), dtype=np.float32)
    thetas = np.arange(7, dtype=np.float32)

    working_raw, working_thetas, _geometry_inputs, provenance = (
        staged_runner._prepare_binned_fixture(
            args,
            native=runner,
            raw_projections=raw,
            thetas=thetas,
        )
    )

    assert working_raw.shape == (3, 4, 4)
    assert working_thetas.tolist() == [0.0, 3.0, 6.0]
    assert provenance["effective_bin_factor"] == 4
    assert provenance["view_indices"] == [0, 3, 6]
    assert args.effective_bin_factor == 4


def test_v2_stage_validation_accepts_repo_relative_artifact_paths(tmp_path) -> None:
    stage_dir = tmp_path / "run" / "01_setup_geometry" / "02_detector_roll"
    stage_dir.mkdir(parents=True)
    artifact = stage_dir / "orthos.png"
    artifact.write_bytes(b"png")
    _write_json(
        stage_dir / "stage_manifest.json",
        {
            "artifacts": {
                "orthos": str(artifact),
            }
        },
    )

    assert staged_runner._artifact_validation_failures(stage_dir) == []


def test_v2_pose_stage_validation_accepts_finite_fast_profile_losses() -> None:
    failures = staged_runner._stat_validation_failures(
        [
            {
                "loss_before": 2.0,
                "loss_after": 1.0,
                "data_loss_computed": False,
            }
        ],
        require_data_loss=True,
    )

    assert failures == []


def test_staged_runtime_default_streams_fista(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
        ],
    )

    args = staged_runner._parse_args()
    assert args.views_per_batch == 0

    staged_runner._normalize_runtime_args(args)

    assert args.views_per_batch == 1


def test_staged_defaults_to_reference_conservative_pose_bounds(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--input",
            "input.nxs",
            "--out",
            str(tmp_path),
        ],
    )

    args = staged_runner._parse_args()

    assert args.pose_bounds_profile == "reference_conservative"
    assert staged_runner._pose_phi_bounds(args) == "phi=-0.00872665:0.00872665"
    assert staged_runner._pose_dx_dz_bounds(args) == "dx=-10:10,dz=-10:10"
    assert "alpha=-0.00872665:0.00872665" in staged_runner._pose_polish_bounds(args)

    args.effective_bin_factor = 4
    assert staged_runner._setup_det_u_bounds(args) == "det_u_px=-6:6"
    assert staged_runner._pose_dx_dz_bounds(args) == "dx=-2.5:2.5,dz=-2.5:2.5"


def test_v2_cor_real_lamino_report_preserves_partial_contract(tmp_path) -> None:
    run_dir = _write_minimal_staged_run(tmp_path)

    summary = staged_runner.build_real_lamino_staged_report(
        run_dir,
        out_dir=tmp_path / "v2_report",
        reference_report=Path("runs/reference/real_lamino_report/real_lamino_summary.json"),
    )

    assert summary["schema"] == "tomojax.real_lamino_staged_report.v2"
    assert summary["contract_compatible_with"] == "tomojax.real_lamino_staged_report.v2"
    assert summary["success"]["passed"] is True
    assert summary["success"]["full_staged_success_deferred"] is True
    assert summary["quality_basis"]["truth_metrics"] == "not_applicable_real_data"
    assert summary["method_constraints"]["cor_grid_search_added"] is False
    assert summary["method_constraints"]["sinogram_or_correlation_method_added"] is False
    assert summary["method_constraints"]["sharpness_or_autofocus_method_added"] is False
    statuses = {record["stage"]: record["status"] for record in summary["staged_path"]}
    assert statuses["00_baseline"] == "completed"
    assert statuses["01_setup_geometry/01_cor"] == "completed"
    assert statuses["06_cor_only_fista"] == "completed"
    assert statuses["01_setup_geometry/02_detector_roll"] == "planned"
    assert statuses["05_final"] == "planned"
    assert summary["reconstruction_comparison"]["cor_only"]["loss"]["last"] == 80.0
    assert (tmp_path / "v2_report" / "real_lamino_summary.json").exists()
    assert (tmp_path / "v2_report" / "real_lamino_residual_trace.csv").exists()
    assert (tmp_path / "v2_report" / "real_lamino_geometry_trace.json").exists()
    assert (tmp_path / "v2_report" / "publication" / "before_orthos.png").exists()
    assert (tmp_path / "v2_report" / "publication" / "cor_only_orthos.png").exists()


def test_v2_full_real_lamino_report_fails_when_final_is_worse_than_cor_only(tmp_path) -> None:
    run_dir = _write_minimal_staged_run(tmp_path)
    final_dir = run_dir / "05_final"
    _write_stage_images(final_dir)
    _write_json(
        final_dir / "stage_manifest.json",
        {
            "stage": "05_final",
            "status": "completed",
            "active_dofs": ["detector_roll", "axis_direction", "pose_5dof"],
            "volume_shape": [256, 256, 96],
            "artifacts": {
                "orthos": "orthos.png",
                "aligned_xy": "aligned_xy_global_z209.png",
                "delta_xy": "delta_xy_global_z209.png",
                "z_stack": "z_stack_global_z198_220.png",
            },
            "geometry_calibration_state": {"det_u_px": 1.25},
            "recon_info": {"loss": [140.0, 120.0], "effective_iters": 2, "regulariser": "tv"},
        },
    )
    for rel in [
        "01_setup_geometry/02_detector_roll",
        "01_setup_geometry/03_axis_direction",
        "02_pose_phi",
        "03_pose_dx_dz",
        "04_pose_polish",
    ]:
        payload = json.loads((run_dir / rel / "stage_manifest.json").read_text())
        payload["status"] = "completed"
        payload["planned_after"] = None
        _write_json(run_dir / rel / "stage_manifest.json", payload)

    summary = staged_runner.build_real_lamino_staged_report(
        run_dir,
        out_dir=tmp_path / "v2_full_report",
    )

    assert summary["success"]["phase"] == "v2_full_staged"
    assert summary["success"]["passed"] is False
    assert summary["success"]["final_loss"] == 120.0
    assert summary["success"]["cor_only_loss"] == 80.0
    assert summary["success"]["loss_improvement_abs"] == -40.0
    assert "did not improve" in summary["success"]["reason"]
    assert (tmp_path / "v2_full_report" / "publication" / "full_orthos.png").exists()


def test_v2_final_reconstruction_selects_lowest_loss_candidate(tmp_path) -> None:
    class FakeContext:
        def __init__(self, root: Path) -> None:
            self.run_root = root
            self.stage_dir = lambda name: root / name

    class FakeNative:
        def _final_reconstruct(self, ctx, **kwargs):
            stage_dir = ctx.stage_dir("05_final")
            stage_dir.mkdir(parents=True, exist_ok=True)
            loss = float(np.asarray(kwargs["params5"])[0, 0])
            volume = np.full((2, 2, 1), loss, dtype=np.float32)
            _write_json(
                stage_dir / "stage_manifest.json",
                {
                    "stage": "05_final",
                    "status": "completed",
                    "recon_info": {"loss": [loss]},
                },
            )
            return volume

        def _write_json(self, path: Path, payload: dict[str, object]) -> None:
            _write_json(path, payload)

    ctx = FakeContext(tmp_path / "run")
    ctx.run_root.mkdir()
    candidates = [
        {
            "label": "worse",
            "source_stage": "04_pose_polish",
            "setup_state": object(),
            "params5": np.asarray([[10.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
        {
            "label": "better",
            "source_stage": "01_setup_geometry/03_axis_direction",
            "setup_state": object(),
            "params5": np.asarray([[4.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
    ]

    volume, choice = staged_runner.run_best_final_reconstruction(
        ctx,
        native=FakeNative(),
        geometry=None,
        grid=None,
        detector=None,
        projections=np.zeros((1, 1, 1), dtype=np.float32),
        full_nz=1,
        candidates=candidates,
    )

    manifest = json.loads((ctx.run_root / "05_final" / "stage_manifest.json").read_text())
    assert choice["label"] == "better"
    assert float(volume[0, 0, 0]) == 4.0
    assert manifest["volume_shape"] == [2, 2, 1]
    assert manifest["selected_final_candidate"]["label"] == "better"
    assert [item["label"] for item in manifest["selected_final_candidate"]["candidates"]] == [
        "worse",
        "better",
    ]
    assert manifest["selected_final_candidate"]["candidate_policy"] == "all"


def test_v2_final_reconstruction_can_score_only_last_valid_candidate(tmp_path) -> None:
    class FakeContext:
        def __init__(self, root: Path) -> None:
            self.run_root = root
            self.args = SimpleNamespace(final_candidate_policy="last_valid")
            self.stage_dir = lambda name: root / name

    class FakeNative:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def _final_reconstruct(self, ctx, **kwargs):
            stage_dir = ctx.stage_dir("05_final")
            stage_dir.mkdir(parents=True, exist_ok=True)
            loss = float(np.asarray(kwargs["params5"])[0, 0])
            self.calls.append(loss)
            volume = np.full((2, 2, 1), loss, dtype=np.float32)
            _write_json(
                stage_dir / "stage_manifest.json",
                {
                    "stage": "05_final",
                    "status": "completed",
                    "recon_info": {"loss": [loss]},
                },
            )
            return volume

        def _write_json(self, path: Path, payload: dict[str, object]) -> None:
            _write_json(path, payload)

    ctx = FakeContext(tmp_path / "run")
    ctx.run_root.mkdir()
    native = FakeNative()
    candidates = [
        {
            "label": "early",
            "source_stage": "01_setup_geometry/01_cor",
            "setup_state": object(),
            "params5": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
        {
            "label": "last",
            "source_stage": "04_pose_polish",
            "setup_state": object(),
            "params5": np.asarray([[9.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
    ]

    volume, choice = staged_runner.run_best_final_reconstruction(
        ctx,
        native=native,
        geometry=None,
        grid=None,
        detector=None,
        projections=np.zeros((1, 1, 1), dtype=np.float32),
        full_nz=1,
        candidates=candidates,
    )

    manifest = json.loads((ctx.run_root / "05_final" / "stage_manifest.json").read_text())
    assert native.calls == [9.0]
    assert choice["label"] == "last"
    assert float(volume[0, 0, 0]) == 9.0
    assert manifest["selected_final_candidate"]["candidate_policy"] == "last_valid"
    assert [item["label"] for item in manifest["selected_final_candidate"]["candidates"]] == [
        "last"
    ]


def test_v2_pose_stage_validation_fails_closed_on_nan_volume(tmp_path) -> None:
    class FakeCalibrationState:
        def __init__(self, label: str) -> None:
            self.label = label

        def to_calibration_state(self):
            return self

        def to_dict(self) -> dict[str, str]:
            return {"label": self.label}

    class FakeContext:
        def __init__(self, root: Path) -> None:
            self.run_root = root
            self.args = SimpleNamespace(
                levels_setup=[1],
                levels_phi=[1],
                levels_dx_dz=[1],
                levels_polish=[1],
                pose_bounds_profile="reference_conservative",
            )

        def stage_dir(self, name: str) -> Path:
            return self.run_root / name

    class FakeNative:
        def _write_json(self, path: Path, payload: dict[str, object]) -> None:
            _write_json(path, payload)

        def _params_summary(self, params: np.ndarray) -> dict[str, object]:
            return {"phi": {"std": float(np.std(params[:, 2]))}}

        def run_setup_stage(self, _ctx, *, stage_dir: Path, stage_name: str, **_kwargs):
            stage_dir.mkdir(parents=True, exist_ok=True)
            _write_stage_images(stage_dir)
            _write_json(
                stage_dir / "stage_manifest.json",
                {
                    "stage": stage_name,
                    "status": "completed",
                    "active_dofs": [],
                    "artifacts": {
                        "orthos": "orthos.png",
                        "aligned_xy": "aligned_xy_global_z209.png",
                        "delta_xy": "delta_xy_global_z209.png",
                        "z_stack": "z_stack_global_z198_220.png",
                    },
                },
            )
            x = np.ones((2, 2, 1), dtype=np.float32)
            return x, FakeCalibrationState(stage_name), [
                {"geometry_loss_before": 2.0, "geometry_loss_after": 1.0}
            ]

        def run_pose_stage(self, _ctx, *, stage_dir: Path, stage_name: str, params5, **_kwargs):
            stage_dir.mkdir(parents=True, exist_ok=True)
            _write_stage_images(stage_dir)
            (stage_dir / "checkpoints").mkdir()
            x = np.full((2, 2, 1), np.nan, dtype=np.float32)
            np.savez(stage_dir / "checkpoints" / "outer_001_level01_iter01.npz", x=x)
            _write_json(
                stage_dir / "stage_manifest.json",
                {
                    "stage": stage_name,
                    "status": "completed",
                    "active_dofs": ["phi"],
                    "artifacts": {
                        "orthos": "orthos.png",
                        "aligned_xy": "aligned_xy_global_z209.png",
                        "delta_xy": "delta_xy_global_z209.png",
                        "z_stack": "z_stack_global_z198_220.png",
                    },
                },
            )
            return x, params5, [
                {
                    "loss_before": "nan",
                    "loss_after": "nan",
                    "data_loss_computed": False,
                }
            ]

    ctx = FakeContext(tmp_path / "run")
    ctx.run_root.mkdir()
    setup_state, params5, records, candidates = staged_runner.run_remaining_stages(
        ctx,
        native=FakeNative(),
        geometry=None,
        grid=None,
        detector=None,
        projections=np.zeros((1, 1, 1), dtype=np.float32),
        full_nz=1,
        setup_state=FakeCalibrationState("cor"),
        params5=np.zeros((1, 5), dtype=np.float32),
    )

    phi_manifest = json.loads((ctx.run_root / "02_pose_phi" / "stage_manifest.json").read_text())
    dx_manifest = json.loads((ctx.run_root / "03_pose_dx_dz" / "stage_manifest.json").read_text())
    assert setup_state.to_dict()["label"] == "01_setup_geometry/03_axis_direction"
    assert np.all(np.isfinite(params5))
    assert phi_manifest["status"] == "failed"
    assert "data_loss_computed is false" in "\n".join(
        phi_manifest["failure_provenance"]["failures"]
    )
    assert dx_manifest["status"] == "skipped"
    assert records[-1]["stage"] == "02_pose_phi"
    assert records[-1]["status"] == "failed"
    assert [candidate["source_stage"] for candidate in candidates] == [
        "01_setup_geometry/01_cor",
        "01_setup_geometry/02_detector_roll",
        "01_setup_geometry/03_axis_direction",
    ]


def test_v2_report_records_failed_pose_stage_and_valid_final_candidate(tmp_path) -> None:
    run_dir = _write_minimal_staged_run(tmp_path)
    for rel in ["01_setup_geometry/02_detector_roll", "01_setup_geometry/03_axis_direction"]:
        stage_dir = run_dir / rel
        _write_stage_images(stage_dir)
        payload = json.loads((stage_dir / "stage_manifest.json").read_text())
        payload["status"] = "completed"
        payload["planned_after"] = None
        payload["artifacts"] = {
            "orthos": "orthos.png",
            "aligned_xy": "aligned_xy_global_z209.png",
            "delta_xy": "delta_xy_global_z209.png",
            "z_stack": "z_stack_global_z198_220.png",
        }
        _write_json(stage_dir / "stage_manifest.json", payload)
    phi_manifest = json.loads((run_dir / "02_pose_phi" / "stage_manifest.json").read_text())
    phi_manifest["status"] = "failed"
    phi_manifest["failure_provenance"] = {
        "passed": False,
        "failures": ["reconstruction volume finite fraction is 0"],
    }
    _write_json(run_dir / "02_pose_phi" / "stage_manifest.json", phi_manifest)
    for rel in ["03_pose_dx_dz", "04_pose_polish"]:
        payload = json.loads((run_dir / rel / "stage_manifest.json").read_text())
        payload["status"] = "skipped"
        payload["skip_reason"] = "upstream pose stage 02_pose_phi failed validation"
        _write_json(run_dir / rel / "stage_manifest.json", payload)
    final_dir = run_dir / "05_final"
    _write_stage_images(final_dir)
    _write_json(
        final_dir / "stage_manifest.json",
        {
            "stage": "05_final",
            "status": "completed",
            "volume_shape": [256, 256, 96],
            "artifacts": {
                "orthos": "orthos.png",
                "aligned_xy": "aligned_xy_global_z209.png",
                "delta_xy": "delta_xy_global_z209.png",
                "z_stack": "z_stack_global_z198_220.png",
            },
            "recon_info": {"loss": [100.0, 70.0], "effective_iters": 2, "regulariser": "tv"},
        },
    )

    summary = staged_runner.build_real_lamino_staged_report(
        run_dir,
        out_dir=tmp_path / "failed_pose_report",
    )

    assert summary["success"]["passed"] is False
    assert summary["success"]["phase"] == "v2_full_staged_failed_validation"
    assert summary["success"]["validation_failed"] is True
    assert summary["success"]["final_loss"] == 70.0
    failed = summary["success"]["failed_or_skipped_stages"]
    assert failed[0]["stage"] == "02_pose_phi"
    assert summary["staged_path"][4]["failure_provenance"]["failures"] == [
        "reconstruction volume finite fraction is 0"
    ]
    assert (tmp_path / "failed_pose_report" / "publication" / "full_orthos.png").exists()


def test_v2_report_copies_final_pose_summary_from_run_manifest(tmp_path) -> None:
    run_dir = _write_minimal_real_lamino_run(tmp_path, final_last_loss=70.0)

    summary = staged_runner.build_real_lamino_staged_report(
        run_dir,
        out_dir=tmp_path / "pose_summary_report",
    )

    assert summary["provenance"]["final_pose_summary"] == {"dx": {"std": 1.0}}


def test_v2_report_emits_reference_regression_table_and_flags_pose_loss_scale(tmp_path) -> None:
    (tmp_path / "reference").mkdir()
    (tmp_path / "v2").mkdir()
    reference_run = _write_minimal_real_lamino_run(tmp_path / "reference", final_last_loss=80.0)
    reference_report = reference_run / "real_lamino_report" / "real_lamino_summary.json"
    reference_report.parent.mkdir()
    _write_json(reference_report, {"success": {"passed": True}})

    v2_run = _write_minimal_real_lamino_run(tmp_path / "v2", final_last_loss=70.0)
    manifest = json.loads((v2_run / "run_manifest.json").read_text())
    manifest["workflow"] = {
        "reference_regression": True,
        "reference_regression_contract": {
            "passed": True,
            "mismatches": {},
        },
    }
    _write_json(v2_run / "run_manifest.json", manifest)
    (v2_run / "02_pose_phi" / "stage_summary.csv").write_text(
        "stage,level_factor,outer_idx,loss_before,loss_after,active_dofs\n"
        "02_pose_phi,1,1,200.0,200.0,phi\n",
        encoding="utf-8",
    )

    summary = staged_runner.build_real_lamino_staged_report(
        v2_run,
        out_dir=tmp_path / "reference_regression_report",
        reference_report=reference_report,
    )

    audit = summary["reference_regression"]
    assert audit["enabled"] is True
    assert audit["status"] == "failed"
    assert audit["pose_loss_scale_failures"][0]["stage"] == "02_pose_phi"
    table = tmp_path / "reference_regression_report" / "real_lamino_reference_regression_table.csv"
    assert table.exists()
    assert "loss_scale_mismatch" in table.read_text()
    assert summary["artifacts"]["reference_regression_table_csv"] == str(table.resolve())


def test_reference_regression_phi_level2_loss_scale_on_reference_path_is_recorded(
    tmp_path: Path,
) -> None:
    (tmp_path / "reference").mkdir()
    (tmp_path / "v2").mkdir()
    reference_run = _write_minimal_real_lamino_run(tmp_path / "reference", final_last_loss=80.0)
    reference_report = reference_run / "real_lamino_report" / "real_lamino_summary.json"
    reference_report.parent.mkdir()
    _write_json(reference_report, {"success": {"passed": True}})

    v2_run = _write_minimal_real_lamino_run(tmp_path / "v2", final_last_loss=70.0)
    manifest = json.loads((v2_run / "run_manifest.json").read_text())
    manifest["workflow"] = {
        "reference_regression": True,
        "reference_regression_contract": {
            "passed": True,
            "mismatches": {},
        },
    }
    _write_json(v2_run / "run_manifest.json", manifest)
    phi_rows = (
        "stage,level_factor,outer_idx,loss_before,loss_after,active_dofs\n"
        "02_pose_phi,4,1,129.0,128.9,phi\n"
        "02_pose_phi,2,1,482.2211,482.1140,phi\n"
        "02_pose_phi,1,1,1859.1869,1859.1372,phi\n"
    )
    (reference_run / "02_pose_phi" / "stage_summary.csv").write_text(
        phi_rows,
        encoding="utf-8",
    )
    (v2_run / "02_pose_phi" / "stage_summary.csv").write_text(
        phi_rows.replace("482.2211,482.1140", "481.8929,481.8202"),
        encoding="utf-8",
    )

    summary = staged_runner.build_real_lamino_staged_report(
        v2_run,
        out_dir=tmp_path / "reference_phi_report",
        reference_report=reference_report,
    )

    audit = summary["reference_regression"]
    assert audit["status"] == "recorded"
    assert audit["pose_loss_scale_failures"] == []
    table = (
        tmp_path / "reference_phi_report" / "real_lamino_reference_regression_table.csv"
    ).read_text()
    assert "02_pose_phi,2,1,482.2211,482.1140,481.8929,481.8202" in table
    assert "loss_scale_mismatch" not in table


def test_reference_regression_table_uses_cor_only_reconstruction_loss_and_flags_missing_rows(
    tmp_path,
) -> None:
    (tmp_path / "reference").mkdir()
    (tmp_path / "v2").mkdir()
    reference_run = _write_minimal_real_lamino_run(tmp_path / "reference", final_last_loss=80.0)
    reference_report = reference_run / "real_lamino_report" / "real_lamino_summary.json"
    reference_report.parent.mkdir()
    _write_json(reference_report, {"success": {"passed": True}})

    v2_run = _write_minimal_real_lamino_run(tmp_path / "v2", final_last_loss=70.0)
    manifest = json.loads((v2_run / "run_manifest.json").read_text())
    manifest["workflow"] = {
        "reference_regression": True,
        "reference_regression_contract": {
            "passed": True,
            "mismatches": {},
        },
    }
    _write_json(v2_run / "run_manifest.json", manifest)
    (reference_run / "01_setup_geometry" / "03_axis_direction" / "stage_summary.csv").write_text(
        "stage,level_factor,outer_iter,geometry_loss_before,geometry_loss_after,geometry_accepted\n"
        "01_setup_geometry/03_axis_direction,1,1,2.0,1.0,True\n"
        "01_setup_geometry/03_axis_direction,1,2,1.0,0.9,True\n",
        encoding="utf-8",
    )
    (reference_run / "06_cor_only_fista" / "stage_summary.csv").write_text(
        "stage,level_factor,outer_iter,geometry_loss_before,geometry_loss_after,geometry_accepted\n"
        "01_setup_geometry/01_cor,8,1,38.0,37.0,True\n",
        encoding="utf-8",
    )

    summary = staged_runner.build_real_lamino_staged_report(
        v2_run,
        out_dir=tmp_path / "reference_shape_report",
        reference_report=reference_report,
    )

    audit = summary["reference_regression"]
    assert audit["status"] == "failed"
    assert audit["row_shape_failures"][0]["stage"] == "01_setup_geometry/03_axis_direction"
    table = (
        tmp_path / "reference_shape_report" / "real_lamino_reference_regression_table.csv"
    ).read_text()
    assert "06_cor_only_fista,final,,120.0,100.0,120.0,100.0,1.0,matched," in table
    assert "06_cor_only_fista,8,1" not in table


def _write_minimal_real_lamino_run(tmp_path: Path, *, final_last_loss: float = 80.0) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "input": "real.nxs",
            "backend": "gpu",
            "devices": ["cuda:0"],
            "final_volume_shape": [256, 256, 96],
            "final_setup_estimates": {"detector": []},
            "final_pose_summary": {"dx": {"std": 1.0}},
        },
    )
    _write_json(run_dir / "status.json", {"completed_at": "2026-05-11T00:00:00"})
    stages = [
        ("00_baseline", [], {}),
        ("01_setup_geometry/01_cor", ["det_u_px"], {}),
        ("01_setup_geometry/02_detector_roll", ["detector_roll_deg"], {}),
        ("01_setup_geometry/03_axis_direction", ["axis_rot_x_deg", "axis_rot_y_deg"], {}),
        ("02_pose_phi", ["phi"], {"params_summary": {"phi": {"std": 0.1}}}),
        ("03_pose_dx_dz", ["dx", "dz"], {"params_summary": {"dx": {"std": 1.0}}}),
        ("04_pose_polish", ["alpha", "beta", "phi", "dx", "dz"], {}),
    ]
    for rel, dofs, extra in stages:
        stage_dir = run_dir / rel
        stage_dir.mkdir(parents=True)
        _write_stage_images(stage_dir)
        _write_json(
            stage_dir / "stage_manifest.json",
            {
                "stage": rel,
                "status": "completed",
                "active_dofs": dofs,
                "artifacts": {
                    "orthos": "orthos.png",
                    "aligned_xy": "aligned_xy_global_z209.png",
                    "delta_xy": "delta_xy_global_z209.png",
                    "z_stack": "z_stack_global_z198_220.png",
                },
                "geometry_calibration_state": {"stage": rel},
                **extra,
            },
        )
        (stage_dir / "stage_summary.csv").write_text(
            "stage,level_factor,outer_iter,geometry_loss_before,geometry_loss_after,geometry_accepted\n"
            f"{rel},1,1,2.0,1.0,True\n",
            encoding="utf-8",
        )
    final_dir = run_dir / "05_final"
    final_dir.mkdir()
    _write_stage_images(final_dir)
    _write_json(
        final_dir / "stage_manifest.json",
        {
            "stage": "05_final",
            "status": "completed",
            "volume_shape": [256, 256, 96],
            "artifacts": {
                "orthos": "orthos.png",
                "aligned_xy": "aligned_xy_global_z209.png",
                "delta_xy": "delta_xy_global_z209.png",
                "z_stack": "z_stack_global_z198_220.png",
            },
            "geometry_calibration_state": {"stage": "05_final"},
            "params_summary": {"dx": {"std": 2.0}},
            "recon_info": {
                "loss": [120.0, final_last_loss],
                "effective_iters": 2,
                "regulariser": "tv",
            },
        },
    )
    cor_dir = run_dir / "06_cor_only_fista"
    cor_dir.mkdir()
    _write_stage_images(cor_dir)
    _write_json(
        cor_dir / "stage_manifest.json",
        {
            "stage": "06_cor_only_fista",
            "status": "completed",
            "volume_shape": [256, 256, 96],
            "artifacts": {
                "orthos": "orthos.png",
                "aligned_xy": "aligned_xy_global_z209.png",
                "delta_xy": "delta_xy_global_z209.png",
                "z_stack": "z_stack_global_z198_220.png",
            },
            "setup_state": {"stage": "06_cor_only_fista"},
            "fista_info": {"loss": [120.0, 100.0], "effective_iters": 2, "regulariser": "tv"},
        },
    )
    return run_dir


def _write_minimal_staged_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "v2_run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema": "tomojax.real_lamino_staged_run.v2",
            "input": "real.nxs",
            "backend": "gpu",
            "devices": ["cuda:0"],
            "final_volume_shape": [256, 256, 96],
            "final_setup_estimates": {"det_u_px": 1.25},
            "workflow": {
                "implemented_stages": [
                    "00_baseline",
                    "01_setup_geometry/01_cor",
                    "06_cor_only_fista",
                ],
                "planned_stages": ["05_final"],
            },
        },
    )
    _write_json(run_dir / "status.json", {"completed_at": "2026-05-11T00:00:00"})
    for rel, dofs, status in [
        ("00_baseline", [], "completed"),
        ("01_setup_geometry/01_cor", ["det_u_px"], "completed"),
        ("01_setup_geometry/02_detector_roll", ["detector_roll_deg"], "planned"),
        ("01_setup_geometry/03_axis_direction", ["axis_rot_x_deg", "axis_rot_y_deg"], "planned"),
        ("02_pose_phi", ["phi"], "planned"),
        ("03_pose_dx_dz", ["dx", "dz"], "planned"),
        ("04_pose_polish", ["alpha", "beta", "phi", "dx", "dz"], "planned"),
        ("05_final", ["detector_roll", "axis_direction", "pose_5dof"], "planned"),
    ]:
        stage_dir = run_dir / rel
        stage_dir.mkdir(parents=True)
        if status == "completed":
            _write_stage_images(stage_dir)
        _write_json(
            stage_dir / "stage_manifest.json",
            {
                "stage": rel,
                "status": status,
                "active_dofs": dofs,
                "volume_shape": [256, 256, 96] if rel == "00_baseline" else None,
                "artifacts": {
                    "orthos": "orthos.png",
                    "aligned_xy": "aligned_xy_global_z209.png",
                    "delta_xy": "delta_xy_global_z209.png",
                    "z_stack": "z_stack_global_z198_220.png",
                }
                if status == "completed"
                else {},
                "geometry_calibration_state": {"det_u_px": 1.25},
                "planned_after": "v2 COR-only path works" if status == "planned" else None,
            },
        )
        if status == "completed":
            (stage_dir / "stage_summary.csv").write_text(
                "stage,level_factor,outer_iter,geometry_loss_before,geometry_loss_after,geometry_accepted\n"
                f"{rel},1,1,2.0,1.0,True\n",
                encoding="utf-8",
            )
    cor_dir = run_dir / "06_cor_only_fista"
    cor_dir.mkdir()
    _write_stage_images(cor_dir)
    _write_json(
        cor_dir / "stage_manifest.json",
        {
            "stage": "06_cor_only_fista",
            "status": "completed",
            "active_dofs": ["det_u_px"],
            "volume_shape": [256, 256, 96],
            "artifacts": {
                "orthos": "orthos.png",
                "aligned_xy": "aligned_xy_global_z209.png",
                "delta_xy": "delta_xy_global_z209.png",
                "z_stack": "z_stack_global_z198_220.png",
            },
            "geometry_calibration_state": {"det_u_px": 1.25},
            "fista_info": {"loss": [120.0, 80.0], "effective_iters": 2, "regulariser": "tv"},
        },
    )
    return run_dir


def _write_stage_images(stage_dir: Path) -> None:
    for name in (
        "orthos.png",
        "aligned_xy_global_z209.png",
        "delta_xy_global_z209.png",
        "z_stack_global_z198_220.png",
    ):
        (stage_dir / name).write_bytes(b"png")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
