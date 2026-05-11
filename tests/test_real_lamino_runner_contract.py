# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "run_real_lamino_native_setup_pose_256.py"
)
_MVP_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "summarize_real_lamino_mvp.py"
)
_V2_COR_MVP_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "real_laminography"
    / "run_real_lamino_v2_cor_mvp.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_real_lamino_native_setup_pose_256",
    _SCRIPT_PATH,
)
assert _SPEC is not None
assert _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

_MVP_SPEC = importlib.util.spec_from_file_location("summarize_real_lamino_mvp", _MVP_SCRIPT_PATH)
assert _MVP_SPEC is not None
assert _MVP_SPEC.loader is not None
mvp_runner = importlib.util.module_from_spec(_MVP_SPEC)
_MVP_SPEC.loader.exec_module(mvp_runner)

_V2_COR_MVP_SPEC = importlib.util.spec_from_file_location(
    "run_real_lamino_v2_cor_mvp",
    _V2_COR_MVP_SCRIPT_PATH,
)
assert _V2_COR_MVP_SPEC is not None
assert _V2_COR_MVP_SPEC.loader is not None
v2_cor_mvp_runner = importlib.util.module_from_spec(_V2_COR_MVP_SPEC)
_V2_COR_MVP_SPEC.loader.exec_module(v2_cor_mvp_runner)


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
    assert args.views_per_batch == 0
    assert cfg.align_profile == "lightning"
    assert cfg.projector_backend == "pallas"
    assert cfg.regulariser == "huber_tv"
    assert cfg.quality_tier == "fast"
    assert cfg.fallback_policy == "fallback"


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


def test_real_mvp_summary_uses_final_vs_cor_only_quality_contract(tmp_path) -> None:
    run_dir = _write_minimal_real_mvp_run(tmp_path)

    summary = mvp_runner.build_real_mvp_report(run_dir, out_dir=tmp_path / "mvp")

    assert summary["schema"] == "tomojax.real_lamino_mvp_report.v1"
    assert summary["success"]["passed"] is True
    assert summary["quality_basis"]["truth_metrics"] == "not_applicable_real_data"
    assert summary["reconstruction_comparison"]["loss_improvement_abs"] == 20.0
    assert (tmp_path / "mvp" / "real_mvp_summary.json").exists()
    assert (tmp_path / "mvp" / "real_mvp_residual_trace.csv").exists()
    assert (tmp_path / "mvp" / "real_mvp_geometry_trace.json").exists()
    assert (tmp_path / "mvp" / "publication" / "before_orthos.png").exists()
    assert (tmp_path / "mvp" / "publication" / "cor_only_orthos.png").exists()
    assert (tmp_path / "mvp" / "publication" / "full_orthos.png").exists()


def test_real_mvp_summary_can_require_quality_success(tmp_path) -> None:
    run_dir = _write_minimal_real_mvp_run(tmp_path, final_last_loss=130.0)

    with pytest.raises(RuntimeError, match="did not improve"):
        mvp_runner.build_real_mvp_report(
            run_dir,
            out_dir=tmp_path / "mvp",
            require_success=True,
        )


def test_v2_cor_mvp_smoke_reduces_to_det_u_and_cor_only(monkeypatch, tmp_path) -> None:
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

    args = v2_cor_mvp_runner._parse_args()

    assert args.levels_setup == [8]
    assert args.outer_iters == 1
    assert args.recon_iters == 3
    assert args.tv_prox_iters == 2
    assert args.views_per_batch == 16


def test_v2_cor_mvp_report_preserves_partial_contract(tmp_path) -> None:
    run_dir = _write_minimal_v2_cor_mvp_run(tmp_path)

    summary = v2_cor_mvp_runner.build_v2_cor_mvp_report(
        run_dir,
        out_dir=tmp_path / "v2_report",
        reference_report=Path("runs/reference/real_mvp_report/real_mvp_summary.json"),
    )

    assert summary["schema"] == "tomojax.real_lamino_v2_cor_mvp_report.v1"
    assert summary["contract_compatible_with"] == "tomojax.real_lamino_mvp_report.v1"
    assert summary["success"]["passed"] is True
    assert summary["success"]["full_mvp_success_deferred"] is True
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
    assert (tmp_path / "v2_report" / "real_mvp_summary.json").exists()
    assert (tmp_path / "v2_report" / "real_mvp_residual_trace.csv").exists()
    assert (tmp_path / "v2_report" / "real_mvp_geometry_trace.json").exists()
    assert (tmp_path / "v2_report" / "publication" / "before_orthos.png").exists()
    assert (tmp_path / "v2_report" / "publication" / "cor_only_orthos.png").exists()


def _write_minimal_real_mvp_run(tmp_path: Path, *, final_last_loss: float = 80.0) -> Path:
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


def _write_minimal_v2_cor_mvp_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "v2_run"
    run_dir.mkdir()
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema": "tomojax.real_lamino_v2_cor_mvp_run.v1",
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
