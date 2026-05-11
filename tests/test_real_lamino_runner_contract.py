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
    assert args.views_per_batch == 1
    assert args.bin_factor == 4


def test_v2_cor_mvp_accepts_explicit_binned_smoke_shape(monkeypatch, tmp_path) -> None:
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

    args = v2_cor_mvp_runner._parse_args()

    assert args.bin_factor == 2
    assert args.smoke_shape == (16, 64, 64)


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
        v2_cor_mvp_runner._prepare_binned_fixture(
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
        v2_cor_mvp_runner._prepare_binned_fixture(
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

    assert v2_cor_mvp_runner._artifact_validation_failures(stage_dir) == []


def test_v2_pose_stage_validation_accepts_finite_fast_profile_losses() -> None:
    failures = v2_cor_mvp_runner._stat_validation_failures(
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


def test_v2_cor_mvp_runtime_default_streams_fista(monkeypatch, tmp_path) -> None:
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

    args = v2_cor_mvp_runner._parse_args()
    assert args.views_per_batch == 0

    v2_cor_mvp_runner._normalize_runtime_args(args)

    assert args.views_per_batch == 1


def test_v2_cor_mvp_defaults_to_reference_conservative_pose_bounds(monkeypatch, tmp_path) -> None:
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

    args = v2_cor_mvp_runner._parse_args()

    assert args.pose_bounds_profile == "reference_conservative"
    assert v2_cor_mvp_runner._pose_phi_bounds(args) == "phi=-0.00872665:0.00872665"
    assert v2_cor_mvp_runner._pose_dx_dz_bounds(args) == "dx=-10:10,dz=-10:10"
    assert "alpha=-0.00872665:0.00872665" in v2_cor_mvp_runner._pose_polish_bounds(args)

    args.effective_bin_factor = 4
    assert v2_cor_mvp_runner._setup_det_u_bounds(args) == "det_u_px=-6:6"
    assert v2_cor_mvp_runner._pose_dx_dz_bounds(args) == "dx=-2.5:2.5,dz=-2.5:2.5"


def test_v2_cor_mvp_report_preserves_partial_contract(tmp_path) -> None:
    run_dir = _write_minimal_v2_cor_mvp_run(tmp_path)

    summary = v2_cor_mvp_runner.build_v2_cor_mvp_report(
        run_dir,
        out_dir=tmp_path / "v2_report",
        reference_report=Path("runs/reference/real_mvp_report/real_mvp_summary.json"),
    )

    assert summary["schema"] == "tomojax.real_lamino_v2_mvp_report.v1"
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


def test_v2_full_mvp_report_fails_when_final_is_worse_than_cor_only(tmp_path) -> None:
    run_dir = _write_minimal_v2_cor_mvp_run(tmp_path)
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

    summary = v2_cor_mvp_runner.build_v2_cor_mvp_report(
        run_dir,
        out_dir=tmp_path / "v2_full_report",
    )

    assert summary["success"]["phase"] == "v2_full_mvp"
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

    volume, choice = v2_cor_mvp_runner.run_best_final_reconstruction(
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
    setup_state, params5, records, candidates = v2_cor_mvp_runner.run_remaining_stages(
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
    run_dir = _write_minimal_v2_cor_mvp_run(tmp_path)
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

    summary = v2_cor_mvp_runner.build_v2_cor_mvp_report(
        run_dir,
        out_dir=tmp_path / "failed_pose_report",
    )

    assert summary["success"]["passed"] is False
    assert summary["success"]["phase"] == "v2_full_mvp_failed_validation"
    assert summary["success"]["validation_failed"] is True
    assert summary["success"]["final_loss"] == 70.0
    failed = summary["success"]["failed_or_skipped_stages"]
    assert failed[0]["stage"] == "02_pose_phi"
    assert summary["staged_path"][4]["failure_provenance"]["failures"] == [
        "reconstruction volume finite fraction is 0"
    ]
    assert (tmp_path / "failed_pose_report" / "publication" / "full_orthos.png").exists()


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
