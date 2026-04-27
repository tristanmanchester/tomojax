from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


SCRIPT_PATH = Path("scripts/generate_alignment_before_after_128.py")


def _load_generator():
    spec = importlib.util.spec_from_file_location("geometry_block_taxonomy", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_block_taxonomy_uses_selected_center_biased_random_shapes():
    generator = _load_generator()
    phantom = generator._phantom_metadata()

    assert phantom["kind"] == "random_shapes/center_biased_sphere_cubes_spheres"
    assert phantom["seed"] == 20260893
    assert phantom["n_cubes"] == 22
    assert phantom["n_spheres"] == 22
    assert phantom["placement"] == "center_biased_sphere"
    assert phantom["radial_exponent"] == 0.75


def test_geometry_block_taxonomy_docs_profile_matches_historical_run_contract(tmp_path):
    generator = _load_generator()
    out = tmp_path / "taxonomy"

    generator.main(["--out", str(out), "--dry-run", "--profile", "docs"])

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((out / "artifacts" / "status.json").read_text(encoding="utf-8"))

    assert manifest["phantom"]["kind"] == "random_shapes/center_biased_sphere_cubes_spheres"
    assert manifest["phantom"]["seed"] == 20260893
    assert manifest["phantom"]["n_cubes"] == 22
    assert manifest["phantom"]["n_spheres"] == 22
    assert manifest["phantom"]["placement"] == "center_biased_sphere"
    assert manifest["phantom"]["radial_exponent"] == 0.75
    assert manifest["phantom"]["selection"] == (
        "phantom_picker_128_10x10_center_biased_sphere_slot_94"
    )
    assert manifest["profile"]["size"] == 128
    assert manifest["profile"]["views"] == 128
    assert tuple(manifest["profile"]["levels"]) == (8, 4, 2, 1)
    assert manifest["profile"]["outer_iters"] == 16
    assert manifest["profile"]["early_stop"] is True
    assert manifest["profile"]["early_stop_rel_impr"] == 1e-3
    assert manifest["profile"]["early_stop_patience"] == 2
    assert manifest["profile"]["views_per_batch"] == 1
    assert status["state"] == "dry_run_completed"
    assert status["scenario_count"] == len(manifest["scenarios"])


def test_geometry_block_taxonomy_scenarios_cover_new_geometry_blocks():
    generator = _load_generator()
    scenarios = generator.scenario_catalog()
    dof_sets = {scenario.slug: set(scenario.geometry_dofs) for scenario in scenarios}

    assert dof_sets["parallel_cor_u_m004"] == {"det_u_px"}
    assert dof_sets["parallel_detector_roll_p2p5"] == {"detector_roll_deg"}
    assert dof_sets["parallel_axis_pitch_full360_p2p0"] == {"axis_rot_x_deg"}
    assert dof_sets["parallel_axis_yaw_full360_m2p0"] == {"axis_rot_y_deg"}
    assert dof_sets["lamino_tilt_34p4"] == {"tilt_deg"}
    assert dof_sets["parallel_cor_roll_combo"] == {"det_u_px", "detector_roll_deg"}
    assert dof_sets["lamino_cor_tilt_combo"] == {"det_u_px", "tilt_deg"}


def test_visual_stress_scenarios_record_explicit_acquisition_span(tmp_path):
    generator = _load_generator()
    out = tmp_path / "taxonomy"

    generator.main(
        [
            "--out",
            str(out),
            "--dry-run",
            "--profile",
            "docs",
            "--scenario-set",
            "visual_stress",
        ]
    )

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    spans = {scenario["slug"]: scenario["theta_span_deg"] for scenario in manifest["scenarios"]}
    titles = {scenario["slug"]: scenario["title"] for scenario in manifest["scenarios"]}

    assert spans["stress_parallel_cor_u_m008"] == 180.0
    assert spans["stress_parallel_detector_roll_p10"] == 180.0
    assert spans["stress_parallel_axis_pitch_full360_p18"] == 360.0
    assert spans["stress_parallel_axis_yaw_full360_m18"] == 360.0
    assert spans["stress_lamino_tilt_50"] == 360.0
    assert "Parallel CT" not in titles["stress_parallel_axis_pitch_full360_p18"]
    assert "Parallel CT" not in titles["stress_parallel_axis_yaw_full360_m18"]
    assert all(scenario["n_views"] == 128 for scenario in manifest["scenarios"])


def test_dry_run_catalog_metadata_includes_suite_category_and_expectation(tmp_path):
    generator = _load_generator()
    out = tmp_path / "catalog"

    generator.main(["--out", str(out), "--dry-run", "--profile", "docs", "--scenario-set", "capability"])

    scenarios = json.loads((out / "artifacts" / "scenario_catalog.json").read_text())
    first = scenarios[0]

    assert first["suite_name"] == "capability"
    assert first["scenario_category"] == "capability"
    assert first["scenario_family"]
    assert first["expectation"] == "success"
    assert first["headline_eligible"] is True
    assert first["phantom_key"] == "phantom94"


def test_comprehensive_dry_run_marks_diagnostics_non_headline(tmp_path):
    generator = _load_generator()
    out = tmp_path / "comprehensive"

    generator.main(
        [
            "--out",
            str(out),
            "--dry-run",
            "--profile",
            "docs",
            "--scenario-set",
            "comprehensive_128",
        ]
    )

    scenarios = json.loads((out / "artifacts" / "scenario_catalog.json").read_text())
    diagnostics = [scenario for scenario in scenarios if scenario["scenario_category"] == "diagnostic"]

    assert diagnostics
    assert all(scenario["headline_eligible"] is False for scenario in diagnostics)


def test_stress_dry_run_writes_only_stress_scenarios(tmp_path):
    generator = _load_generator()
    out = tmp_path / "stress"

    generator.main(["--out", str(out), "--dry-run", "--profile", "docs", "--scenario-set", "stress"])

    scenarios = json.loads((out / "artifacts" / "scenario_catalog.json").read_text())

    assert scenarios
    assert {scenario["scenario_category"] for scenario in scenarios} == {"stress"}


def test_geometry_block_taxonomy_passes_profile_early_stop_to_align_config(monkeypatch):
    generator = _load_generator()
    grid = generator.Grid(4, 4, 4, 1.0, 1.0, 1.0)
    detector = generator.Detector(4, 4, 1.0, 1.0)
    geometry = generator.ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=[0.0, 90.0],
    )
    profile = generator.docs_profile()
    captured = {}

    def fake_align_multires(*args, **kwargs):
        cfg = kwargs["cfg"]
        captured["early_stop"] = cfg.early_stop
        captured["early_stop_rel_impr"] = cfg.early_stop_rel_impr
        captured["early_stop_patience"] = cfg.early_stop_patience
        captured["schedule"] = cfg.schedule
        captured["optimise_dofs"] = cfg.optimise_dofs
        state = generator.GeometryCalibrationState.from_geometry(
            geometry,
            active_geometry_dofs=("det_u_px",),
        )
        info = {
            "geometry_calibration_state": state.to_calibration_state().to_dict(),
            "geometry_calibration_diagnostics": {"schema_version": 1, "blocks": []},
            "outer_stats": [],
        }
        return generator.jnp.zeros((4, 4, 4), dtype=generator.jnp.float32), None, info

    monkeypatch.setattr(generator, "align_multires", fake_align_multires)

    generator._run_geometry_alignment(
        generator.Scenario(
            slug="probe",
            title="probe",
            description="probe",
            geometry_type="parallel",
            geometry_dofs=("det_u_px",),
            schedule="cor",
        ),
        nominal_geometry=geometry,
        grid=grid,
        detector=detector,
        projections=generator.jnp.zeros((2, 4, 4), dtype=generator.jnp.float32),
        profile=profile,
    )

    assert captured == {
        "early_stop": True,
        "early_stop_rel_impr": 1e-3,
        "early_stop_patience": 2,
        "schedule": "cor",
        "optimise_dofs": None,
    }


def test_master_panel_ignores_failed_rows_without_panel_paths(tmp_path):
    generator = _load_generator()
    master = tmp_path / "master.png"

    generator._write_master_panel(
        [
            {"slug": "failed", "before_after_panel": ""},
            {"slug": "also_failed"},
        ],
        master,
    )

    assert not master.exists()


def test_write_visuals_emits_rich_inspection_artifacts(tmp_path):
    generator = _load_generator()
    base = np.zeros((16, 16, 16), dtype=np.float32)
    base[3:7, 4:8, 5:9] = 1.0
    naive = base.copy()
    naive[8:11, 8:11, 8:11] = 0.5
    calibrated = base * 0.9
    aligned = base * 0.98
    scenario = generator.Scenario(
        slug="visual_probe",
        title="Visual probe",
        description="Visual probe",
        geometry_type="parallel",
        geometry_dofs=("det_u_px",),
        hidden_det_u_px=-3.0,
    )
    metrics = {
        "naive_volume_nmse": generator._volume_nmse(naive, base),
        "calibrated_volume_nmse": generator._volume_nmse(calibrated, base),
        "aligned_tv_volume_nmse": generator._volume_nmse(aligned, base),
    }
    paths = generator._write_visuals(
        scenario,
        out_dir=tmp_path,
        profile=generator.smoke_profile(),
        theta_span=180.0,
        truth=base,
        naive_fbp=naive,
        calibrated_fbp=calibrated,
        aligned_tv=aligned,
        estimates={
            "det_u_px": -2.9,
            "det_v_px": 0.0,
            "detector_roll_deg": 0.0,
            "axis_rot_x_deg": 0.0,
            "axis_rot_y_deg": 0.0,
        },
        metrics=metrics,
        diagnostics={"schema_version": 1, "overall_status": "converged", "blocks": []},
        outer_stats=[
            {
                "loss_kind": "l2_otsu",
                "level_factor": 4,
                "geometry_block": "detector_center",
                "geometry_loss_before": 1.0,
                "geometry_loss_after": 0.7,
                "geometry_accepted": True,
            },
            {
                "loss_kind": "l2_otsu",
                "level_factor": 4,
                "geometry_block": "detector_center",
                "geometry_loss_before": 0.7,
                "geometry_loss_after": 0.5,
                "geometry_accepted": False,
            },
        ],
    )

    expected = {
        "inspection_panel",
        "loss_panel",
        "diagnostics_panel",
        "difference_calibrated_truth_orthos",
        "difference_aligned_truth_orthos",
        "difference_aligned_naive_orthos",
    }
    assert expected <= set(paths)
    assert paths["before_after_panel"] != ""
    for key in expected | {"before_after_panel"}:
        image = iio.imread(paths[key])
        assert image.ndim == 3
        assert image.shape[2] == 3
    diff = iio.imread(paths["difference_aligned_naive_orthos"])
    assert not np.array_equal(diff[..., 0], diff[..., 2])


def test_write_naive_visuals_emits_reduced_rich_panel(tmp_path):
    generator = _load_generator()
    base = np.zeros((16, 16, 16), dtype=np.float32)
    base[4:10, 5:9, 3:8] = 1.0
    naive = np.roll(base, 1, axis=0)
    scenario = generator.Scenario(
        slug="naive_probe",
        title="Naive probe",
        description="Naive probe",
        geometry_type="parallel",
        geometry_dofs=(),
    )

    paths = generator._write_naive_visuals(
        scenario,
        out_dir=tmp_path,
        truth=base,
        naive_fbp=naive,
    )

    assert paths["inspection_panel"] == paths["before_after_panel"]
    assert paths["diagnostics_panel"]
    assert paths["difference_aligned_naive_orthos"]
    for key in ("before_after_panel", "diagnostics_panel", "difference_aligned_naive_orthos"):
        image = iio.imread(paths[key])
        assert image.ndim == 3
        assert image.shape[2] == 3


def test_solver_metadata_summary_extracts_validation_lm_fields():
    generator = _load_generator()

    metadata = generator._last_solver_metadata(
        [
            {
                "objective_kind": "bilevel_cv",
                "optimizer_kind": "validation_lm",
                "outer_loss_kind": "l2_otsu",
                "recon_sensitivity": "stopped",
                "fold_eval_mode": "stopped_train_recon_validation_lm",
                "active_gradient_mode": "validation_residual_jvp",
                "train_reconstruction_gradient": False,
                "schedule_name": "cor",
                "schedule_stage_name": "cor",
                "gauge_policy": "reject",
                "gauge_status": "ok",
            }
        ]
    )

    assert metadata["optimizer_kind"] == "validation_lm"
    assert metadata["outer_loss_kind"] == "l2_otsu"
    assert metadata["recon_sensitivity"] == "stopped"
    assert metadata["train_reconstruction_gradient"] is False
    assert metadata["schedule_stage_name"] == "cor"
    assert metadata["gauge_status"] == "ok"
