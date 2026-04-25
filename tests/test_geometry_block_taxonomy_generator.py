from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/generate_alignment_before_after_128.py")


def _load_generator():
    spec = importlib.util.spec_from_file_location("geometry_block_taxonomy", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_geometry_block_taxonomy_uses_lamino_disk_not_shepp_logan():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "shepp_logan" not in source
    assert "lamino_disk" in source


def test_geometry_block_taxonomy_docs_profile_matches_historical_run_contract(tmp_path):
    generator = _load_generator()
    out = tmp_path / "taxonomy"

    generator.main(["--out", str(out), "--dry-run", "--profile", "docs"])

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((out / "artifacts" / "status.json").read_text(encoding="utf-8"))

    assert manifest["phantom"] == {
        "kind": "random_shapes/lamino_disk",
        "seed": 20260458,
        "shared_across_cases": True,
        "source": "tomojax.data.phantoms.lamino_disk",
    }
    assert manifest["profile"]["size"] == 128
    assert manifest["profile"]["views"] == 128
    assert tuple(manifest["profile"]["levels"]) == (8, 4, 2, 1)
    assert manifest["profile"]["views_per_batch"] == 1
    assert status["state"] == "dry_run_completed"
    assert status["scenario_count"] == len(manifest["scenarios"])


def test_geometry_block_taxonomy_records_estimated_supplied_and_gauge_metadata(tmp_path):
    generator = _load_generator()
    out = tmp_path / "taxonomy"

    generator.main(["--out", str(out), "--dry-run", "--scenario", "known_det_u_control"])

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    scenarios = manifest["scenarios"]

    assert [scenario["slug"] for scenario in scenarios] == ["known_det_u_control"]
    control = scenarios[0]
    assert control["geometry_dofs"] == []
    assert control["hidden_truth"]["det_u_px"] == -4.0
    assert control["supplied_corrections"] == {"det_u_px": -4.0}
    assert "detector/ray-grid centre offset" in manifest["gauge_notes"]["det_u_px"]
    assert "not reported as estimated" in manifest["gauge_notes"]["supplied_controls"]


def test_geometry_block_taxonomy_scenarios_cover_new_geometry_blocks():
    generator = _load_generator()
    scenarios = generator.scenario_catalog()
    dof_sets = {scenario.slug: set(scenario.geometry_dofs) for scenario in scenarios}

    assert dof_sets["parallel_det_u_m004"] == {"det_u_px"}
    assert dof_sets["parallel_detector_roll_p2p5"] == {"detector_roll_deg"}
    assert dof_sets["parallel_axis_pitch_p2p0"] == {"axis_rot_x_deg"}
    assert dof_sets["parallel_axis_yaw_m2p0"] == {"axis_rot_y_deg"}
    assert dof_sets["lamino_tilt_34p4"] == {"tilt_deg"}
    assert dof_sets["parallel_det_u_roll_combo"] == {"det_u_px", "detector_roll_deg"}
    assert dof_sets["parallel_det_u_axis_refine"] == {"det_u_px", "axis_rot_x_deg"}
    assert dof_sets["lamino_det_u_tilt_combo"] == {"det_u_px", "tilt_deg"}


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
