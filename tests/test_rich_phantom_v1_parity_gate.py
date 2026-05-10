from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from tomojax.datasets import generate_synthetic_dataset
from tomojax.forward import project_parallel_reference
from tomojax.geometry import GeometryState, read_geometry_json, read_pose_params_csv


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_rich_phantom_v1_parity_gate.py"
    spec = importlib.util.spec_from_file_location("run_rich_phantom_v1_parity_gate", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load rich phantom parity gate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_variable_projection_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_detu_variable_projection_diagnostic.py"
    )
    spec = importlib.util.spec_from_file_location("run_detu_variable_projection_diagnostic", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load variable projection diagnostic module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multires_geometry_scaling_doubles_pixel_dofs_between_levels(tmp_path: Path) -> None:
    gate = _load_gate_module()
    dataset = generate_synthetic_dataset(
        "rich_phantom94_det_u_only_v1_parity",
        tmp_path / "dataset",
        size=32,
        clean=True,
        views=4,
        supported_only=True,
    )
    state = read_geometry_json(
        dataset.v2_true_geometry,
        read_pose_params_csv(dataset.v2_true_pose),
    )

    scaled = gate._rescale_geometry(state, from_factor=4, to_factor=2)

    np.testing.assert_allclose(scaled.setup.det_u_px.value, state.setup.det_u_px.value * 2.0)
    np.testing.assert_allclose(scaled.setup.det_v_px.value, state.setup.det_v_px.value * 2.0)
    np.testing.assert_allclose(scaled.pose.dx_px, state.pose.dx_px * 2.0)
    np.testing.assert_allclose(scaled.pose.dz_px, state.pose.dz_px * 2.0)


def test_forward_consistent_coarse_sidecar_projects_level_volume(tmp_path: Path) -> None:
    gate = _load_gate_module()
    source = generate_synthetic_dataset(
        "rich_phantom94_det_u_only_v1_parity",
        tmp_path / "source",
        size=32,
        clean=True,
        views=4,
        supported_only=True,
    )
    output_dir = tmp_path / "coarse"

    gate._write_downsampled_sidecar(
        source_dir=source.dataset_dir,
        output_dir=output_dir,
        factor=2,
        carried_geometry=None,
    )

    manifest = cast(
        "dict[str, object]",
        json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8")),
    )
    assert manifest["artifact_contract"] == (
        "tomojax-v2.synthetic-dataset.multires-forward-consistent.v1"
    )
    volume = np.load(output_dir / "ground_truth_volume.npy")
    projections = np.load(output_dir / "projections.npy")
    true_pose = read_pose_params_csv(output_dir / "v2_true_pose_params.csv")
    true_geometry = read_geometry_json(output_dir / "v2_true_geometry.json", true_pose)

    expected = project_parallel_reference(jnp.asarray(volume), true_geometry)

    np.testing.assert_allclose(np.asarray(expected), projections, rtol=1.0e-5, atol=1.0e-5)


def test_multires_summary_collates_carried_detu_curves(tmp_path: Path) -> None:
    gate = _load_gate_module()
    run_dir = tmp_path / "stopped_otsu_l2_multires_f4_32_128v"
    run_dir.mkdir(parents=True)
    with (run_dir / "detu_loss_curves.csv").open("w", encoding="utf-8") as handle:
        _ = handle.write(
            "volume_source,det_u_px,loss,finite_difference_gradient,mask_role,loss_mode,"
            "residual_sigma,residual_filters\n"
            "true_volume,1.0,4.0,0.0,alignment_loss_mask,l2,1.0,raw\n"
            "final_stopped_volume,1.0,3.0,0.0,alignment_loss_mask,l2,1.0,raw\n"
            "final_stopped_volume,2.0,2.0,0.0,alignment_loss_mask,l2,1.0,raw\n"
        )
    rows = [
        {
            "run_name": "stopped_otsu_l2_multires_f4_32_128v",
            "volume_shape": "32x32x32",
            "artifact_dir": str(run_dir),
        }
    ]

    gate._write_multires_carried_detu_landscape(tmp_path, rows)

    summary = json.loads(
        (tmp_path / "multires_carried_detu_summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema"] == "tomojax.multires_carried_detu_landscape.v1"
    curve = summary["curves"][0]
    assert curve["volume_source"] == "multires_carried_f4_final_volume"
    assert curve["argmin_det_u_px"] == 2.0
    csv_rows = (tmp_path / "multires_carried_detu_loss_curves.csv").read_text(
        encoding="utf-8"
    )
    assert "multires_carried_f4_final_volume" in csv_rows
    assert (tmp_path / "multires_carried_detu_summary.md").exists()


def test_rich_phantom_gate_passes_preview_gauge_config(monkeypatch, tmp_path: Path) -> None:
    gate = _load_gate_module()
    captured: dict[str, object] = {}

    class EmptyConsistency:
        def to_dict(self) -> dict[str, object]:
            return {}

    class FakeSolver:
        def __init__(self, config: object) -> None:
            captured["config"] = config

        def run_smoke(self, run_dir: Path) -> None:
            run_dir.mkdir(parents=True, exist_ok=True)

    def summary_row(run_name: str, run_dir: Path, _elapsed: float) -> dict[str, str]:
        return {"run_name": run_name, "artifact_dir": str(run_dir)}

    monkeypatch.setattr(gate, "AlternatingAlignmentSolver", FakeSolver)
    monkeypatch.setattr(
        gate,
        "load_synthetic_dataset_sidecars",
        lambda _path: SimpleNamespace(
            manifest={"recovery_tolerances": {}, "unsupported_dofs_not_evaluated": []},
            true_geometry=GeometryState.zeros(7),
            consistency=EmptyConsistency(),
        ),
    )
    monkeypatch.setattr(gate, "_summary_row", summary_row)

    _ = gate._run_solver_inprocess(
        root=tmp_path,
        run_name="stopped",
        dataset_dir=tmp_path / "dataset",
        size=64,
        views=7,
        profile="lightning",
        volume_source="stopped_reconstruction",
        preview_volume_support="scout_soft",
        preview_support_outside_weight=0.1,
        preview_low_frequency_anchor_weight=0.05,
        preview_det_u_gauge_mode_weight=0.2,
    )

    config = cast("Any", captured["config"])
    assert config.preview_volume_support == "scout_soft"
    assert config.preview_support_outside_weight == 0.1
    assert config.preview_low_frequency_anchor_weight == 0.05
    assert config.preview_det_u_gauge_mode_weight == 0.2


def test_variable_projection_candidate_grid_covers_markers() -> None:
    diagnostic = _load_variable_projection_module()

    values = diagnostic._candidate_det_u_values(
        true_det_u=7.25,
        initial_det_u=0.0,
        final_det_u=5.75,
        radius=1.0,
        step=0.5,
    )

    assert 7.25 in values
    assert 0.0 in values
    assert 5.75 in values
    assert list(values) == sorted(values)
