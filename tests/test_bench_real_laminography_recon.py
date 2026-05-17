from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import numpy as np

import tomojax.bench.real_laminography_recon as recon_stage


class _CalibrationState:
    def to_dict(self) -> dict[str, Any]:
        return {"det_u_px": 1.25, "active_geometry_dofs": ["det_u_px"]}


class _SetupState:
    def to_calibration_state(self) -> _CalibrationState:
        return _CalibrationState()


class _Context:
    def __init__(self, root: Path, *, canonical_det_grid: bool) -> None:
        self.args = Namespace(
            canonical_det_grid=canonical_det_grid,
            final_candidate_policy="all",
            filter_name="ramp",
            gather_dtype="float32",
            lambda_tv=0.002,
            recon_iters=7,
            recon_positivity=True,
            regulariser="huber_tv",
            tv_prox_iters=3,
            views_per_batch=2,
        )
        self.run_root = root
        self.status_path = root / "status.json"
        self.naive_slice = np.zeros((2, 2), dtype=np.float32)
        self.preview_global_z = 209
        self.stack_z_range = (203, 215)
        self.saved_products: list[dict[str, Any]] = []

    def stage_dir(self, name: str) -> Path:
        return self.run_root / name

    def save_stage_products(self, **kwargs: Any) -> dict[str, str]:
        self.saved_products.append(kwargs)
        stage_dir = Path(kwargs["stage_dir"])
        products = {
            "aligned_xy": stage_dir / "aligned_xy_global_z209.png",
            "delta_xy": stage_dir / "delta_xy_global_z209.png",
            "orthos": stage_dir / "orthos.png",
            "z_stack": stage_dir / "z_stack_global_z203_215.png",
        }
        for path in products.values():
            path.write_bytes(b"png")
        return {key: str(path) for key, path in products.items()}


class _FinalContext:
    def __init__(self, root: Path, *, final_candidate_policy: str = "all") -> None:
        self.args = Namespace(final_candidate_policy=final_candidate_policy)
        self.run_root = root
        self.stage_dir = lambda name: root / name


def test_baseline_stage_writes_raw_fbp_artifact_contract(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, Any] = {}
    rendered_pngs: list[Path] = []
    expected_slice = np.full((2, 2), 0.5, dtype=np.float32)

    class _BlockUntilReady:
        def __call__(self, value):
            calls["block_until_ready"] = value
            return value

    def fake_fbp(
        geometry,
        grid,
        detector,
        projections,
        *,
        filter_name,
        views_per_batch,
        checkpoint_projector,
        gather_dtype,
    ):
        calls["fbp"] = {
            "geometry": geometry,
            "grid": grid,
            "detector": detector,
            "projections_shape": tuple(np.asarray(projections).shape),
            "filter_name": filter_name,
            "views_per_batch": views_per_batch,
            "checkpoint_projector": checkpoint_projector,
            "gather_dtype": gather_dtype,
        }
        return np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    def fake_xy_at_global_z(volume, *, grid, full_nz, global_z):
        calls["xy_at_global_z"] = {
            "volume_shape": tuple(np.asarray(volume).shape),
            "grid": grid,
            "full_nz": full_nz,
            "global_z": global_z,
        }
        return expected_slice

    def fake_save_uint8_png(path, image):
        rendered_pngs.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(np.asarray(image).tobytes())

    monkeypatch.setattr(recon_stage, "fbp", fake_fbp)
    monkeypatch.setattr(recon_stage.jax, "block_until_ready", _BlockUntilReady())
    monkeypatch.setattr(recon_stage, "real_lamino_xy_at_global_z", fake_xy_at_global_z)
    monkeypatch.setattr(recon_stage, "save_uint8_png", fake_save_uint8_png)

    ctx = _Context(tmp_path, canonical_det_grid=False)
    raw_projections = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    volume = recon_stage.run_baseline_stage(
        ctx,
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=raw_projections,
        full_nz=96,
    )

    stage_dir = tmp_path / "00_baseline"
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    align_info = json.loads((stage_dir / "align_info.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    summary_lines = (stage_dir / "stage_summary.csv").read_text().splitlines()

    assert volume.shape == (2, 2, 2)
    assert np.load(stage_dir / "naive_fbp.npy").shape == (2, 2, 2)
    assert manifest["stage"] == "00_baseline"
    assert manifest["status"] == "completed"
    assert manifest["volume_shape"] == [2, 2, 2]
    assert manifest["preview_z"] == 209
    assert manifest["z_stack_range"] == [203, 215]
    assert align_info == {"stage": "baseline", "outer_stats": []}
    assert status["stage"] == "00_baseline"
    assert status["message"] == "baseline_fbp"
    assert summary_lines[0] == "stage,status,elapsed_seconds"
    assert summary_lines[1].startswith("00_baseline,completed,")
    assert ctx.naive_slice is expected_slice
    assert ctx.saved_products[0]["input_reference"] is expected_slice
    assert ctx.saved_products[0]["suffix"] == "aligned"
    assert calls["fbp"] == {
        "geometry": "geometry",
        "grid": "grid",
        "detector": "detector",
        "projections_shape": (3, 4, 5),
        "filter_name": "ramp",
        "views_per_batch": 2,
        "checkpoint_projector": True,
        "gather_dtype": "float32",
    }
    assert calls["xy_at_global_z"] == {
        "volume_shape": (2, 2, 2),
        "grid": "grid",
        "full_nz": 96,
        "global_z": 209,
    }
    assert stage_dir / "naive_or_input_xy_global_z209.png" in rendered_pngs
    assert stage_dir / "measured_sinogram.png" in rendered_pngs


def test_cor_only_fista_stage_writes_existing_artifact_contract(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_geometry_with_axis_state(geometry, grid, detector, setup_state):
        calls["geometry_with_axis_state"] = (geometry, grid, detector, setup_state)
        return "effective-geometry"

    def fake_level_detector_grid(detector, *, state, factor):
        calls["level_detector_grid"] = (detector, state, factor)
        return "det-grid"

    def fake_fista_tv(geometry, grid, detector, projections, *, config, det_grid):
        calls["fista_tv"] = {
            "geometry": geometry,
            "grid": grid,
            "detector": detector,
            "projections_shape": tuple(np.asarray(projections).shape),
            "config": config,
            "det_grid": det_grid,
        }
        return (
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            {"loss": [10.0, 4.0], "regulariser": "huber_tv"},
        )

    monkeypatch.setattr(recon_stage, "geometry_with_axis_state", fake_geometry_with_axis_state)
    monkeypatch.setattr(recon_stage, "level_detector_grid", fake_level_detector_grid)
    monkeypatch.setattr(recon_stage, "fista_tv", fake_fista_tv)

    ctx = _Context(tmp_path, canonical_det_grid=False)
    setup_state = _SetupState()
    volume = recon_stage.run_cor_only_fista_stage(
        ctx,
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((3, 4, 5), dtype=np.float32),
        full_nz=9,
        setup_state=setup_state,
    )

    stage_dir = tmp_path / "06_cor_only_fista"
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    align_info = json.loads((stage_dir / "align_info.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    summary_lines = (stage_dir / "stage_summary.csv").read_text().splitlines()

    assert volume.shape == (2, 2, 2)
    assert (stage_dir / "cor_only_fista_fullres_slab.npy").exists()
    assert manifest["stage"] == "06_cor_only_fista"
    assert manifest["status"] == "completed"
    assert manifest["active_dofs"] == ["det_u_px"]
    assert manifest["volume_shape"] == [2, 2, 2]
    assert manifest["fista_info"] == {"loss": [10.0, 4.0], "regulariser": "huber_tv"}
    assert manifest["geometry_calibration_state"] == {
        "active_geometry_dofs": ["det_u_px"],
        "det_u_px": 1.25,
    }
    assert manifest["setup_state"] == manifest["geometry_calibration_state"]
    assert set(manifest["artifacts"]) == {"aligned_xy", "delta_xy", "orthos", "z_stack"}
    assert align_info == {"fista_info": {"loss": [10.0, 4.0], "regulariser": "huber_tv"}}
    assert status["stage"] == "06_cor_only_fista"
    assert summary_lines[0] == "stage,status,elapsed_seconds,loss_first,loss_last"
    assert summary_lines[1].startswith("06_cor_only_fista,completed,")
    assert summary_lines[1].endswith(",10.0,4.0")
    assert ctx.saved_products[0]["input_reference"] is ctx.naive_slice
    assert ctx.saved_products[0]["suffix"] == "aligned"
    assert calls["level_detector_grid"] == ("detector", setup_state, 1)
    assert calls["fista_tv"]["det_grid"] == "det-grid"
    assert calls["fista_tv"]["config"].iters == 7
    assert calls["fista_tv"]["config"].lambda_tv == 0.002
    assert calls["fista_tv"]["config"].views_per_batch == 2
    assert calls["fista_tv"]["config"].checkpoint_projector is True


def test_cor_only_fista_stage_preserves_canonical_detector_grid_policy(
    monkeypatch,
    tmp_path,
) -> None:
    def fail_level_detector_grid(*_args, **_kwargs):
        raise AssertionError("canonical detector-grid runs should not fold a detector grid")

    observed: dict[str, Any] = {}

    def fake_fista_tv(*_args, det_grid, **_kwargs):
        observed["det_grid"] = det_grid
        return np.zeros((1, 1, 1), dtype=np.float32), {"loss": [1.0]}

    monkeypatch.setattr(recon_stage, "geometry_with_axis_state", lambda geometry, *_args: geometry)
    monkeypatch.setattr(recon_stage, "level_detector_grid", fail_level_detector_grid)
    monkeypatch.setattr(recon_stage, "fista_tv", fake_fista_tv)

    recon_stage.run_cor_only_fista_stage(
        _Context(tmp_path, canonical_det_grid=True),
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((1, 1, 1), dtype=np.float32),
        full_nz=1,
        setup_state=_SetupState(),
    )

    assert observed["det_grid"] is None


def test_final_reconstruction_stage_writes_artifact_contract(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, Any] = {}

    class EffectiveGeometry:
        def pose_for_view(self, i):
            calls["pose_for_view"] = i
            return np.eye(4, dtype=np.float32)

        def rays_for_view(self, i):
            return f"rays-{i}"

    def fake_geometry_with_axis_state(geometry, grid, detector, setup_state):
        calls["geometry_with_axis_state"] = (geometry, grid, detector, setup_state)
        return EffectiveGeometry()

    def fake_level_detector_grid(detector, *, state, factor):
        calls["level_detector_grid"] = (detector, state, factor)
        return "det-grid"

    def fake_se3_from_5d(params5):
        calls["se3_from_5d"] = np.asarray(params5).copy()
        return np.eye(4, dtype=np.float32)

    def fake_fista_tv(geometry, grid, detector, projections, *, config, det_grid):
        calls["fista_tv"] = {
            "pose_for_view": geometry.pose_for_view(0),
            "rays_for_view": geometry.rays_for_view(0),
            "grid": grid,
            "detector": detector,
            "projections_shape": tuple(np.asarray(projections).shape),
            "config": config,
            "det_grid": det_grid,
        }
        return (
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            {"loss": [9.0, 2.0], "regulariser": "huber_tv"},
        )

    monkeypatch.setattr(recon_stage, "geometry_with_axis_state", fake_geometry_with_axis_state)
    monkeypatch.setattr(recon_stage, "level_detector_grid", fake_level_detector_grid)
    monkeypatch.setattr(recon_stage, "se3_from_5d", fake_se3_from_5d)
    monkeypatch.setattr(recon_stage, "fista_tv", fake_fista_tv)

    ctx = _Context(tmp_path, canonical_det_grid=False)
    ctx.args.views_per_batch = 0
    setup_state = _SetupState()
    params5 = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    volume = recon_stage.run_final_reconstruction_stage(
        ctx,
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((1, 2, 3), dtype=np.float32),
        full_nz=9,
        setup_state=setup_state,
        params5=params5,
    )

    stage_dir = tmp_path / "05_final"
    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    align_info = json.loads((stage_dir / "align_info.json").read_text())
    geometry_state = json.loads((stage_dir / "geometry_calibration_state.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())

    assert volume.shape == (2, 2, 2)
    assert np.load(stage_dir / "final_setup_aligned_slab.npy").shape == (2, 2, 2)
    assert manifest["stage"] == "05_final"
    assert manifest["status"] == "completed"
    assert manifest["recon_info"] == {"loss": [9.0, 2.0], "regulariser": "huber_tv"}
    assert manifest["geometry_calibration_state"] == {
        "active_geometry_dofs": ["det_u_px"],
        "det_u_px": 1.25,
    }
    assert align_info["recon_info"] == manifest["recon_info"]
    assert geometry_state == manifest["geometry_calibration_state"]
    assert set(manifest["artifacts"]) == {"aligned_xy", "delta_xy", "orthos", "z_stack"}
    assert status["stage"] == "05_final"
    assert status["message"] == "final_fista_tv"
    assert calls["level_detector_grid"] == ("detector", setup_state, 1)
    assert calls["fista_tv"]["det_grid"] == "det-grid"
    assert calls["fista_tv"]["config"].views_per_batch is None
    assert np.allclose(calls["fista_tv"]["pose_for_view"], np.eye(4, dtype=np.float32))
    assert calls["fista_tv"]["rays_for_view"] == "rays-0"
    assert np.allclose(calls["se3_from_5d"], params5[0])
    assert ctx.saved_products[0]["input_reference"] is ctx.naive_slice
    assert ctx.saved_products[0]["suffix"] == "aligned"


def test_best_final_reconstruction_stage_scores_candidates_and_publishes_best(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_final_reconstruction(
        ctx,
        *,
        geometry,
        grid,
        detector,
        projections,
        full_nz,
        setup_state,
        params5,
    ):
        stage_dir = ctx.stage_dir("05_final")
        stage_dir.mkdir(parents=True, exist_ok=True)
        loss = float(setup_state["loss"])
        recon_stage.write_real_lamino_json(
            stage_dir / "stage_manifest.json",
            {"stage": "05_final", "recon_info": {"loss": [10.0, loss]}},
        )
        calls.append(
            {
                "geometry": geometry,
                "grid": grid,
                "detector": detector,
                "projections_shape": tuple(np.asarray(projections).shape),
                "full_nz": full_nz,
                "setup_state": setup_state,
                "params5": np.asarray(params5).copy(),
            }
        )
        return np.full((2, 2, 2), loss, dtype=np.float32)

    monkeypatch.setattr(
        recon_stage,
        "validate_real_lamino_stage_output",
        lambda *_args, **_kwargs: {"passed": True, "failures": []},
    )
    monkeypatch.setattr(recon_stage, "run_final_reconstruction_stage", fake_final_reconstruction)

    ctx = _Context(tmp_path, canonical_det_grid=False)
    original_sentinel_dir = ctx.stage_dir("sentinel")
    final_volume, choice = recon_stage.run_best_final_reconstruction_stage(
        ctx,
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((2, 3, 4), dtype=np.float32),
        full_nz=96,
        candidates=[
            {
                "label": "01_setup",
                "source_stage": "01_setup",
                "setup_state": {"loss": 8.0},
                "params5": np.ones((2, 5), dtype=np.float32),
            },
            {
                "label": "04_polish",
                "source_stage": "04_polish",
                "setup_state": {"loss": 3.0},
                "params5": np.full((2, 5), 2.0, dtype=np.float32),
            },
        ],
    )

    final_dir = tmp_path / "05_final"
    manifest = json.loads((final_dir / "stage_manifest.json").read_text())

    assert np.all(final_volume == 3.0)
    assert choice["label"] == "04_polish"
    assert choice["source_stage"] == "04_polish"
    assert choice["loss_last"] == 3.0
    assert manifest["selected_final_candidate"]["label"] == "04_polish"
    assert manifest["selected_final_candidate"]["candidate_policy"] == "all"
    assert [item["label"] for item in manifest["selected_final_candidate"]["candidates"]] == [
        "01_setup",
        "04_polish",
    ]
    assert calls[0]["setup_state"] == {"loss": 8.0}
    assert calls[1]["setup_state"] == {"loss": 3.0}
    assert np.asarray(calls[1]["params5"]).shape == (2, 5)
    assert ctx.stage_dir("sentinel") == original_sentinel_dir


def test_best_final_reconstruction_stage_honors_last_valid_candidate_policy(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[float] = []

    def fake_final_reconstruction(ctx, **kwargs):
        stage_dir = ctx.stage_dir("05_final")
        stage_dir.mkdir(parents=True, exist_ok=True)
        loss = float(np.asarray(kwargs["params5"])[0, 0])
        calls.append(loss)
        (stage_dir / "stage_manifest.json").write_text(
            json.dumps(
                {
                    "stage": "05_final",
                    "status": "completed",
                    "recon_info": {"loss": [loss]},
                }
            )
        )
        return np.full((1, 1, 1), loss, dtype=np.float32)

    ctx = _FinalContext(tmp_path / "run", final_candidate_policy="last_valid")
    ctx.run_root.mkdir()
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
            "params5": np.asarray([[6.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
    ]
    monkeypatch.setattr(
        recon_stage,
        "run_final_reconstruction_stage",
        fake_final_reconstruction,
    )

    volume, choice = recon_stage.run_best_final_reconstruction_stage(
        ctx,
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.zeros((1, 1, 1), dtype=np.float32),
        full_nz=1,
        candidates=candidates,
    )

    manifest = json.loads((ctx.run_root / "05_final" / "stage_manifest.json").read_text())
    assert calls == [6.0]
    assert choice["label"] == "last"
    assert float(volume[0, 0, 0]) == 6.0
    assert manifest["selected_final_candidate"]["candidate_policy"] == "last_valid"
    assert [item["label"] for item in manifest["selected_final_candidate"]["candidates"]] == [
        "last",
    ]
