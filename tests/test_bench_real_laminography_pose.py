from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import numpy as np

import tomojax.bench.real_laminography_pose as pose_stage


class _CalibrationState:
    def to_dict(self) -> dict[str, Any]:
        return {"det_u_px": 1.25, "active_geometry_dofs": ["det_u_px"]}


class _SetupState:
    def to_calibration_state(self) -> _CalibrationState:
        return _CalibrationState()


class _Grid:
    nx = 2
    ny = 2
    nz = 2


class _Context:
    def __init__(self, root: Path, *, canonical_det_grid: bool = False) -> None:
        self.args = Namespace(
            align_profile="lightning",
            canonical_det_grid=canonical_det_grid,
            early_stop=True,
            early_stop_patience=2,
            early_stop_rel=1e-3,
            fallback_policy="fallback",
            fold_rigid_detector_grid=True,
            gather_dtype="float32",
            gn_damping=1e-3,
            knot_spacing=8,
            lambda_tv=0.002,
            outer_iters=3,
            pose_degree=3,
            pose_model="spline",
            projector_backend="jax",
            quality_tier="fast",
            recon_iters=7,
            recon_positivity=True,
            regulariser="huber_tv",
            snapshot_max_cols=4,
            tv_prox_iters=3,
            views_per_batch=2,
        )
        self.run_root = root
        self.status_path = root / "status.json"
        self.naive_slice = np.zeros((2, 2), dtype=np.float32)
        self.preview_global_z = 209
        self.stack_z_range = (203, 215)
        self.saved_products: list[dict[str, Any]] = []

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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
        return {key: str(path) for key, path in products.items()}


def test_pose_stage_writes_existing_artifact_contract(monkeypatch, tmp_path) -> None:
    calls: dict[str, Any] = {}
    rendered_pngs: list[Path] = []

    def fake_scale_grid(grid, factor):
        calls.setdefault("scale_grid", []).append((grid, factor))
        return _Grid()

    def fake_scale_detector(detector, factor):
        calls["scale_detector"] = (detector, factor)
        return "scaled-detector"

    def fake_bin_projections(projections, factor):
        calls["bin_projections"] = (tuple(np.asarray(projections).shape), factor)
        return np.asarray(projections)

    def fake_geometry_with_axis_state(geometry, grid, detector, setup_state):
        calls["geometry_with_axis_state"] = (geometry, grid, detector, setup_state)
        return "effective-geometry"

    def fake_level_detector_grid(detector, *, state, factor):
        calls["level_detector_grid"] = (detector, state, factor)
        return "det-grid"

    def fake_xy_at_global_z(volume, *, grid, full_nz, global_z):
        calls["xy_at_global_z"] = {
            "volume_shape": tuple(np.asarray(volume).shape),
            "grid": grid,
            "full_nz": full_nz,
            "global_z": global_z,
        }
        return np.ones((2, 2), dtype=np.float32)

    def fake_save_uint8_png(path, image):
        rendered_pngs.append(Path(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(np.asarray(image).tobytes())

    def fake_save_z_stack(path, volume, *, grid, full_nz, z_range, max_cols):
        calls["save_z_stack"] = {
            "volume_shape": tuple(np.asarray(volume).shape),
            "grid": grid,
            "full_nz": full_nz,
            "z_range": z_range,
            "max_cols": max_cols,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"stack")

    def fake_align(
        geometry,
        grid,
        detector,
        projections,
        *,
        cfg,
        init_x,
        init_params5,
        observer,
        det_grid_override,
    ):
        calls["align"] = {
            "geometry": geometry,
            "grid": grid,
            "detector": detector,
            "projections_shape": tuple(np.asarray(projections).shape),
            "cfg": cfg,
            "init_x": init_x,
            "init_params5_shape": tuple(np.asarray(init_params5).shape),
            "det_grid_override": det_grid_override,
        }
        volume = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        params = np.full((2, 5), 0.25, dtype=np.float32)
        action = observer(
            volume,
            params,
            {
                "outer_idx": 1,
                "loss_before": 4.0,
                "loss_after": 2.0,
                "rms_update": 0.1,
                "accepted": True,
                "cumulative_time": 1.5,
            },
        )
        assert action == "continue"
        return volume, params, {"outer_stats": [{"loss_after": 2.0}]}

    monkeypatch.setattr(pose_stage, "scale_grid", fake_scale_grid)
    monkeypatch.setattr(pose_stage, "scale_detector", fake_scale_detector)
    monkeypatch.setattr(pose_stage, "bin_projections", fake_bin_projections)
    monkeypatch.setattr(pose_stage, "geometry_with_axis_state", fake_geometry_with_axis_state)
    monkeypatch.setattr(pose_stage, "level_detector_grid", fake_level_detector_grid)
    monkeypatch.setattr(pose_stage, "real_lamino_xy_at_global_z", fake_xy_at_global_z)
    monkeypatch.setattr(pose_stage, "save_uint8_png", fake_save_uint8_png)
    monkeypatch.setattr(pose_stage, "save_real_lamino_z_stack", fake_save_z_stack)
    monkeypatch.setattr(pose_stage, "align", fake_align)

    ctx = _Context(tmp_path)
    setup_state = _SetupState()
    stage_dir = tmp_path / "02_pose_phi"
    volume, params, stats = pose_stage.run_real_lamino_pose_stage(
        ctx,
        stage_dir=stage_dir,
        stage_name="02_pose_phi",
        active_pose=("phi",),
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((2, 3, 4), dtype=np.float32),
        full_nz=9,
        setup_state=setup_state,
        params5=np.zeros((2, 5), dtype=np.float32),
        levels=(1,),
        bounds="phi=-2:2",
    )

    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    align_info = json.loads((stage_dir / "align_info.json").read_text())
    level_info = json.loads((stage_dir / "level_01_align_info.json").read_text())
    geometry_state = json.loads((stage_dir / "geometry_calibration_state.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    summary_lines = (stage_dir / "stage_summary.csv").read_text().splitlines()

    assert volume.shape == (2, 2, 2)
    assert params.shape == (2, 5)
    assert len(stats) == 1
    assert (stage_dir / "params.csv").exists()
    assert (stage_dir / "checkpoints" / "outer_001_level01_iter01.npz").exists()
    assert (stage_dir / "checkpoints" / "latest.npz").exists()
    assert manifest["stage"] == "02_pose_phi"
    assert manifest["status"] == "completed"
    assert manifest["active_dofs"] == ["phi"]
    assert manifest["levels"] == [1]
    assert manifest["bounds"] == "phi=-2:2"
    assert manifest["stats_count"] == 1
    assert set(manifest["artifacts"]) == {"aligned_xy", "delta_xy", "orthos", "z_stack"}
    assert align_info["outer_stats"][0]["active_dofs"] == "phi"
    assert align_info["params_summary"] == manifest["params_summary"]
    assert level_info == {"outer_stats": [{"loss_after": 2.0}]}
    assert geometry_state == {"active_geometry_dofs": ["det_u_px"], "det_u_px": 1.25}
    assert status["stage"] == "02_pose_phi"
    assert status["active_dofs"] == ["phi"]
    assert summary_lines[0] == (
        "stage,level_factor,outer_idx,loss_before,loss_after,rms_update,"
        "accepted,active_dofs,cumulative_time"
    )
    assert summary_lines[1].startswith("02_pose_phi,1,1,4.0,2.0,0.1,True,phi,1.5")
    assert calls["level_detector_grid"] == ("scaled-detector", setup_state, 1)
    assert calls["align"]["det_grid_override"] == "det-grid"
    assert calls["align"]["cfg"].optimise_dofs == ("phi",)
    assert calls["align"]["cfg"].bounds == (("phi", -2.0, 2.0),)
    assert calls["align"]["cfg"].gauge_fix == "none"
    assert calls["xy_at_global_z"]["global_z"] == 209
    assert calls["save_z_stack"]["z_range"] == (203, 215)
    assert ctx.saved_products[0]["input_reference"] is ctx.naive_slice
    assert (
        stage_dir / "timeline_z" / "slices" / "outer_001_level01_iter01_global_z209.png"
        in rendered_pngs
    )


def test_pose_stage_stops_on_checkpoint_validation_failure(monkeypatch, tmp_path) -> None:
    def fail_level_detector_grid(*_args, **_kwargs):
        raise AssertionError("canonical detector-grid policy should pass det_grid_override=None")

    observed: dict[str, Any] = {}

    monkeypatch.setattr(pose_stage, "scale_grid", lambda _grid, _factor: _Grid())
    monkeypatch.setattr(pose_stage, "scale_detector", lambda _detector, _factor: "scaled-detector")
    monkeypatch.setattr(
        pose_stage, "bin_projections", lambda projections, _factor: np.asarray(projections)
    )
    monkeypatch.setattr(pose_stage, "geometry_with_axis_state", lambda geometry, *_args: geometry)
    monkeypatch.setattr(pose_stage, "level_detector_grid", fail_level_detector_grid)
    monkeypatch.setattr(
        pose_stage,
        "real_lamino_xy_at_global_z",
        lambda *_args, **_kwargs: np.ones((1, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        pose_stage,
        "save_uint8_png",
        lambda path, _image: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"png"),
        ),
    )
    monkeypatch.setattr(
        pose_stage,
        "save_real_lamino_z_stack",
        lambda path, *_args, **_kwargs: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"stack"),
        ),
    )

    def fake_align(*_args, observer, det_grid_override, **_kwargs):
        observed["det_grid_override"] = det_grid_override
        volume = np.asarray([[[np.nan]]], dtype=np.float32)
        params = np.zeros((1, 5), dtype=np.float32)
        action = observer(volume, params, {"outer_idx": 1, "loss_before": "nan"})
        assert action == "stop_run"
        return (
            volume,
            params,
            {"stopped_by_observer": True, "observer_action": "stop_run"},
        )

    monkeypatch.setattr(pose_stage, "align", fake_align)

    ctx = _Context(tmp_path, canonical_det_grid=True)
    stage_dir = tmp_path / "02_pose_phi"
    volume, _params, stats = pose_stage.run_real_lamino_pose_stage(
        ctx,
        stage_dir=stage_dir,
        stage_name="02_pose_phi",
        active_pose=("phi",),
        geometry="geometry",
        grid="grid",
        detector="detector",
        projections=np.ones((1, 1, 1), dtype=np.float32),
        full_nz=1,
        setup_state=_SetupState(),
        params5=np.zeros((1, 5), dtype=np.float32),
        levels=(1, 2),
        bounds="phi=-2:2",
    )

    manifest = json.loads((stage_dir / "stage_manifest.json").read_text())
    assert np.isnan(volume).all()
    assert observed["det_grid_override"] is None
    assert len(stats) == 1
    assert stats[0]["checkpoint_validation_failed"] is True
    assert "x finite fraction is 0" in stats[0]["checkpoint_validation_failures"]
    assert "loss_before is non-finite" in stats[0]["checkpoint_validation_failures"]
    assert manifest["stats_count"] == 1
    assert manifest["levels"] == [1, 2]
