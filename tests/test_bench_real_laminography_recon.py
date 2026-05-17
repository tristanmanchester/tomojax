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
