from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

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
_SPEC = importlib.util.spec_from_file_location("run_real_lamino_native_setup_pose_256", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


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
