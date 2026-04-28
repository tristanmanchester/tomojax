from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_real_lamino_native_setup_pose_256.py"
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
