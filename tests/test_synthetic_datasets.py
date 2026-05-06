from __future__ import annotations

# pyright: reportAny=false, reportUnknownArgumentType=false
import csv
import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tomojax.datasets import (
    generate_synthetic_dataset,
    load_synthetic128_specs,
    make_benchmark_phantom,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_synthetic128_specs_load_all_manifest_datasets() -> None:
    specs = load_synthetic128_specs()

    assert sorted(specs) == [
        "synth128_combined_nuisance_jumps",
        "synth128_lamino_axis_roll_pose",
        "synth128_pose_random_extreme",
        "synth128_setup_global_tomo",
        "synth128_thermal_object_drift",
    ]
    assert specs["synth128_setup_global_tomo"].detector_shape == (160, 160)
    assert specs["synth128_lamino_axis_roll_pose"].mode == "parallel_laminography"


def test_benchmark_phantom_is_deterministic_and_structured() -> None:
    first = make_benchmark_phantom(32, seed=1001)
    second = make_benchmark_phantom(32, seed=1001)
    different = make_benchmark_phantom(32, seed=1002)

    assert first.shape == (32, 32, 32)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert float(first.max()) > 0.5
    assert np.count_nonzero(first) > 1000


def test_generate_synthetic_dataset_writes_deterministic_smoke_artifacts(tmp_path) -> None:
    first = generate_synthetic_dataset("synth128_setup_global_tomo", tmp_path, size=32, clean=True)
    first_projection_hash = _array_hash(first.projections)
    second = generate_synthetic_dataset("synth128_setup_global_tomo", tmp_path, size=32, clean=True)

    assert first.dataset_dir == second.dataset_dir
    assert _array_hash(second.projections) == first_projection_hash

    expected = {
        first.manifest,
        first.volume,
        first.projections,
        first.mask,
        first.nominal_geometry,
        first.corrupted_geometry,
        first.true_geometry,
        first.true_pose,
        first.true_motion,
        first.nuisance_truth,
        first.noise_truth,
    }
    assert all(path.is_file() for path in expected)

    manifest = cast("dict[str, Any]", json.loads(first.manifest.read_text(encoding="utf-8")))
    assert manifest["name"] == "synth128_setup_global_tomo"
    assert manifest["volume_shape"] == [32, 32, 32]
    assert manifest["detector_shape"] == [40, 40]
    assert manifest["views"] == 16
    assert "det_u_error_px_lt" in manifest["recovery_tolerances"]

    projections = np.load(first.projections)
    mask = np.load(first.mask)
    assert projections.shape == (16, 40, 40)
    assert projections.dtype == np.float32
    assert mask.shape == projections.shape
    assert mask.dtype == np.bool_

    true_geometry = cast(
        "dict[str, Any]", json.loads(first.true_geometry.read_text(encoding="utf-8"))
    )
    nominal_geometry = cast(
        "dict[str, Any]", json.loads(first.nominal_geometry.read_text(encoding="utf-8"))
    )
    assert true_geometry["setup"]["det_u_px"] == 14.5
    assert nominal_geometry["setup"]["det_u_px"] == 0.0

    with first.true_pose.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 16
    assert set(rows[0]) == {
        "view",
        "theta_deg",
        "alpha_deg",
        "beta_deg",
        "phi_residual_deg",
        "dx_px",
        "dz_px",
    }


def test_generate_synthetic_dataset_applies_gain_offset_nuisance(tmp_path: Path) -> None:
    clean = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "clean",
        size=32,
        clean=True,
        views=8,
    )
    drifted = generate_synthetic_dataset(
        "synth128_thermal_object_drift",
        tmp_path / "drifted",
        size=32,
        clean=False,
        views=8,
    )

    clean_projections = np.load(clean.projections)
    drifted_projections = np.load(drifted.projections)
    assert not np.array_equal(clean_projections, drifted_projections)
    nuisance = cast(
        "dict[str, Any]",
        json.loads(drifted.nuisance_truth.read_text(encoding="utf-8")),
    )
    assert nuisance["schema"] == "tomojax.synthetic_nuisance_truth.v1"
    assert nuisance["applied_terms"] == {
        "gain": True,
        "offset": True,
        "background_vertical_gradient": False,
    }
    gain = cast("list[float]", nuisance["gain"])
    offset = cast("list[float]", nuisance["offset"])
    assert len(gain) == 8
    assert len(offset) == 8
    np.testing.assert_allclose(gain[0], 0.98, atol=1.0e-6)
    np.testing.assert_allclose(gain[-1], 1.03, atol=1.0e-6)
    np.testing.assert_allclose(offset[0], -0.015, atol=1.0e-6)
    np.testing.assert_allclose(offset[-1], 0.015, atol=1.0e-6)
    expected_first = clean_projections[0] * np.float32(gain[0]) + np.float32(offset[0])
    np.testing.assert_allclose(drifted_projections[0], expected_first, atol=1.0e-6)


def test_generate_clean_synthetic_dataset_records_but_skips_nuisance(tmp_path: Path) -> None:
    clean = generate_synthetic_dataset(
        "synth128_lamino_axis_roll_pose",
        tmp_path,
        size=32,
        clean=True,
        views=8,
    )
    dirty = generate_synthetic_dataset(
        "synth128_lamino_axis_roll_pose",
        tmp_path / "dirty",
        size=32,
        clean=False,
        views=8,
    )

    clean_projections = np.load(clean.projections)
    dirty_projections = np.load(dirty.projections)
    assert not np.array_equal(clean_projections, dirty_projections)
    nuisance = cast(
        "dict[str, Any]",
        json.loads(clean.nuisance_truth.read_text(encoding="utf-8")),
    )
    assert nuisance["applied_terms"] == {
        "gain": True,
        "offset": False,
        "background_vertical_gradient": False,
    }


def test_generate_synthetic_dataset_applies_vertical_background_nuisance(
    tmp_path: Path,
) -> None:
    clean = generate_synthetic_dataset(
        "synth128_combined_nuisance_jumps",
        tmp_path / "clean",
        size=32,
        clean=True,
        views=8,
    )
    drifted = generate_synthetic_dataset(
        "synth128_combined_nuisance_jumps",
        tmp_path / "drifted",
        size=32,
        clean=False,
        views=8,
    )

    clean_projections = np.load(clean.projections)
    drifted_projections = np.load(drifted.projections)
    nuisance = cast(
        "dict[str, Any]",
        json.loads(drifted.nuisance_truth.read_text(encoding="utf-8")),
    )
    assert nuisance["applied_terms"] == {
        "gain": True,
        "offset": False,
        "background_vertical_gradient": True,
    }
    gain = np.asarray(cast("list[float]", nuisance["gain"]), dtype=np.float32)
    gradient = np.asarray(
        cast("list[float]", nuisance["background_vertical_gradient"]),
        dtype=np.float32,
    )
    assert gradient.shape == (8,)
    assert float(np.max(np.abs(gradient))) > 0.0

    view_index = int(np.argmax(np.abs(gradient)))
    vertical = np.linspace(-1.0, 1.0, drifted_projections.shape[1], dtype=np.float32)
    expected = (
        clean_projections[view_index] * gain[view_index] + gradient[view_index] * vertical[:, None]
    )
    np.testing.assert_allclose(drifted_projections[view_index], expected, atol=1.0e-6)

    clean_nuisance = cast(
        "dict[str, Any]",
        json.loads(clean.nuisance_truth.read_text(encoding="utf-8")),
    )
    assert clean_nuisance["applied_terms"]["background_vertical_gradient"] is True
    np.testing.assert_array_equal(clean_projections, np.load(clean.projections))


def _array_hash(path: Path) -> str:
    data = np.ascontiguousarray(np.load(path))
    return hashlib.sha256(data.tobytes()).hexdigest()
