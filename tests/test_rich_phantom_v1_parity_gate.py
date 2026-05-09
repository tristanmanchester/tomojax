from __future__ import annotations

# pyright: reportAny=false, reportPrivateUsage=false, reportUnknownMemberType=false
import importlib.util
import json
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np

from tomojax.datasets import generate_synthetic_dataset
from tomojax.forward import project_parallel_reference
from tomojax.geometry import read_geometry_json, read_pose_params_csv


def _load_gate_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_rich_phantom_v1_parity_gate.py"
    spec = importlib.util.spec_from_file_location("run_rich_phantom_v1_parity_gate", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load rich phantom parity gate module")
    module = importlib.util.module_from_spec(spec)
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
