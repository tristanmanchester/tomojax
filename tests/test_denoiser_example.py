"""The optional learned-reconstruction example keeps its test views untouched."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import tomojax as tj


def example():
    pytest.importorskip("torch")
    pytest.importorskip("tomojax.torch")
    path = Path(__file__).resolve().parents[1] / "examples" / "train_denoiser.py"
    spec = importlib.util.spec_from_file_location("train_denoiser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_view_splits_are_disjoint_and_angle_interleaved():
    module = example()
    angles = np.array([90, 0, 150, 30, 120, 60, 165, 15, 45, 75, 105, 135])
    geometry = tj.ParallelGeometry(tj.Grid(8, 8, 2, 1, 1, 1), tj.Detector(8, 2, 1, 1), angles)
    data = np.broadcast_to(np.arange(len(angles))[:, None, None], (len(angles), 2, 8))
    groups = module.split_views(tj.Scan(data, geometry))
    for k, group in enumerate(groups):
        expected = np.sort(np.argsort(angles, kind="stable")[k::3])
        np.testing.assert_array_equal(group.projections[:, 0, 0], expected)
        np.testing.assert_array_equal(group.geometry.angles, angles[expected])
    np.testing.assert_array_equal(
        np.sort(np.concatenate([g.projections[:, 0, 0] for g in groups])), np.arange(12)
    )


@pytest.mark.gpu
def test_training_uses_projector_gradients_but_not_test_targets():
    module = example()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch required")
    angles = np.linspace(0, 180, 24, endpoint=False)
    geometry = tj.ParallelGeometry(tj.Grid(12, 12, 2, 1, 1, 1), tj.Detector(12, 2, 1, 1), angles)
    axis = np.arange(12) - 5.5
    volume = np.exp(-(axis[:, None, None] ** 2 + axis[None, :, None] ** 2) / 8)
    volume = np.broadcast_to(volume, (12, 12, 2)).astype(np.float32)
    scan = tj.Scan(np.asarray(tj.project(geometry, volume)), geometry)
    torch.manual_seed(2)
    baseline, denoised, model, metrics = module.train(scan, steps=2, learning_rate=1e-3)
    assert torch.isfinite(denoised).all()
    assert not torch.equal(baseline, denoised)
    assert np.isfinite(list(metrics.values())).all()
    changed = np.array(scan.projections)
    changed[2::3] *= 2
    torch.manual_seed(2)
    _, second, other, other_metrics = module.train(
        replace(scan, projections=changed), steps=2, learning_rate=1e-3
    )
    torch.testing.assert_close(second, denoised)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert metrics["held_out_denoised_nrmse"] != other_metrics["held_out_denoised_nrmse"]
