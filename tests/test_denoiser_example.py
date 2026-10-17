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


def test_loss_balance_is_invariant_to_attenuation_units():
    module = example()
    torch = pytest.importorskip("torch")
    inputs = [torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8, torch.ones(2, 3)]
    operators = [lambda value: value * 2, lambda value: value * 3]
    targets = [torch.full((2, 3), value) for value in (0.4, 0.7)]
    scale = torch.tensor(2.0)
    norm = torch.stack([target.square().mean() for target in targets]).mean()
    weight = module.projection_loss_weight(inputs, operators, targets, scale, norm)
    rescaled = module.projection_loss_weight(
        inputs, operators, [target * 7 for target in targets], scale * 7, norm * 49
    )
    image_norm = (inputs[0] - inputs[1]).square().mean().sqrt()
    projection_norms = []
    for source, factor in enumerate((3, 2)):
        gradient = (factor * inputs[source] * scale - targets[1 - source]) * factor * scale / norm
        projection_norms.append(gradient.square().mean().sqrt())
    projection_norm = torch.stack(projection_norms).mean()
    torch.testing.assert_close(weight, image_norm / projection_norm)
    torch.testing.assert_close(weight, rescaled)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_loss_balance_is_finite_at_an_exact_fit(device):
    module = example()
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch required")
    images = [torch.ones(2, 3, device=device)] * 2
    scale = torch.tensor(1.0, device=device)
    weight = module.projection_loss_weight(images, [lambda value: value] * 2, images, scale, scale)
    assert weight == 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_denoiser_identity_and_slice_batch_gradients(device, monkeypatch):
    module = example()
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch required")
    volume = torch.arange(192, dtype=torch.float32, device=device).reshape(8, 8, 3) / 192
    model = module.Denoiser().to(device=device, memory_format=torch.channels_last)
    output = model(volume)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, volume, rtol=0, atol=0)
    with torch.no_grad():
        model.layers[-1].weight.fill_(0.125)
    volume.requires_grad_()
    parameters = (volume, *model.parameters())
    full = model(volume)
    full_gradients = torch.autograd.grad(full.sum(), parameters)
    monkeypatch.setattr(model, "_slice_batch_size", lambda _: 2)
    monkeypatch.setattr(model, "_checkpoint_layers", lambda _: True)
    split = model(volume)
    split_gradients = torch.autograd.grad(split.sum(), parameters)
    torch.testing.assert_close(split, full, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(split_gradients, full_gradients, rtol=2e-3, atol=2e-4)
