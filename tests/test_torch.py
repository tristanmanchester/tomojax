import jax
import jax.dlpack
import numpy as np
import pytest

import tomojax as tj

torch = pytest.importorskip("torch")
Projector = pytest.importorskip("tomojax.torch").Projector

pytestmark = pytest.mark.gpu


def geometry(kind):
    grid = tj.Grid(3, 4, 2, 0.8, 1.1, 1.3)
    detector = tj.Detector(5, 3, 0.9, 1.2, (0.2, -0.1))
    if kind == "axial":
        grid = tj.Grid(3, 4, 4, 0.8, 1.1, 1.3)
        detector = tj.Detector(5, 4, 0.9, 1.3, (0.2, 0))
    angles = [7.0, 41.0, 93.0]
    if kind == "slice":
        grid = tj.Grid(3, 4, 1, 0.8, 1.1, 1.3)
        detector = tj.Detector(5, 1, 0.9, 1.3)
    if kind == "cone":
        return tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(15.0, 25.0))
    if kind == "lamino":
        return tj.LaminographyGeometry(grid, detector, angles, tilt_deg=25)
    return tj.ParallelGeometry(grid, detector, angles)


@pytest.mark.parametrize("kind", ["parallel", "cone", "lamino", "axial", "slice"])
@pytest.mark.filterwarnings("ignore:Input #0 requires gradient and is not a double precision")
def test_matched_autograd(kind, monkeypatch):
    if kind == "axial":
        monkeypatch.setenv("TOMOJAX_CUDA_KERNELS", "1")
    geom = geometry(kind)
    operator = Projector(geom)
    torch.manual_seed(3)
    x = torch.rand(operator.volume_shape, device="cuda", requires_grad=True)
    y = torch.rand(operator.projection_shape, device="cuda", requires_grad=True)
    if kind == "slice":
        x, y = x.squeeze(0), y.squeeze(1)
    projected, transposed = operator(x), operator.adjoint(y)
    expected = tj.project(geom, jax.dlpack.from_dlpack(x.detach(), copy=False))
    np.testing.assert_allclose(projected.detach().cpu(), np.asarray(expected), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close((projected * y).sum(), (x * transposed).sum(), rtol=2e-6, atol=2e-6)
    # CUDA projectors use float32: choose perturbations above FP32 rounding.
    for function, value in [(operator, x), (operator.adjoint, y)]:
        assert torch.autograd.gradcheck(function, (value,), eps=1e-2, atol=2e-4, rtol=2e-3)
    gradient = torch.autograd.grad(projected.square().sum() / 2, x, create_graph=True)[0]
    hessian_ones = torch.autograd.grad(gradient.sum(), x)[0]
    torch.testing.assert_close(hessian_ones, operator.adjoint(operator(torch.ones_like(x))))


def test_dlpack_shares_storage_and_respects_torch_stream(monkeypatch):
    incoming, outgoing = [], []
    from_torch, from_jax = jax.dlpack.from_dlpack, torch.utils.dlpack.from_dlpack

    def import_tensor(value, **kwargs):
        array = from_torch(value, **kwargs)
        incoming.append((value.data_ptr(), array.unsafe_buffer_pointer()))
        return array

    def export_array(array):
        value = from_jax(array)
        outgoing.append((array.unsafe_buffer_pointer(), value.data_ptr()))
        return value

    monkeypatch.setattr(jax.dlpack, "from_dlpack", import_tensor)
    monkeypatch.setattr(torch.utils.dlpack, "from_dlpack", export_array)
    operator = Projector(geometry("cone"))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.ones(operator.volume_shape, device="cuda", requires_grad=True)
        y = operator(x * 2)
        y.sum().backward()  # backward's expanded gradient must be materialized
        actual = y * 3
    stream.synchronize()
    expected = operator(torch.full_like(x, 2)) * 3
    torch.testing.assert_close(actual, expected)
    assert incoming and outgoing
    assert all(a == b for a, b in incoming + outgoing)
    with pytest.raises(ValueError, match="contiguous"):
        operator(torch.zeros(3, 4, 2, device="cuda").permute(2, 1, 0))
    with pytest.raises(ValueError, match="float32"):
        operator(x.double())
