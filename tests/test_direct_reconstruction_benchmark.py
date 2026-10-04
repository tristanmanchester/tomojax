"""Independent physical-scale checks for external direct reconstruction adapters."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest


@pytest.fixture
def adapters(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    return importlib.import_module("direct_reconstruction")


@pytest.mark.parametrize("nu", [1, 7, 32])
def test_external_filter_matches_finite_linear_ramp_convolution(adapters, nu):
    spacing = 0.63
    impulse = adapters.ramp_impulse(nu, spacing)
    data = np.random.default_rng(131).normal(size=(3, nu))
    filtered = np.fft.irfft(
        np.fft.rfft(data, n=len(impulse)) * np.fft.rfft(impulse), n=len(impulse)
    )[..., :nu]
    distance = np.arange(nu)[:, None] - np.arange(nu)[None, :]
    matrix = np.zeros((nu, nu))
    matrix[distance == 0] = 0.25 / spacing
    mask = distance % 2 == 1
    matrix[mask] = -1 / (np.pi**2 * distance[mask] ** 2 * spacing)
    np.testing.assert_allclose(filtered, data @ matrix, atol=1e-7, rtol=1e-6)


def gaussian_case(scale, width=55):
    from compare_projectors import Case

    from tomojax.geometry import Detector, Grid, ParallelGeometry

    grid = Grid(33, 35, 5, 0.7 * scale, 0.7 * scale, 0.7 * scale)
    detector = Detector(width, 5, 0.7 * scale, 0.7 * scale)
    angles = np.linspace(0, 180, 60, endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid, detector, angles)
    poses = np.array([geometry.pose_for_view(i) for i in range(60)], np.float32)
    x, y, z = [(np.arange(n) - (n - 1) / 2) * 0.7 for n in [33, 35, 5]]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    truth = np.exp(
        -0.5 * ((xx - 0.8) ** 2 / 3.1**2 + (yy + 1.2) ** 2 / 3.1**2 + (zz - 0.5) ** 2 / 0.9**2)
    )
    u = (np.arange(width) - (width - 1) / 2) * 0.7
    center_u = np.cos(np.deg2rad(angles)) * 0.8 + np.sin(np.deg2rad(angles)) * 1.2
    data = (
        np.sqrt(2 * np.pi)
        * 3.1
        * scale
        * np.exp(-0.5 * (u[None, None, :] - center_u[:, None, None]) ** 2 / 3.1**2)
        * np.exp(-0.5 * (z[None, :, None] - 0.5) ** 2 / 0.9**2)
    )
    return Case(
        "parallel-33-60",
        grid,
        detector,
        poses,
        angles,
        truth.astype(np.float32),
        data.astype(np.float32),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("scale", [0.5, 2.0])
@pytest.mark.parametrize("width", [33, 55])
@pytest.mark.parametrize("method", ["astra_fbp3d_cupy", "astra_fbp2d", "tigre_fbp"])
def test_external_fbp_recovers_offset_gaussian_and_length_units(adapters, method, scale, width):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    module = "tigre" if method == "tigre_fbp" else "astra"
    if importlib.util.find_spec(module) is None or (
        method == "astra_fbp3d_cupy" and importlib.util.find_spec("cupy") is None
    ):
        pytest.skip("optional external benchmark library is not installed")
    from compare_reconstructions import solve_astra, solve_tigre

    case = gaussian_case(scale, width)
    result, _ = (
        solve_tigre(case, 1, method) if method == "tigre_fbp" else solve_astra(case, 1, method)
    )
    assert np.linalg.norm(result - case.volume) / np.linalg.norm(case.volume) < 0.013


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["tomojax_fbp_joseph_cgls_pallas", "astra_fbp_cgls"])
def test_fbp_initialization_then_one_update_passes_the_existing_gaussian_gate(adapters, method):
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    if method == "astra_fbp_cgls" and any(
        importlib.util.find_spec(n) is None for n in ["astra", "cupy"]
    ):
        pytest.skip("optional benchmark libraries are not installed")
    from compare_projectors import make_case
    from compare_reconstructions import solve_astra, solve_tomojax

    case = make_case(64, 180, "parallel")
    if method.startswith("tomojax"):
        volume, info = solve_tomojax(case, 1, 180, method)
    else:
        volume, info = solve_astra(case, 1, method)
    assert info["initialization"] == "fbp"
    assert np.linalg.norm(volume - case.volume) / np.linalg.norm(case.volume) < 0.03


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 7, 100])
def test_astra_filter_batches_preserve_all_views_and_attenuation(adapters, batch):
    if jax.default_backend() != "gpu" or any(
        importlib.util.find_spec(n) is None for n in ["astra", "cupy"]
    ):
        pytest.skip("requires CUDA and optional external benchmark libraries")
    case = gaussian_case(1.0, width=33)
    actual, info = adapters.astra_fbp3d(case, filter_batch=batch)
    reference, _ = adapters.astra_fbp3d(case, filter_batch=len(case.poses))
    np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-5)
    assert np.linalg.norm(actual - case.volume) / np.linalg.norm(case.volume) < 0.013
    assert info["filter_views_per_batch"] == min(batch, len(case.poses))
