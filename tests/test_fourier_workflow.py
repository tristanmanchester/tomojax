"""Fourier inversion through the same context-carrying Python and CLI workflow."""

from dataclasses import asdict, replace
import subprocess
import sys

import jax
import numpy as np
import pytest

import tomojax as tj
from tomojax.recon import FourierConfig, fourier_reconstruct


def _scan():
    grid = tj.Grid(24, 24, 2, 1.0, 1.0, 1.0)
    detector = tj.Detector(40, 2, 1.0, 1.0, (-0.75, 0.0))
    angles = np.linspace(0, 180, 120, endpoint=False)
    theta = np.deg2rad(angles)
    u = np.arange(detector.nu) - (detector.nu - 1) / 2 + detector.center[0]
    position = 1.2 * np.cos(theta) + 0.8 * np.sin(theta)
    rows = np.sqrt(2 * np.pi) * 3 * np.exp(-0.5 * ((u[None, :] - position[:, None]) / 3) ** 2)
    data = np.broadcast_to(rows[:, None, :], (120, 2, 40)).astype(np.float32).copy()
    return tj.Scan(data, tj.ParallelGeometry(grid, detector, angles), name="gaussian")


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_fourier_workflow_preserves_geometry_and_recovers_independent_gaussian(backend):
    if backend == "cupy" and jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    scan = _scan()
    grid = replace(scan.grid, nx=20, ny=22, vol_center=(0.4, -0.3, 0.0))
    cfg = FourierConfig(backend=backend, slices_per_batch=1)
    actual = tj.reconstruct(scan, "fourier", grid=grid, config=cfg)
    assert isinstance(actual.volume, np.ndarray)
    assert actual.method == "fourier" and actual.grid == grid
    assert actual.scan.name == scan.name and actual.scan.detector == scan.detector
    assert actual.info == {"config": cfg, "host_slabs": True, "backend": backend}
    expected = fourier_reconstruct(scan.geometry, grid, scan.detector, scan.projections, config=cfg)
    np.testing.assert_array_equal(actual.volume, expected)
    x = (np.arange(grid.nx) - (grid.nx - 1) / 2) + 0.4
    y = (np.arange(grid.ny) - (grid.ny - 1) / 2) - 0.3
    truth = np.exp(-0.5 * ((x[None, :] - 1.2) ** 2 + (y[:, None] + 0.8) ** 2) / 9)
    assert np.linalg.norm(actual.volume[0] - truth) / np.linalg.norm(truth) < 0.02


@pytest.mark.parametrize("options", [{"filter": "ramp"}, {"iterations": 1}, {"warm_start": False}])
def test_fourier_refuses_inapplicable_workflow_options(options):
    with pytest.raises(ValueError, match="method 'fourier' does not take"):
        tj.reconstruct(_scan(), "fourier", **options)


def test_cli_fourier_uses_config_and_round_trips_context(tmp_path):
    from tomojax.cli.main import main

    scan = _scan()
    source, output = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    tj.save(source, scan)
    settings = tmp_path / "fourier.toml"
    settings.write_text('backend = "numpy"\nslices_per_batch = 1\n')
    assert (
        main(
            [
                "recon",
                str(source),
                "-o",
                str(output),
                "--method",
                "fourier",
                "--roi",
                "off",
                "--config",
                str(settings),
            ]
        )
        == 0
    )
    actual = tj.load_reconstruction(output)
    expected = tj.reconstruct(
        scan, "fourier", config=FourierConfig(backend="numpy", slices_per_batch=1)
    )
    np.testing.assert_allclose(actual.volume, expected.volume, atol=1e-6)
    assert actual.method == "fourier" and actual.grid == scan.grid
    assert actual.info["config"] == asdict(expected.info["config"])


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_default_fourier_matches_the_selected_device(backend):
    device = jax.devices("cpu" if backend == "numpy" else "gpu")[0]
    with jax.default_device(device):
        actual = tj.reconstruct(_scan(), "fourier")
        expected = tj.reconstruct(_scan(), "fourier", config=FourierConfig(backend=backend))
    np.testing.assert_array_equal(actual.volume, expected.volume)
    assert actual.info["backend"] == backend


def test_auto_fourier_handles_missing_cupy_without_hiding_errors(monkeypatch):
    import sys
    from types import SimpleNamespace

    scan = _scan()
    expected = tj.reconstruct(scan, "fourier", config=FourierConfig(backend="numpy"))
    device = SimpleNamespace(
        platform="gpu", client=SimpleNamespace(platform_version="CUDA"), local_hardware_id=0
    )
    monkeypatch.setattr(jax, "devices", lambda _: [device])
    monkeypatch.setitem(sys.modules, "cupy", None)
    np.testing.assert_array_equal(tj.reconstruct(scan, "fourier").volume, expected.volume)
    with pytest.raises(ImportError, match="requires"):
        tj.reconstruct(scan, "fourier", config=FourierConfig(backend="cupy"))
    runtime = SimpleNamespace(getDevice=lambda: 1)
    cuda = SimpleNamespace(is_available=lambda: True, runtime=runtime)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=cuda))
    np.testing.assert_array_equal(tj.reconstruct(scan, "fourier").volume, expected.volume)

    def broken_device():
        raise RuntimeError("device discovery failed")

    monkeypatch.setitem(
        sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(is_available=broken_device))
    )
    with pytest.raises(RuntimeError, match="device discovery failed"):
        tj.reconstruct(scan, "fourier")


def test_explicit_host_workflow_does_not_import_jax():
    code = """
import sys, numpy as np
import tomojax as tj
from tomojax.recon import FourierConfig
g, d = tj.Grid(4, 4, 1, 1, 1, 1), tj.Detector(4, 1, 1, 1)
scan = tj.Scan(np.zeros((2, 1, 4), np.float32), tj.ParallelGeometry(g, d, [0., 90.]))
result = tj.reconstruct(scan, 'fourier', config=FourierConfig(backend='numpy'))
assert 'jax' not in sys.modules and np.all(result.volume == 0)
assert result.scan is scan and result.info['backend'] == 'numpy'
"""
    subprocess.run([sys.executable, "-c", code], check=True)
