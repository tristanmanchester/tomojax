from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.datasets import (
    SimConfig,
    SimulationArtefacts,
    make_phantom,
    random_cubes_spheres,
    simulate,
)
from tomojax.io import load_dataset

pytestmark = pytest.mark.surface


def _install_fast_public_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stack_view_poses(_geometry: object, n_views: int) -> jnp.ndarray:
        return jnp.zeros((int(n_views), 4, 4), dtype=jnp.float32)

    def fake_detector_grid_device(_detector: object) -> None:
        return None

    def fake_forward_project_view(
        _geometry: object,
        _grid: object,
        detector: Any,
        volume: jnp.ndarray,
        *,
        view_index: int,
    ) -> jnp.ndarray:
        value = jnp.mean(volume) + jnp.asarray(float(view_index), dtype=jnp.float32)
        return jnp.full((int(detector.nv), int(detector.nu)), value, dtype=jnp.float32)

    monkeypatch.setitem(simulate.__globals__, "stack_view_poses", fake_stack_view_poses)
    monkeypatch.setitem(simulate.__globals__, "get_detector_grid_device", fake_detector_grid_device)
    monkeypatch.setitem(simulate.__globals__, "forward_project_view", fake_forward_project_view)


def test_make_phantom_is_deterministic_for_fixed_seed() -> None:
    config = SimConfig(nx=8, ny=8, nz=8, nu=8, nv=8, n_views=2, phantom="blobs", seed=7)

    first = make_phantom(config)
    second = make_phantom(config)

    assert first.shape == (8, 8, 8)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def test_simulate_parallel_projection_shape_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fast_public_simulator(monkeypatch)
    config = SimConfig(
        nx=2,
        ny=2,
        nz=2,
        nu=2,
        nv=2,
        n_views=2,
        phantom="sphere",
        single_rotate=False,
    )

    data = simulate(config)

    assert tuple(data["projections"].shape) == (2, 2, 2)
    assert tuple(data["volume"].shape) == (2, 2, 2)
    assert data["geometry_type"] == "parallel"
    assert data["geometry_meta"] is None
    np.testing.assert_allclose(data["angles"], [0.0, 90.0])


def test_simulate_lamino_records_tilt_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fast_public_simulator(monkeypatch)
    config = SimConfig(
        nx=2,
        ny=2,
        nz=2,
        nu=2,
        nv=2,
        n_views=2,
        geometry="lamino",
        tilt=35.0,
        tilt_about="x",
        phantom="sphere",
        single_rotate=False,
    )

    data = simulate(config)

    assert tuple(data["projections"].shape) == (2, 2, 2)
    assert data["geometry_type"] == "lamino"
    assert data["geometry_meta"] == {"tilt": 35.0, "tilt_about": "x"}


def test_simulation_artefact_metadata_exists_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fast_public_simulator(monkeypatch)
    config = SimConfig(
        nx=2,
        ny=2,
        nz=2,
        nu=2,
        nv=2,
        n_views=2,
        phantom="sphere",
        single_rotate=False,
        seed=11,
        artefacts=SimulationArtefacts(dead_pixel_fraction=0.25, dead_pixel_value=-1.0),
    )

    first = simulate(config)
    second = simulate(config)

    assert first["simulation_artefacts"] is not None
    assert first["simulation_artefacts"]["dead_pixel_count"] == 1
    np.testing.assert_allclose(np.asarray(first["projections"]), np.asarray(second["projections"]))
    assert bool(jnp.any(first["projections"] == -1.0))
    assert "noise" not in first["meta"]
    assert "noise_level" not in first["meta"]


def test_random_shapes_uses_current_center_biased_placement_only() -> None:
    first = random_cubes_spheres(16, 16, 16, n_cubes=2, n_spheres=2, seed=3)
    second = random_cubes_spheres(16, 16, 16, n_cubes=2, n_spheres=2, seed=3)

    assert first.shape == (16, 16, 16)
    np.testing.assert_allclose(first, second)
    with pytest.raises(TypeError, match="placement"):
        random_cubes_spheres(16, 16, 16, placement="old")  # type: ignore[call-arg]


def test_sim_config_no_longer_accepts_noise_shorthand() -> None:
    with pytest.raises(TypeError, match="noise"):
        SimConfig(  # type: ignore[call-arg]
            nx=2,
            ny=2,
            nz=2,
            nu=2,
            nv=2,
            n_views=2,
            noise="gaussian",
            noise_level=0.1,
        )


def test_workflow_simulate_preserves_placement_unpacks_and_saves_directly(tmp_path: Path) -> None:
    import tomojax as tj

    shifted_grid = tj.Grid(6, 5, 4, 0.5, 0.7, 0.9, vol_center=(0.4, -0.3, 0.2))
    offset_detector = tj.Detector(10, 9, 0.6, 0.8, center=(0.25, -0.15))

    sim = tj.simulate(
        "sphere",
        geometry="lamino",
        tilt=25.0,
        n_views=5,
        grid=shifted_grid,
        detector=offset_detector,
        seed=9,
    )
    scan, phantom = sim

    assert isinstance(sim, tj.Simulation)
    assert scan is sim.scan and phantom is sim.phantom
    assert scan.geometry.grid == shifted_grid
    assert scan.geometry.detector == offset_detector
    assert phantom.shape == shifted_grid.shape
    assert len(scan.angles) == 5
    assert isinstance(scan.geometry, tj.LaminographyGeometry)
    assert scan.geometry.tilt == 25.0
    assert scan.source is not None
    sim_info = scan.source.geometry_metadata["simulation_info"]
    assert sim_info["phantom"] == "sphere" and sim_info["seed"] == 9
    expected = np.asarray(tj.project(scan.geometry, phantom))
    np.testing.assert_allclose(np.asarray(scan.projections), expected, rtol=1e-5, atol=1e-5)

    out_path = tmp_path / "synthetic.nxs"
    tj.save(out_path, sim)
    loaded = load_dataset(out_path)
    assert loaded.projections is not None and loaded.projections.shape == (5, 9, 10)
    assert loaded.volume is not None and loaded.volume.shape == (4, 5, 6)
    assert loaded.grid == shifted_grid
    assert loaded.detector == offset_detector


def test_workflow_simulate_cone_sizes_detector_from_physical_extent() -> None:
    import tomojax as tj

    grid = tj.Grid(6, 5, 4, 0.5, 0.5, 0.5, vol_center=(0.5, 0.0, 0.25))
    scan, phantom = tj.simulate("sphere", geometry="cone", grid=grid, n_views=4, seed=2)
    assert isinstance(scan.geometry, tj.ConeGeometry)
    assert scan.geometry.grid == grid
    assert scan.geometry.detector.du == 0.5 and scan.geometry.detector.dv == 0.5
    expected = np.asarray(tj.project(scan.geometry, phantom))
    np.testing.assert_allclose(np.asarray(scan.projections), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "phantom", ["sphere", "cube", "shepp", "blobs", "random_shapes", "lamino_disk"]
)
def test_phantoms_use_non_cubic_zyx_arrays(phantom):
    config = SimConfig(nx=15, ny=13, nz=11, nu=15, nv=11, n_views=2, phantom=phantom)
    volume = np.asarray(make_phantom(config))
    assert volume.shape == (11, 13, 15)
    assert volume.flags.c_contiguous
    if phantom == "lamino_disk":
        assert not volume[:4].any() and not volume[7:].any()


@pytest.mark.gpu
@pytest.mark.parametrize("geometry", ["parallel", "lamino"])
def test_simulated_data_on_a_gpu_agree_with_float32_on_the_cpu(
    tmp_path: Path, geometry: str
) -> None:
    # The simulator projected with a bf16 gather on GPUs: 8.4e-4 relative RMS against float32
    # on a 64^3 Shepp phantom, a repeatable error four orders of magnitude above float32
    # rounding (1e-7) in data that methods are scored against.
    import os
    import subprocess
    import sys

    config = SimConfig(nx=32, ny=32, nz=32, nu=48, nv=48, n_views=64, geometry=geometry)
    gpu = np.asarray(simulate(config)["projections"], np.float64)
    reference = tmp_path / "cpu.npy"
    code = (
        "import numpy as np, sys; from tomojax.datasets import SimConfig, simulate; "
        f"c = SimConfig(nx=32, ny=32, nz=32, nu=48, nv=48, n_views=64, geometry={geometry!r}); "
        "np.save(sys.argv[1], np.asarray(simulate(c)['projections'], np.float64))"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, "-c", code, str(reference)], check=True, env=env)
    cpu = np.load(reference)
    assert np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu) < 1e-6
