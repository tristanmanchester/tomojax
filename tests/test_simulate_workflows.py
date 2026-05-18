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
    simulate_to_file,
)
from tomojax.io import load_dataset


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
        gather_dtype: str,
    ) -> jnp.ndarray:
        del gather_dtype
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
    np.testing.assert_allclose(data["thetas_deg"], [0.0, 90.0])


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
        tilt_deg=35.0,
        tilt_about="x",
        phantom="sphere",
        single_rotate=False,
    )

    data = simulate(config)

    assert tuple(data["projections"].shape) == (2, 2, 2)
    assert data["geometry_type"] == "lamino"
    assert data["geometry_meta"] == {"tilt_deg": 35.0, "tilt_about": "x"}


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


def test_simulate_to_file_writes_loadable_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fast_public_simulator(monkeypatch)
    out_path = tmp_path / "synthetic.nxs"
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

    simulate_to_file(config, str(out_path))
    loaded = load_dataset(out_path)

    assert loaded.projections.shape == (2, 2, 2)
    assert loaded.volume is not None
    assert loaded.volume.shape == (2, 2, 2)
    assert loaded.grid is not None
    assert loaded.detector is not None
