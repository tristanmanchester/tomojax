from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.pallas_projector import (
    PallasProjectorUnsupported,
    forward_project_view_T_pallas,
    pallas_projector_actual_variant_metadata,
    pallas_projector_traversal_metadata,
    pallas_projector_variant_metadata,
)
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device


def _pose(theta_deg: float = 0.0, *, grid: Grid, detector: Detector) -> jnp.ndarray:
    geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[theta_deg])
    return jnp.asarray(geom.pose_for_view(0), dtype=jnp.float32)


def _assert_matches_jax(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    volume: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    det_grid=None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    oracle = forward_project_view_T(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
    )
    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
        interpret=True,
    )
    assert candidate.shape == oracle.shape == (detector.nv, detector.nu)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=atol, rtol=rtol)


def test_pallas_forward_project_uniform_volume_returns_path_length() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((16, 16, 16), dtype=jnp.float32)

    projected = forward_project_view_T_pallas(T, grid, detector, volume, interpret=True)

    assert projected.shape == (16, 16)
    np.testing.assert_allclose(np.asarray(projected), 16.0, atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_localized_center_voxel_matches_jax() -> None:
    grid = Grid(nx=5, ny=5, nz=5, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=1, du=0.25, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 2, 2].set(1.0)

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_localized_voxel_with_grid_center_matches_jax() -> None:
    grid = Grid(
        nx=5,
        ny=5,
        nz=5,
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_center=(1.0, 0.0, 0.0),
    )
    detector = Detector(nu=9, nv=1, du=0.25, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.zeros((5, 5, 5), dtype=jnp.float32).at[2, 2, 2].set(1.0)

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_non_cubic_rotated_case_matches_jax() -> None:
    grid = Grid(nx=64, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=3, nv=3, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(90.0, grid=grid, detector=detector)
    volume = jnp.ones((64, 16, 16), dtype=jnp.float32)

    _assert_matches_jax(T, grid, detector, volume, atol=1e-3, rtol=1e-4)


def test_pallas_forward_project_handles_detector_tile_remainder() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=18, nv=17, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(17.0, grid=grid, detector=detector)
    rng = np.random.default_rng(0)
    volume = jnp.asarray(np.abs(rng.normal(size=(16, 16, 16))).astype(np.float32))

    _assert_matches_jax(T, grid, detector, volume)


def test_pallas_forward_project_explicit_traversal_controls_match_jax() -> None:
    grid = Grid(nx=12, ny=10, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=7, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(31.0, grid=grid, detector=detector)
    volume = jnp.arange(12 * 10 * 8, dtype=jnp.float32).reshape((12, 10, 8)) / 1000.0

    _assert_matches_jax(
        T,
        grid,
        detector,
        volume,
        step_size=0.5,
        n_steps=64,
        atol=1e-3,
        rtol=1e-4,
    )


def test_pallas_forward_project_accepts_canonical_detector_grid() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)
    det_grid = get_detector_grid_device(detector)

    _assert_matches_jax(T, grid, detector, volume, det_grid=det_grid)


def test_pallas_forward_project_accepts_explicit_supported_variant_controls() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)

    oracle = forward_project_view_T(T, grid, detector, volume)
    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        num_warps=1,
        kernel_variant="auto",
        layout_variant="detector_vu",
        state_mode="inline",
    )

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_z_integer_variant_matches_jax() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(37.0, grid=grid, detector=detector)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    _assert_matches_jax(T, grid, detector, volume)
    explicit = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        interpret=True,
        kernel_variant="z_integer4",
    )
    oracle = forward_project_view_T(T, grid, detector, volume)
    np.testing.assert_allclose(np.asarray(explicit), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_variant_metadata_normalizes_auto_to_generic() -> None:
    metadata = pallas_projector_variant_metadata(
        tile_shape=(4, 8),
        num_warps=1,
        kernel_variant="auto",
        layout_variant="detector_vu",
        state_mode="inline",
        gather_dtype="float32",
    )

    assert metadata == {
        "tile_shape": [4, 8],
        "num_warps": 1,
        "kernel_variant": "generic",
        "layout_variant": "detector_vu",
        "state_mode": "inline",
        "gather_dtype": "fp32",
    }


def test_pallas_actual_variant_metadata_selects_z_integer_for_auto() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(37.0, grid=grid, detector=detector)

    metadata = pallas_projector_actual_variant_metadata(
        T,
        grid,
        detector,
        kernel_variant="auto",
    )

    assert metadata["kernel_variant"] == "z_integer4"


def test_pallas_traversal_metadata_tightens_default_diagonal_bound() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)

    metadata = pallas_projector_traversal_metadata(T, grid)

    assert metadata["resolved_n_steps"] == 30
    assert metadata["effective_pallas_n_steps"] == 19


def test_pallas_tightened_traversal_matches_jax_for_uniform_volume() -> None:
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((16, 16, 16), dtype=jnp.float32)

    _assert_matches_jax(T, grid, detector, volume)


@pytest.mark.parametrize("gather_dtype", ["bf16", "fp16"])
def test_pallas_forward_project_rejects_unsupported_gather_dtype(gather_dtype: str) -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match="fp32 only"):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            gather_dtype=gather_dtype,
            interpret=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_warps": 3}, "num_warps"),
        ({"kernel_variant": "z_locked8"}, "kernel_variant"),
        ({"layout_variant": "detector_uv"}, "layout_variant"),
        ({"state_mode": "cached"}, "state_mode"),
    ],
)
def test_pallas_forward_project_rejects_unsupported_variant_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match=message):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            interpret=True,
            **kwargs,
        )


def test_pallas_forward_project_rejects_z_integer_when_detector_is_not_aligned() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0, vol_center=(0.0, 0.0, 0.25))
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match="z_integer4"):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            interpret=True,
            kernel_variant="z_integer4",
        )


def test_pallas_forward_project_rejects_noncanonical_detector_grid() -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)
    Xr, Zr = get_detector_grid_device(detector)
    shifted_grid = (Xr + jnp.float32(0.125), Zr)

    with pytest.raises(PallasProjectorUnsupported, match="get_detector_grid_device"):
        forward_project_view_T_pallas(
            T,
            grid,
            detector,
            volume,
            det_grid=shifted_grid,
            interpret=True,
        )
