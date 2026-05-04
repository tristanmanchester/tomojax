from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.pallas_projector import (
    PallasProjectorUnsupported,
    backproject_view_T_pallas,
    bind_forward_project_residual_sse_T_pallas,
    bind_forward_project_view_T_pallas,
    bind_forward_project_views_T_pallas,
    forward_project_parallel_z_views_pallas,
    forward_project_loss_and_grad_T_pallas,
    forward_project_residual_sse_T_pallas,
    forward_project_residual_sse_T_pallas_with_state,
    forward_project_view_T_pallas_with_state,
    forward_project_view_T_pallas,
    forward_project_views_T_pallas,
    pallas_projector_actual_sinogram_variant_metadata,
    pallas_projector_actual_variant_metadata,
    pallas_projector_sinogram_traversal_metadata,
    pallas_projector_traversal_metadata,
    pallas_projector_variant_metadata,
    prepare_forward_project_view_T_pallas_state,
    prepare_forward_project_views_T_pallas_state,
)
from tomojax.core.projector import (
    backproject_view_T,
    forward_project_view_T,
    get_detector_grid_device,
    sum_backproject_views_T,
)
from tomojax.bench.forward_projector import (
    ForwardSinogramBenchmarkConfig,
    make_forward_sinogram_fixture,
)


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
    gather_dtype: str = "fp32",
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
        gather_dtype=gather_dtype,
    )
    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        step_size=step_size,
        n_steps=n_steps,
        det_grid=det_grid,
        gather_dtype=gather_dtype,
        interpret=True,
    )
    assert candidate.shape == oracle.shape == (detector.nv, detector.nu)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=atol, rtol=rtol)


def _relative_l2(candidate: jnp.ndarray, reference: jnp.ndarray) -> float:
    reference_np = np.asarray(reference)
    denom = float(np.linalg.norm(reference_np.ravel()))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm((np.asarray(candidate) - reference_np).ravel()) / denom)


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


def test_pallas_forward_project_cached_state_matches_jax() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(31.0, grid=grid, detector=detector)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    state = prepare_forward_project_view_T_pallas_state(
        T,
        grid,
        detector,
        step_size=0.5,
        tile_shape=(4, 8),
        num_warps=1,
        kernel_variant="auto",
    )
    candidate = forward_project_view_T_pallas_with_state(
        state,
        volume,
        interpret=True,
    )
    oracle = forward_project_view_T(T, grid, detector, volume, step_size=0.5)

    assert state.n_steps <= state.resolved_n_steps
    assert candidate.shape == (7, 9)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_bound_forward_project_callable_matches_jax_for_repeated_volumes() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(31.0, grid=grid, detector=detector)
    volume_a = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    volume_b = jnp.flip(volume_a, axis=0)

    projector = bind_forward_project_view_T_pallas(
        T,
        grid,
        detector,
        step_size=0.5,
        tile_shape=(4, 8),
        num_warps=1,
        kernel_variant="auto",
        interpret=True,
    )

    for volume in (volume_a, volume_b):
        candidate = projector(volume)
        oracle = forward_project_view_T(T, grid, detector, volume, step_size=0.5)
        np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_precompute_inclusive_mode_matches_jax() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(37.0, grid=grid, detector=detector)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        interpret=True,
        state_mode="precompute_inclusive",
        kernel_variant="auto",
    )
    oracle = forward_project_view_T(T, grid, detector, volume)

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_views_matches_jax_loop() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    candidate = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    oracle = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume) for T in poses],
        axis=0,
    )

    assert candidate.shape == (3, 7, 9)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_general_pose_stack_uses_directional_traversal_bound() -> None:
    fixture = make_forward_sinogram_fixture(
        ForwardSinogramBenchmarkConfig(
            nx=24,
            ny=24,
            nz=24,
            nu=24,
            nv=24,
            n_views=24,
            pose_mode="general_5d",
        )
    )

    metadata = pallas_projector_sinogram_traversal_metadata(
        fixture.T_stack,
        fixture.grid,
    )

    assert metadata["effective_pallas_n_steps"] == 38
    assert metadata["effective_pallas_n_steps"] < metadata["resolved_n_steps"]


def test_pallas_general_pose_cached_state_uses_directional_traversal_bound() -> None:
    fixture = make_forward_sinogram_fixture(
        ForwardSinogramBenchmarkConfig(
            nx=24,
            ny=24,
            nz=24,
            nu=24,
            nv=24,
            n_views=24,
            pose_mode="general_5d",
        )
    )

    state = prepare_forward_project_views_T_pallas_state(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        tile_shape=(16, 4),
        kernel_variant="generic",
    )

    assert state.n_steps == 38
    assert state.n_steps < state.resolved_n_steps


def test_pallas_general_pose_stack_real_lowering_matches_jax() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Pallas lowering requires GPU")
    fixture = make_forward_sinogram_fixture(
        ForwardSinogramBenchmarkConfig(
            nx=24,
            ny=24,
            nz=24,
            nu=24,
            nv=24,
            n_views=24,
            pose_mode="general_5d",
        )
    )

    candidate = forward_project_views_T_pallas(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        fixture.volume,
        tile_shape=(8, 16),
        kernel_variant="generic",
    )
    oracle = jnp.stack(
        [
            forward_project_view_T(T, fixture.grid, fixture.detector, fixture.volume)
            for T in fixture.T_stack
        ],
        axis=0,
    )

    assert _relative_l2(candidate, oracle) <= 1e-5


def test_pallas_general_pose_stack_nondividing_tile_resolves_to_safe_divisor() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Pallas lowering requires GPU")
    fixture = make_forward_sinogram_fixture(
        ForwardSinogramBenchmarkConfig(
            nx=24,
            ny=24,
            nz=24,
            nu=24,
            nv=24,
            n_views=24,
            pose_mode="general_5d",
        )
    )

    metadata = pallas_projector_actual_sinogram_variant_metadata(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        tile_shape=(16, 4),
        kernel_variant="generic",
    )
    candidate = forward_project_views_T_pallas(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        fixture.volume,
        tile_shape=(16, 4),
        kernel_variant="generic",
    )
    oracle = jnp.stack(
        [
            forward_project_view_T(T, fixture.grid, fixture.detector, fixture.volume)
            for T in fixture.T_stack
        ],
        axis=0,
    )

    assert metadata["tile_shape"] == [8, 4]
    assert _relative_l2(candidate, oracle) <= 1e-5


def test_pallas_parallel_z_views_specialization_matches_jax_loop() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    candidate = forward_project_parallel_z_views_pallas(
        T_stack,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
    )
    oracle = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume) for T in poses],
        axis=0,
    )

    assert candidate.shape == (3, 8, 9)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


def test_pallas_parallel_z_views_specialization_rejects_translated_pose() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T_stack = jnp.stack([_pose(0.0, grid=grid, detector=detector)], axis=0)
    T_stack = T_stack.at[0, 0, 3].set(0.25)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match="z-axis specialization"):
        forward_project_parallel_z_views_pallas(
            T_stack,
            grid,
            detector,
            volume,
            interpret=True,
        )


def test_pallas_forward_project_views_cached_call_uses_runtime_volume_and_pose() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume_a = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    volume_b = jnp.flip(volume_a, axis=0).at[1:4, 2:6, 3:5].add(3.0)

    base = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume_a,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    changed_volume = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume_b,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    shifted_stack = T_stack.at[:, 0, 3].add(0.25)
    changed_pose = forward_project_views_T_pallas(
        shifted_stack,
        grid,
        detector,
        volume_a,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )

    oracle_changed_volume = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume_b) for T in poses],
        axis=0,
    )
    oracle_changed_pose = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume_a) for T in shifted_stack],
        axis=0,
    )

    assert _relative_l2(changed_volume, base) > 1e-3
    assert _relative_l2(changed_pose, base) > 1e-3
    np.testing.assert_allclose(
        np.asarray(changed_volume),
        np.asarray(oracle_changed_volume),
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(changed_pose),
        np.asarray(oracle_changed_pose),
        atol=1e-4,
        rtol=1e-4,
    )


def test_pallas_forward_project_views_cached_call_handles_static_config_changes() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector_a = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    detector_b = Detector(nu=9, nv=7, du=0.75, dv=1.25, det_center=(0.25, -0.5))
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    stack_a = jnp.stack(
        [_pose(theta, grid=grid, detector=detector_a) for theta in (0.0, 45.0)],
        axis=0,
    )
    stack_b = jnp.stack(
        [_pose(theta, grid=grid, detector=detector_b) for theta in (0.0, 45.0, 90.0)],
        axis=0,
    )

    candidate_a = forward_project_views_T_pallas(
        stack_a,
        grid,
        detector_a,
        volume,
        interpret=True,
        tile_shape=(4, 4),
        kernel_variant="auto",
    )
    candidate_b = forward_project_views_T_pallas(
        stack_b,
        grid,
        detector_b,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    oracle_a = jnp.stack(
        [forward_project_view_T(T, grid, detector_a, volume) for T in stack_a],
        axis=0,
    )
    oracle_b = jnp.stack(
        [forward_project_view_T(T, grid, detector_b, volume) for T in stack_b],
        axis=0,
    )

    assert candidate_a.shape == (2, 8, 8)
    assert candidate_b.shape == (3, 7, 9)
    np.testing.assert_allclose(np.asarray(candidate_a), np.asarray(oracle_a), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(candidate_b), np.asarray(oracle_b), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("first_dtype", "second_dtype"),
    [
        ("fp32", "bf16"),
        ("bf16", "fp32"),
        ("fp32", "fp16"),
    ],
)
def test_pallas_forward_project_views_cached_call_handles_gather_dtype_changes(
    first_dtype: str,
    second_dtype: str,
) -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=7, nv=5, du=0.75, dv=0.75, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 23.0)]
    T_stack = jnp.stack(poses, axis=0)
    rng = np.random.default_rng(0)
    volume = jnp.asarray(rng.normal(size=(8, 8, 8)).astype(np.float32))

    _ = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        gather_dtype=first_dtype,
        interpret=True,
        tile_shape=(4, 4),
        kernel_variant="auto",
    )
    candidate = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        gather_dtype=second_dtype,
        interpret=True,
        tile_shape=(4, 4),
        kernel_variant="auto",
    )
    oracle = jnp.stack(
        [
            forward_project_view_T(
                T,
                grid,
                detector,
                volume,
                gather_dtype=second_dtype,
            )
            for T in poses
        ],
        axis=0,
    )

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_forward_project_views_cached_call_uses_runtime_inputs_on_gpu() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume_a = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    volume_b = jnp.flip(volume_a, axis=0).at[1:4, 2:6, 3:5].add(3.0)

    base = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume_a,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    changed_volume = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume_b,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    shifted_pose = forward_project_views_T_pallas(
        T_stack.at[:, 0, 3].add(0.25),
        grid,
        detector,
        volume_a,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )

    base.block_until_ready()
    changed_volume.block_until_ready()
    shifted_pose.block_until_ready()
    assert _relative_l2(changed_volume, base) > 1e-3
    assert _relative_l2(shifted_pose, base) > 1e-3


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_backproject_view_matches_jax_on_general_geometry() -> None:
    grid = Grid(nx=6, ny=5, nz=4, vx=1.2, vy=0.9, vz=1.1)
    detector = Detector(nu=7, nv=5, du=0.8, dv=1.3, det_center=(0.4, -0.2))
    T = _pose(17.0, grid=grid, detector=detector)
    image = jnp.arange(detector.nv * detector.nu, dtype=jnp.float32).reshape(
        detector.nv, detector.nu
    ) / 10.0

    oracle = backproject_view_T(T, grid, detector, image, step_size=0.45)
    candidate = backproject_view_T_pallas(
        T,
        grid,
        detector,
        image,
        step_size=0.45,
        tile_shape=(5, 7),
    )

    assert candidate.shape == oracle.shape == (6, 5, 4)
    assert _relative_l2(candidate, oracle) < 1e-5
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(oracle), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_forward_loss_grad_matches_jax_explicit_adjoint() -> None:
    grid = Grid(nx=6, ny=5, nz=4, vx=1.2, vy=0.9, vz=1.1)
    detector = Detector(nu=7, nv=5, du=0.8, dv=1.3, det_center=(0.4, -0.2))
    poses = jnp.stack([_pose(theta, grid=grid, detector=detector) for theta in (0.0, 17.0, 41.0)])
    volume = jnp.arange(grid.nx * grid.ny * grid.nz, dtype=jnp.float32).reshape(
        grid.nx, grid.ny, grid.nz
    ) / 100.0
    target = jnp.stack(
        [
            forward_project_view_T(T, grid, detector, volume * 1.1, step_size=0.45)
            for T in poses
        ],
        axis=0,
    )
    pred = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume, step_size=0.45) for T in poses],
        axis=0,
    )
    weights = jnp.asarray([1.0, 0.5, 0.25], dtype=jnp.float32)[:, None, None]
    raw_resid = (pred - target).astype(jnp.float32)
    weighted_resid = raw_resid * weights
    oracle_loss = jnp.float32(0.5) * jnp.vdot(weighted_resid, weighted_resid).real
    oracle_grad = sum_backproject_views_T(
        poses,
        grid,
        detector,
        raw_resid * weights * weights,
        step_size=0.45,
    )

    loss, grad = forward_project_loss_and_grad_T_pallas(
        poses,
        grid,
        detector,
        volume,
        target,
        weights=weights,
        step_size=0.45,
        tile_shape=(5, 7),
    )

    np.testing.assert_allclose(np.asarray(loss), np.asarray(oracle_loss), atol=1e-4, rtol=1e-5)
    assert _relative_l2(grad, oracle_grad) < 1e-5
    np.testing.assert_allclose(np.asarray(grad), np.asarray(oracle_grad), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires real Pallas lowering")
def test_pallas_forward_loss_grad_general_pose_matches_jax_explicit_adjoint() -> None:
    fixture = make_forward_sinogram_fixture(
        ForwardSinogramBenchmarkConfig(
            nx=8,
            ny=8,
            nz=8,
            nu=8,
            nv=8,
            n_views=4,
            pose_mode="general_5d",
        )
    )
    volume = fixture.volume
    target = jnp.stack(
        [
            forward_project_view_T(T, fixture.grid, fixture.detector, volume * 1.05)
            for T in fixture.T_stack
        ],
        axis=0,
    )
    pred = jnp.stack(
        [
            forward_project_view_T(T, fixture.grid, fixture.detector, volume)
            for T in fixture.T_stack
        ],
        axis=0,
    )
    weights = jnp.linspace(0.5, 1.25, int(fixture.T_stack.shape[0]), dtype=jnp.float32)[
        :, None, None
    ]
    raw_resid = (pred - target).astype(jnp.float32)
    weighted_resid = raw_resid * weights
    oracle_loss = jnp.float32(0.5) * jnp.vdot(weighted_resid, weighted_resid).real
    oracle_grad = sum_backproject_views_T(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        raw_resid * weights * weights,
        det_grid=fixture.det_grid,
    )

    loss, grad = forward_project_loss_and_grad_T_pallas(
        fixture.T_stack,
        fixture.grid,
        fixture.detector,
        volume,
        target,
        weights=weights,
        det_grid=fixture.det_grid,
        tile_shape=(8, 4),
        kernel_variant="generic",
    )

    np.testing.assert_allclose(np.asarray(loss), np.asarray(oracle_loss), atol=1e-4, rtol=1e-5)
    assert _relative_l2(grad, oracle_grad) < 1e-5
    np.testing.assert_allclose(np.asarray(grad), np.asarray(oracle_grad), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "pose_stack",
    [
        jnp.eye(4, dtype=jnp.float32),
        jnp.zeros((0, 4, 4), dtype=jnp.float32),
        jnp.zeros((2, 3, 4), dtype=jnp.float32),
    ],
)
def test_pallas_forward_project_views_rejects_invalid_pose_stack_shapes(pose_stack) -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)

    with pytest.raises((ValueError, PallasProjectorUnsupported), match="pose|stack|shape|view"):
        forward_project_views_T_pallas(
            pose_stack,
            grid,
            detector,
            volume,
            interpret=True,
        )


def test_parallel_geometry_stack_view_poses_matches_per_view_poses() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[0.0, 31.0, 73.0])

    stacked = stack_view_poses(geom, 3)
    expected = jnp.stack(
        [jnp.asarray(geom.pose_for_view(i), dtype=jnp.float32) for i in range(3)],
        axis=0,
    )

    assert stacked.shape == (3, 4, 4)
    np.testing.assert_allclose(np.asarray(stacked), np.asarray(expected), atol=1e-7, rtol=1e-7)


def test_pallas_forward_project_views_one_view_matches_single_view_pallas() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=8, nv=8, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(37.0, grid=grid, detector=detector)
    volume = jnp.ones((8, 8, 8), dtype=jnp.float32)

    batched = forward_project_views_T_pallas(
        T[jnp.newaxis, :, :],
        grid,
        detector,
        volume,
        interpret=True,
        kernel_variant="auto",
    )
    single = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        interpret=True,
        kernel_variant="auto",
    )

    assert batched.shape == (1, 8, 8)
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(single), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_views_cached_state_matches_inline_generic() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    inline = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )
    state = prepare_forward_project_views_T_pallas_state(
        T_stack,
        grid,
        detector,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )
    bound = bind_forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )
    cached = bound(volume)
    state_cached = forward_project_views_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
        state_mode="cached",
    )

    assert state.ix0.shape == (3 * detector.nv * detector.nu,)
    np.testing.assert_allclose(np.asarray(cached), np.asarray(inline), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(state_cached), np.asarray(inline), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_residual_sse_matches_materialized_jax() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    oracle_projection = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume) for T in poses],
        axis=0,
    )
    target = oracle_projection + jnp.linspace(
        0.0,
        0.01,
        oracle_projection.size,
        dtype=jnp.float32,
    ).reshape(oracle_projection.shape)

    candidate = forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        target,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="auto",
    )
    expected = jnp.sum((oracle_projection - target) ** 2, dtype=jnp.float32)

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_residual_sse_cached_state_matches_inline() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (0.0, 31.0, 73.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0
    oracle_projection = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume) for T in poses],
        axis=0,
    )
    target = oracle_projection + jnp.linspace(
        0.0,
        0.01,
        oracle_projection.size,
        dtype=jnp.float32,
    ).reshape(oracle_projection.shape)

    inline = forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        target,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )
    cached = forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        target,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
        state_mode="cached",
    )
    state = prepare_forward_project_views_T_pallas_state(
        T_stack,
        grid,
        detector,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )
    state_cached = forward_project_residual_sse_T_pallas_with_state(
        state,
        volume,
        target,
        interpret=True,
    )
    bound_cached = bind_forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        target,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
    )(volume)

    np.testing.assert_allclose(np.asarray(cached), np.asarray(inline), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(state_cached), np.asarray(inline), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(bound_cached), np.asarray(inline), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_residual_sse_zero_target_matches_materialized_jax() -> None:
    grid = Grid(nx=6, ny=6, nz=6, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=6, nv=5, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    poses = [_pose(theta, grid=grid, detector=detector) for theta in (15.0, 45.0)]
    T_stack = jnp.stack(poses, axis=0)
    volume = jnp.ones((6, 6, 6), dtype=jnp.float32)
    oracle_projection = jnp.stack(
        [forward_project_view_T(T, grid, detector, volume) for T in poses],
        axis=0,
    )
    target = jnp.zeros_like(oracle_projection)

    candidate = forward_project_residual_sse_T_pallas(
        T_stack,
        grid,
        detector,
        volume,
        target,
        interpret=True,
        tile_shape=(4, 4),
    )
    expected = jnp.sum(oracle_projection**2, dtype=jnp.float32)

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(expected), atol=1e-4, rtol=1e-4)


def test_pallas_forward_project_transposed_layout_handles_tile_remainder() -> None:
    grid = Grid(nx=8, ny=8, nz=8, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=9, nv=7, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(17.0, grid=grid, detector=detector)
    volume = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape((8, 8, 8)) / 100.0

    oracle = forward_project_view_T(T, grid, detector, volume)
    candidate = forward_project_view_T_pallas(
        T,
        grid,
        detector,
        volume,
        interpret=True,
        tile_shape=(4, 8),
        kernel_variant="generic",
        layout_variant="detector_uv",
    )

    assert candidate.shape == (7, 9)
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


@pytest.mark.parametrize(
    ("gather_dtype", "canonical"),
    [
        ("bfloat16", "bf16"),
        ("half", "fp16"),
    ],
)
def test_pallas_variant_metadata_normalizes_lower_precision_gather_dtype(
    gather_dtype: str,
    canonical: str,
) -> None:
    metadata = pallas_projector_variant_metadata(
        tile_shape=(4, 8),
        num_warps=1,
        kernel_variant="generic",
        layout_variant="detector_vu",
        state_mode="inline",
        gather_dtype=gather_dtype,
    )

    assert metadata["gather_dtype"] == canonical


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
def test_pallas_forward_project_lower_precision_gather_matches_jax(gather_dtype: str) -> None:
    grid = Grid(
        nx=8,
        ny=8,
        nz=8,
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_center=(0.0, 0.0, 0.25),
    )
    detector = Detector(nu=7, nv=5, du=0.75, dv=0.75, det_center=(0.0, 0.0))
    T = _pose(23.0, grid=grid, detector=detector)
    rng = np.random.default_rng(0)
    volume = jnp.asarray(rng.normal(size=(8, 8, 8)).astype(np.float32))

    _assert_matches_jax(
        T,
        grid,
        detector,
        volume,
        gather_dtype=gather_dtype,
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("gather_dtype", ["float64", "unknown", ""])
def test_pallas_forward_project_rejects_invalid_gather_dtype(gather_dtype: str) -> None:
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    T = _pose(grid=grid, detector=detector)
    volume = jnp.ones((4, 4, 4), dtype=jnp.float32)

    with pytest.raises(PallasProjectorUnsupported, match="gather_dtype"):
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
        ({"layout_variant": "unknown_layout"}, "layout_variant"),
        ({"state_mode": "unknown_state"}, "state_mode"),
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
