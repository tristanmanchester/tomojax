"""Independent exact-basis matrices and physical pose derivatives."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.core.trilinear import exact_adjoint, exact_forward
from tomojax.geometry import Detector, Grid


@pytest.fixture
def problem(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bench"))
    oracle = importlib.import_module("voxel_truth").project_voxel_truth
    grid = Grid(3, 4, 2, 0.7, 1.1, 1.4, vol_origin=(0.12, -1.63, 0.4))
    detector = Detector(5, 4, 0.65, 0.91, (0.23, -0.17))
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, :3] = Rotation.from_euler(
        "XYZ", [[0, 0, 0], [0, 0, 90], [17, -21, 38]], degrees=True
    ).as_matrix()
    poses[:, :3, 3] = [0.13, -0.27, 0.09]
    rng = np.random.default_rng(579)
    volume = rng.uniform(0.2, 1.3, (2, 4, 3)).astype(np.float32)
    matrix = np.column_stack(
        [
            oracle(v.reshape(volume.shape), poses, grid, detector).ravel()
            for v in np.eye(volume.size, dtype=np.float32)
        ]
    )
    return grid, detector, jnp.asarray(poses, jnp.float32), jnp.asarray(volume), matrix, oracle


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_exact_integral_and_adjoint_match_independent_matrix(problem, backend):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    grid, detector, poses, volume, matrix, _ = problem
    project = jax.jit(lambda t, x: exact_forward(t, grid, detector, x, backend=backend))
    actual = project(poses, volume)
    np.testing.assert_allclose(actual.ravel(), matrix @ volume.ravel(), rtol=2e-5, atol=3e-6)
    weights = jnp.asarray(np.random.default_rng(24).normal(size=actual.shape), jnp.float32)
    adjoint = jax.jit(lambda t, y: exact_adjoint(t, grid, detector, y, backend=backend))
    np.testing.assert_allclose(
        adjoint(poses, weights).ravel(), matrix.T @ weights.ravel(), rtol=2e-5, atol=3e-6
    )
    # Reuse compiled operators with new poses and data, including rays missing the grid.
    shifted = poses.at[:, 0, 3].add(20)
    np.testing.assert_array_equal(project(shifted, volume), 0)
    np.testing.assert_array_equal(adjoint(shifted, weights * 2), 0)


@pytest.mark.numerical
def test_exact_projection_stays_fp32_with_x64_and_float64_detector_grid(problem):
    grid, detector, poses, volume, matrix, _ = problem
    with jax.enable_x64():
        det_grid = tuple(a.astype(jnp.float64) for a in get_detector_grid_device(detector))
        image = jax.jit(lambda x: exact_forward(poses, grid, detector, x, det_grid=det_grid))(
            volume
        )
    assert image.dtype == jnp.float32
    np.testing.assert_allclose(image.ravel(), matrix @ volume.ravel(), rtol=2e-5, atol=3e-6)


@pytest.mark.numerical
def test_exact_pose_derivatives_match_independent_rigid_perturbations(problem):
    grid, detector, poses, volume, _, oracle = problem
    pose = np.asarray(poses[-1], np.float64)
    axis = np.array([0.3, -0.2, 0.1])
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    tangent = np.zeros((4, 4), np.float32)
    tangent[:3, :3] = pose[:3, :3] @ skew
    tangent[:3, 3] = [0.17, -0.13, 0.21]
    _, actual = jax.jvp(
        lambda t: exact_forward(t[None], grid, detector, volume),
        (jnp.asarray(pose, jnp.float32),),
        (jnp.asarray(tangent),),
    )
    step = 1e-4
    plus, minus = pose.copy(), pose.copy()
    plus[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(step * axis).as_matrix()
    minus[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(-step * axis).as_matrix()
    plus[:3, 3] += step * tangent[:3, 3]
    minus[:3, 3] -= step * tangent[:3, 3]
    expected = (
        oracle(volume, plus[None], grid, detector) - oracle(volume, minus[None], grid, detector)
    ) / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


def test_exact_integrator_rejects_complex_data_and_invalid_backend(problem):
    grid, detector, poses, volume, _, _ = problem
    with pytest.raises(ValueError, match="real arrays"):
        exact_forward(poses, grid, detector, volume.astype(jnp.complex64))
    with pytest.raises(ValueError, match="backend"):
        exact_forward(poses, grid, detector, volume, backend="unknown")


@pytest.mark.numerical
@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_exact_fista_matches_dense_projected_steps(problem, backend):
    if backend == "cuda" and jax.default_backend() != "gpu":
        pytest.skip("CUDA required")
    from tomojax.recon.fista_tv_core import FistaCoreConfig, fista_tv_core_arrays

    grid, detector, poses, volume, matrix, _ = problem
    target = jnp.asarray((matrix @ volume.ravel()).reshape(3, detector.nv, detector.nu))
    weights = jnp.array([0.7, 1.2, 0.3])
    support = jnp.linspace(0.2, 1, volume.size).reshape(volume.shape)
    weighted_matrix = matrix * np.asarray(support).ravel()
    weighted_matrix *= np.repeat(np.sqrt(weights), detector.nv * detector.nu)[:, None]
    weighted_target = np.asarray(target).ravel() * np.repeat(
        np.sqrt(weights), detector.nv * detector.nu
    )
    bound = float(np.linalg.norm(weighted_matrix, 2) ** 2) * 1.2
    config = FistaCoreConfig(
        iterations=3,
        lipschitz=bound,
        nonnegative=True,
        views_per_batch=2,
        support=support,
        ray_integrator="exact",
        forward_projector=backend,
        backprojector=backend,
    )
    result = jax.jit(
        lambda t, y: fista_tv_core_arrays(
            x0=jnp.zeros_like(volume),
            T_all=t,
            det_grid=None,
            projections=y,
            grid=grid,
            detector=detector,
            cfg=config,
            view_weights=weights,
        ).x
    )(poses, target)
    x, z, momentum = np.zeros(volume.size), np.zeros(volume.size), 1.0
    for _ in range(3):
        updated = np.maximum(
            z - weighted_matrix.T @ (weighted_matrix @ z - weighted_target) / bound, 0
        )
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = np.maximum(updated + (momentum - 1) / next_momentum * (updated - x), 0)
        x, momentum = updated, next_momentum
    np.testing.assert_allclose(result.ravel(), x, rtol=2e-5, atol=3e-6)


@pytest.mark.numerical
@pytest.mark.parametrize("multires", [False, True])
def test_public_exact_alignment_scores_its_actual_operator_and_resumes(problem, multires):
    from dataclasses import replace

    from tomojax.alignment import AlignConfig, align_multires
    from tomojax.alignment.api import L2LossSpec, align, apply_pose_updates
    from tomojax.geometry import ParallelGeometry, stack_view_poses

    grid, detector, _, volume, _, oracle = problem
    geometry = ParallelGeometry(grid, detector, [3.7, 51.1, 93.2])
    poses = stack_view_poses(geometry, 3)
    target = jnp.asarray(oracle(volume, poses, grid, detector), jnp.float32)
    cfg = AlignConfig(
        backend="jax",
        gather_dtype="fp32",
        ray_integrator="exact",
        outer_iterations=2,
        iterations=3,
        lipschitz=100.0,
        pose_translation_frame="detector",
        loss=L2LossSpec(),
        early_stop=False,
        gn_jacobian="central",
    )
    checkpoints = []
    run = align_multires if multires else align
    extra = {"factors": [1]} if multires else {"init_x": 0.8 * volume}
    x, params, info = run(
        geometry,
        grid,
        detector,
        target,
        config=cfg,
        checkpoint_callback=checkpoints.append,
        **extra,
    )
    actual_poses = apply_pose_updates(poses, params, translation_frame="detector")
    residual = oracle(x, actual_poses, grid, detector) - np.asarray(target)
    np.testing.assert_allclose(info["loss"][-1], 0.5 * np.sum(residual**2), rtol=2e-4, atol=2e-6)
    assert info["ray_integrator"] == "exact"
    assert info["objective_kind"] == "joint_volume_pose"
    assert checkpoints and all(c.ray_integrator == "exact" for c in checkpoints)
    with pytest.raises(ValueError, match="ray_integrator differs"):
        run(
            geometry,
            grid,
            detector,
            target,
            config=replace(cfg, ray_integrator="sampled"),
            resume_state=checkpoints[-1],
            **extra,
        )


@pytest.mark.numerical
@pytest.mark.parametrize("axis", [0, 2])
def test_cardinal_right_cell_derivative(axis):
    g = Grid(3, 3, 3, 0.5, 1.0, 2.0)
    d = Detector(3, 3, 0.5, 2.0)
    x = jnp.arange(27, dtype=jnp.float32).reshape(3, 3, 3) / 10
    t = jnp.eye(4, dtype=jnp.float32)[None]
    dt = jnp.zeros_like(t).at[:, axis, 3].set(1)
    # Complete aligned y-rays integrate each y-centred hat to exactly vy.
    integrals = np.pad(np.asarray(x).sum(axis=1) * g.vy, ((0, 1), (0, 1)))
    step = np.diff(integrals, axis=1 - axis // 2)
    expected = -(step[:3, :] / g.vx if axis == 0 else step[:, :3] / g.vz)[None]
    _, jac = jax.jvp(lambda p: exact_forward(p, g, d, x), (t,), (dt,))
    np.testing.assert_allclose(jac, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.numerical
def test_exact_fp32_contract_survives_global_x64(problem):
    grid, detector, poses, volume, *_ = problem
    rng = np.random.default_rng(731)
    targets = jnp.asarray(rng.normal(size=(3, detector.nv, detector.nu)), jnp.float32)

    def evaluate(t, x, y):
        return exact_forward(t, grid, detector, x), exact_adjoint(t, grid, detector, y)

    with jax.enable_x64(False):
        expected = jax.jit(evaluate)(poses, volume, targets)
    with jax.enable_x64():
        arrays = (poses, volume, targets)
        actual = jax.jit(evaluate)(*(a.astype(jnp.float64) for a in arrays))
        for value, reference in zip(actual, expected, strict=True):
            assert value.dtype == jnp.float32
            np.testing.assert_allclose(value, reference, rtol=2e-5, atol=3e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["sampled", "exact", "adjoint"])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_generic_rays_use_cpu_reference_on_cuda_host(method, committed, staged, batched):
    cpu = jax.devices("cpu")[0]
    grid, detector = Grid(5, 4, 3, 0.7, 1.1, 0.9), Detector(7, 5, 1.3, 0.8, (0.2, -0.3))
    shape = (2, 5, 7) if method == "adjoint" else (3, 4, 5)
    original = np.random.default_rng(2306).normal(size=(2, *shape)).astype(np.float32)
    host_poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    host_poses[1, :3, 3] = [0.13, -0.27, 0.21]
    with nullcontext() if committed else jax.default_device(cpu):
        values = jax.device_put(original, cpu) if committed else jnp.asarray(original)
        poses = jax.device_put(host_poses, cpu) if committed else jnp.asarray(host_poses)
        assert values.committed == poses.committed == committed

        def call(x, backend):
            if method == "sampled":
                return forward_project_view_T(
                    poses[1], grid, detector, x, backend=backend, step_size=0.4, n_steps=16
                )
            function = exact_adjoint if method == "adjoint" else exact_forward
            return function(poses, grid, detector, x, backend=backend)

        results = []
        for backend in ("jax", "cuda"):
            function = partial(call, backend=backend)
            if batched:
                function = jax.vmap(function)
            if staged:
                function = jax.jit(function)
            result = function(values if batched else values[0])
            assert result.devices() == {cpu}
            results.append(np.asarray(result))
        reference, actual = results
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype == np.float32
        assert np.isfinite(actual).all()
        assert np.max(abs(actual.astype(float) - reference)) <= 1e-6 * max(1, abs(reference).max())
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(poses, host_poses)
