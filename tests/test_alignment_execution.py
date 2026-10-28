from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomojax import project
from tomojax.alignment import AlignConfig, align_multires

# check-public-imports: allow-private
from tomojax.alignment._objectives import validation_residuals as validation

# check-public-imports: allow-private
from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry

# check-public-imports: allow-private
from tomojax.alignment._stages import _reconstruction_stage as reconstruction_stage
from tomojax.alignment.api import L2LossSpec, align
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry

from ._helpers import cor_then_polish_schedule


def test_validation_scorer_reuses_one_program_for_changing_candidates(monkeypatch):
    traces = []

    def chunk(z, index, data):
        traces.append(None)
        assert isinstance(data.volume, jax.core.Tracer)
        assert isinstance(data.targets, jax.core.Tracer)
        return data.volume * (z[0] + index) - data.targets

    data = validation._ResidualData(jnp.array([1.0, 2.0]), jnp.array([0.3, 0.4]), None)
    monkeypatch.setattr(
        validation, "_validation_residuals", lambda _: validation._Residuals(chunk, 3, data)
    )
    score = validation.validation_scorer(None)
    for value in (0.1, -0.7, 2.0):
        residuals = np.array([1.0, 2.0]) * (value + np.arange(3)[:, None]) - [0.3, 0.4]
        expected = 0.5 * np.sum(residuals**2)
        assert float(score(jnp.array([value]))) == pytest.approx(expected, rel=2e-6)
    assert len(traces) == 1


@pytest.mark.numerical
@pytest.mark.parametrize("tilt", [False, True])
def test_alignment_automatic_step_bound_respects_physical_length_units(tilt):
    import jax

    results = []
    for scale in (0.1, 1.0, 10.0):
        grid = Grid(3, 4, 2, 0.8 * scale, 1.1 * scale, 1.3 * scale)
        detector = Detector(5, 3, 0.6 * scale, 1.2 * scale, (0.17 * scale, 0.23 * scale))
        angles = np.array([13.0, 68.0, 142.0])
        geometry = (
            LaminographyGeometry(grid, detector, angles, tilt_deg=30)
            if tilt
            else ParallelGeometry(grid, detector, angles)
        )
        poses = jnp.asarray([geometry.pose_for_view(i) for i in range(3)])
        truth = jnp.linspace(0.2, 1.0, 24).reshape((3, 4, 2))
        data = jax.vmap(lambda t, g=grid, d=detector, x=truth: forward_project_view_T(t, g, d, x))(
            poses
        )
        cfg = AlignConfig(
            outer_iterations=1,
            iterations=4,
            tv_weight=0,
            backend="jax",
            gather_dtype="fp32",
            optimise_dofs=("dx",),
            opt_method="gd",
            lr_trans=0,
            early_stop=False,
        )
        volume, _, info = align(geometry, grid, detector, data, config=cfg)
        results.append((np.asarray(volume), info["lipschitz"] / scale**2))
    for volume, bound in results[1:]:
        np.testing.assert_allclose(volume, results[0][0], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(bound, results[0][1], rtol=3e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("regulariser", ["huber_tv", "tv"])
def test_repeated_alignment_reconstruction_keeps_updating_voxels_and_resumes(regulariser):
    # The rays each sum two identical unknown voxels: A.T A has eigenvalue 2
    # on this constant mode. With L=4 every one-step update halves the error.
    # Repeatedly inflating L instead leaves a finite error even after 40 steps.
    grid = Grid(2, 2, 2, 1, 1, 1)
    detector = Detector(2, 2, 1, 1)
    geometry = ParallelGeometry(grid, detector, np.array([0], np.float32))
    data = jnp.full((1, 2, 2), 2.0)
    cfg = AlignConfig(
        outer_iterations=40,
        iterations=1,
        lipschitz=4,
        tv_weight=0.01,
        huber_delta=0.12,
        regulariser=regulariser,
        backend="jax",
        gather_dtype="fp32",
        optimise_dofs=("dx",),
        opt_method="gd",
        lr_trans=0,
        early_stop=False,
    )
    checkpoints = []
    _, _, interrupted = align(
        geometry,
        grid,
        detector,
        data,
        config=cfg,
        observer=lambda _x, _p, stat: "stop_run" if stat["outer_idx"] == 17 else "continue",
        checkpoint_callback=checkpoints.append,
    )
    assert interrupted["stopped_by_observer"]
    assert checkpoints[-1].lipschitz == 4
    volume, _, info = align(
        geometry, grid, detector, data, config=cfg, resume_state=checkpoints[-1]
    )
    np.testing.assert_allclose(volume, np.ones((2, 2, 2)), atol=1e-6)
    assert info["lipschitz"] == 4
    assert all(stat["lipschitz_next"] == 4 for stat in info["outer_stats"])


def test_align_multires_public_execution_emits_observer_and_resume_metadata() -> None:
    traces = []

    @jax.jit
    def caller_program(value):
        traces.append(None)
        return value + 1

    assert float(caller_program(jnp.zeros(()))) == 1
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)
    config = AlignConfig(
        outer_iterations=1,
        iterations=1,
        backend="jax",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        optimise_dofs=("dx",),
        gauge_policy="anchor_mean",
    )
    observer_stats: list[dict[str, object]] = []
    checkpoint_states = []

    def observer(
        x: jnp.ndarray,
        pose_params: jnp.ndarray,
        stat: dict[str, object],
    ) -> str:
        assert x.shape == (2, 2, 2)
        assert pose_params.shape == (2, 6)
        observer_stats.append(dict(stat))
        return "continue"

    x, pose_params, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=(1,),
        config=config,
        observer=observer,
        checkpoint_callback=checkpoint_states.append,
    )

    assert (x.shape, pose_params.shape) == ((2, 2, 2), (2, 6))
    assert bool(jnp.all(jnp.isfinite(x)))
    assert bool(jnp.all(jnp.isfinite(pose_params)))
    assert info["loss"]
    assert info["outer_stats"]
    assert info["observer_action"] == "continue"
    assert info["stopped_by_observer"] is False
    assert info["total_outer_iterations"] == 1
    assert info["active_pose_dofs"] == ["dx"]
    assert info["active_geometry_dofs"] == []

    assert len(observer_stats) == 1
    assert observer_stats[0]["level_factor"] == 1
    assert observer_stats[0]["global_outer_idx"] == 1
    assert info["outer_stats"][0]["observer_action"] == "continue"
    assert info["outer_stats"][0]["observer_stop"] is False

    assert len(checkpoint_states) >= 2
    first_state = checkpoint_states[0]
    final_state = checkpoint_states[-1]
    assert first_state.level_complete is False
    assert first_state.run_complete is False
    assert final_state.level_complete is True
    assert final_state.run_complete is True
    assert final_state.level_index == 0
    assert final_state.level_factor == 1
    assert final_state.global_outer_iterations_completed == 1
    assert final_state.stage_name == "direct_pose"
    assert final_state.stage_completed is True
    assert (final_state.x.shape, final_state.pose_params.shape) == ((2, 2, 2), (2, 6))
    assert final_state.loss == info["loss"]
    assert final_state.outer_stats == info["outer_stats"]
    assert isinstance(final_state.geometry_calibration_state, dict)
    assert "detector" in final_state.geometry_calibration_state
    assert float(caller_program(jnp.ones(()))) == 2
    assert len(traces) == 1  # Alignment must not invalidate the caller's programs.


def test_align_multires_executes_a_setup_then_pose_schedule_with_real_stages() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)
    config = AlignConfig(
        outer_iterations=1,
        iterations=1,
        lipschitz=12.0,
        backend="jax",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        schedule=cor_then_polish_schedule(),
        gauge_policy="anchor_mean",
    )
    checkpoint_states = []

    x, pose_params, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=(1,),
        config=config,
        checkpoint_callback=checkpoint_states.append,
    )

    assert x.shape == (2, 2, 2)
    assert pose_params.shape == (2, 6)
    assert bool(jnp.all(jnp.isfinite(x)))
    assert bool(jnp.all(jnp.isfinite(pose_params)))

    stages = info["schedule_stages"]
    assert [stage["stage_name"] for stage in stages] == ["cor", "pose_polish"]
    assert stages[0]["active_geometry_dofs"] == ["det_u_px"]
    assert stages[0]["active_pose_dofs"] == []
    assert stages[0]["optimizer_kind"] == "validation_lm"
    assert stages[1]["active_geometry_dofs"] == []
    assert stages[1]["active_pose_dofs"] == ["alpha", "beta", "phi", "dx", "dz"]
    assert stages[1]["optimizer_kind"] == "gn"

    outer_stats = info["outer_stats"]
    assert [stat["schedule_stage_name"] for stat in outer_stats] == ["cor", "pose_polish"]
    assert outer_stats[0]["schedule_stage_active_dofs"] == "det_u_px"
    assert outer_stats[0]["geometry_block"] == "setup_validation_lm"
    assert outer_stats[0]["quality_tier"] == "reference"
    assert outer_stats[0]["train_reconstruction_iterations"] == 2
    assert outer_stats[1]["schedule_stage_active_dofs"] == "alpha,beta,phi,dx,dz"
    assert outer_stats[1]["quality_tier"] == "reference"
    # Pose stages alternate with reconstruction instead of reusing a stale volume.
    assert not outer_stats[1].get("fixed_volume_reconstruction_skipped", False)
    assert outer_stats[1]["recon_actual_backend"] == "jax"
    assert outer_stats[1]["recon_fallback_reason"] is None

    assert info["total_outer_iterations"] == 2
    assert info["active_geometry_dofs"] == ["det_u_px"]
    assert info["active_pose_dofs"] == ["alpha", "beta", "phi", "dx", "dz"]
    assert isinstance(info["geometry_calibration_state"], dict)
    assert "detector" in info["geometry_calibration_state"]

    assert len(checkpoint_states) >= 2
    final_state = checkpoint_states[-1]
    assert final_state.level_complete is True
    assert final_state.run_complete is True
    assert final_state.stage_name == "pose_polish"
    assert final_state.stage_completed is True
    assert final_state.completed_outer_iterations_in_stage == 0
    assert final_state.global_outer_iterations_completed == 2
    assert final_state.outer_stats == outer_stats


def test_pose_adjusted_cuda_fallback_keeps_folded_detector_grid_out_of_jax_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        angles=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    base_det_grid = reconstruction_stage.get_detector_grid_device(detector)
    shifted_det_grid = (base_det_grid[0] + jnp.float32(0.25), base_det_grid[1])
    cfg = AlignConfig(
        outer_iterations=1,
        iterations=1,
        backend="cuda",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        optimise_dofs=("dx",),
        gauge_policy="anchor_mean",
    )
    captured: dict[str, object] = {}

    def resolve_backend(*, det_grid, **_kwargs):
        return ("cuda", None) if det_grid is None else ("jax", "det_grid_unsupported")

    def run_dynamic(x0, _T_all, det_u, det_v, _projections, _L_value, **_kwargs):
        captured["det_u"] = det_u
        captured["det_v"] = det_v
        return (
            x0,
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(
        reconstruction_stage,
        "_resolve_reconstruction_projector_backend",
        resolve_backend,
    )
    monkeypatch.setattr(
        reconstruction_stage,
        "_run_huber_fista_core_dynamic_geometry",
        run_dynamic,
    )

    _x, info = reconstruction_stage._run_huber_fista_core_reconstruction(
        reconstruction_stage._ReconstructionStepInputs(
            recon_geometry=PoseAdjustedGeometry(
                geometry=geometry,
                pose_params=jnp.zeros((2, 5), dtype=jnp.float32),
            ),
            grid=grid,
            detector=detector,
            projections=jnp.ones((2, 2, 2), dtype=jnp.float32),
            det_grid=shifted_det_grid,
            x=jnp.ones((2, 2, 2), dtype=jnp.float32),
            cfg=cfg,
            L_prev=1.0,
            outer_idx=1,
        )
    )

    assert captured["det_u"] is None
    assert captured["det_v"] is None
    assert info["actual_backend"] == "jax"
    assert info["fallback_reason"] == "dynamic_geometry_alignment_uses_jax_core"
    assert info["detector_grid_folded_into_pose"] is True


def test_alignment_cuda_backend_probe_uses_options_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    det_grid = reconstruction_stage.get_detector_grid_device(detector)
    captured: dict[str, object] = {}

    class Options:
        def __init__(self, **kwargs: object) -> None:
            captured["options_kwargs"] = kwargs

    def support_fn(_T_all, _grid, _detector, _volume, *, options) -> None:
        captured["options"] = options

    def resolve(name: str, *, missing_reason: str):
        if name == "cuda_projector_sinogram_unsupported_reason":
            return support_fn, None
        if name == "CudaProjectorOptions":
            return Options, None
        return None, missing_reason

    monkeypatch.setattr(reconstruction_stage.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(reconstruction_stage, "resolve_cuda_callable", resolve)

    backend, reason = reconstruction_stage._resolve_reconstruction_projector_backend(
        requested_backend="cuda",
        T_all=jnp.eye(4, dtype=jnp.float32)[None, ...],
        grid=grid,
        detector=detector,
        volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
        det_grid=det_grid,
        gather_dtype="fp32",
    )

    assert backend == "cuda"
    assert reason is None
    assert isinstance(captured["options"], Options)
    options_kwargs = captured["options_kwargs"]
    assert options_kwargs["gather_dtype"] == "fp32"
    assert options_kwargs["det_grid"] is det_grid
    assert options_kwargs["state_mode"] == "cached"


@pytest.mark.parametrize("shape", [(64, 61, 32), (33, 29, 17)])
def test_finite_volume_is_not_rejected_by_fraction_rounding(shape) -> None:
    volume = jnp.ones(shape, dtype=jnp.float32)
    assert reconstruction_stage._finite_fraction(volume) == 1.0
    damaged = volume.at[0, 0, 0].set(jnp.nan).at[-1, -1, -1].set(jnp.inf)
    assert reconstruction_stage._finite_fraction(damaged) == 1.0 - 2 / np.prod(shape)


@pytest.mark.numerical
def test_alignment_partial_view_batches_run_under_global_x64() -> None:
    grid, detector = Grid(3, 4, 2, 0.8, 1.1, 1.3), Detector(5, 3, 0.6, 1.2)
    geometry = ParallelGeometry(grid, detector, np.array([13, 33, 68, 110, 142], np.float32))
    volume = np.linspace(0.2, 1, 24, dtype=np.float32).reshape(3, 4, 2)
    data = np.asarray(project(geometry, volume))
    config = AlignConfig(
        outer_iterations=1,
        iterations=1,
        tv_weight=0,
        backend="jax",
        optimise_dofs=("dx",),
        opt_method="gd",
        lr_trans=0,
        early_stop=False,
        views_per_batch=2,
        checkpoint_projector=False,
    )
    with jax.enable_x64(False):
        expected, _, _ = align(geometry, grid, detector, data, init_x=volume, config=config)
    with jax.enable_x64():
        actual, poses, _ = align(geometry, grid, detector, data, init_x=volume, config=config)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(poses, 0)


def _shifted_laminography_scan(truth):
    """Noisy projections of a small volume whose views carry detector-plane shifts ``truth``."""
    grid = Grid(7, 6, 5, 0.8, 1.1, 1.3)
    detector = Detector(11, 9, 0.7, 0.9, (0.13, -0.21))
    geometry = LaminographyGeometry(grid, detector, np.linspace(13, 157, len(truth)), tilt_deg=23.0)
    axes = [(np.arange(n) - (n - 1) / 2) * h for n, h in [(7, 0.8), (6, 1.1), (5, 1.3)]]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    volume = jnp.asarray(
        sum(
            np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / width)
            for (cx, cy, cz), width in [((1.1, -0.3, 0.7), 1.2), ((-1.3, 1.5, -1.0), 0.8)]
        ),
        jnp.float32,
    )
    nominal = jnp.asarray([geometry.pose_for_view(i) for i in range(len(truth))])

    def project(shifts):
        poses = nominal.at[:, 0, 3].add(shifts[:, 0]).at[:, 2, 3].add(shifts[:, 1])
        return jax.vmap(lambda pose: forward_project_view_T(pose, grid, detector, volume))(poses)

    noise = np.random.default_rng(227).normal(size=(len(truth), detector.nv, detector.nu))
    data = project(jnp.asarray(truth, jnp.float32)) + 0.03 * noise.astype(np.float32)
    return geometry, grid, detector, volume, data, project


@pytest.mark.parametrize(
    ("optimizer", "settings"), [("gd", {"lr_trans": 0.5}), ("gn", {"w_trans": 10.0})]
)
def test_pose_steps_reduce_the_declared_objective(optimizer, settings):
    # A gradient step much longer than the remaining motion must backtrack, and
    # a Gauss-Newton step must account for its curvature prior.
    coordinate = np.linspace(-1, 1, 9)
    truth = np.column_stack([0.012 * coordinate, 0.008 - 0.006 * coordinate])
    geometry, grid, detector, volume, data, project = _shifted_laminography_scan(truth)
    initial_shifts = truth + np.random.default_rng(3).normal(0, 0.012, truth.shape)
    initial = jnp.zeros((9, 6)).at[:, 3:5].set(initial_shifts)
    config = AlignConfig(
        outer_iterations=8,
        iterations=0,
        opt_method=optimizer,
        backend="jax",
        gather_dtype="fp32",
        views_per_batch=4,
        loss=L2LossSpec(),
        early_stop=False,
        optimise_dofs=("dx", "dz"),
        pose_translation_frame="detector",
        tv_weight=0,
        **settings,
    )
    checkpoints = []
    align(
        geometry,
        grid,
        detector,
        data,
        init_x=volume,
        init_pose_params=initial,
        config=config,
        checkpoint_callback=checkpoints.append,
    )
    found = checkpoints[-1].pose_params[:, 3:5]

    def objective(shifts):
        prior = config.w_trans**2 * jnp.sum(jnp.diff(shifts, n=2, axis=0) ** 2)
        return 0.5 * jnp.sum((project(shifts) - data) ** 2) + prior

    assert float(objective(found)) < float(objective(jnp.asarray(initial_shifts, jnp.float32)))
    if optimizer == "gn":
        assert np.linalg.norm(found - truth) < 0.2 * np.linalg.norm(initial_shifts - truth)
        assert float(jnp.linalg.norm(jax.grad(objective)(found))) < 1e-3


@pytest.mark.numerical
@pytest.mark.parametrize("pose_delta", [0.0, 0.01])
@pytest.mark.parametrize("integrator", ["sampled", "joseph", "joseph_cubic"])
def test_alignment_reconstruction_streams_host_data_without_changing_the_solve(
    monkeypatch, pose_delta, integrator
):
    grid, detector = Grid(3, 4, 2, 0.7, 1.1, 0.9), Detector(5, 3, 0.6, 0.8)
    geometry = ParallelGeometry(grid, detector, np.array([13.0, 56.0, 91.0]))
    geometry = PoseAdjustedGeometry(geometry=geometry, pose_params=jnp.full((3, 5), pose_delta))
    cfg = AlignConfig(
        iterations=3,
        tv_weight=0.02,
        huber_delta=0.2,
        views_per_batch=2,
        ray_integrator=integrator,
        backend="cuda" if jax.default_backend() == "gpu" else "jax",
    )
    step = reconstruction_stage._ReconstructionStepInputs(
        recon_geometry=geometry,
        grid=grid,
        detector=detector,
        projections=np.ones((3, 3, 5), np.float32),
        det_grid=reconstruction_stage.get_detector_grid_device(detector),
        x=jnp.full((3, 4, 2), 0.1),
        cfg=cfg,
        L_prev=100.0,
        outer_idx=1,
    )
    policy = reconstruction_stage.reconstruction_quality_policy("verify")
    monkeypatch.setattr(reconstruction_stage, "reconstruction_quality_policy", lambda _: policy)
    monkeypatch.setattr(reconstruction_stage, "should_stream", lambda *a, **k: False)
    expected, expected_info = reconstruction_stage._run_huber_fista_core_reconstruction(step)
    monkeypatch.setattr(reconstruction_stage, "should_stream", lambda *a, **k: True)
    actual, actual_info = reconstruction_stage._run_huber_fista_core_reconstruction(step)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
    assert actual_info["effective_iterations"] == expected_info["effective_iterations"] == 6
    assert actual_info["actual_backend"] == expected_info["actual_backend"]
    np.testing.assert_allclose(actual_info["loss"], expected_info["loss"], rtol=3e-6)
