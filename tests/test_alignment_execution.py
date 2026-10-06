from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.alignment import AlignConfig, align_multires

# check-public-imports: allow-private
from tomojax.alignment._objectives.recon_layer import PoseAdjustedGeometry

# check-public-imports: allow-private
from tomojax.alignment._stages import _reconstruction_stage as reconstruction_stage
from tomojax.alignment.api import align

# check-public-imports: allow-private
from tomojax.core.projector import forward_project_view_T
from tomojax.geometry import Detector, Grid, LaminographyGeometry, ParallelGeometry

from ._helpers import cor_then_polish_schedule


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
            outer_iters=1,
            recon_iters=4,
            lambda_tv=0,
            projector_backend="jax",
            gather_dtype="fp32",
            optimise_dofs=("dx",),
            opt_method="gd",
            lr_trans=0,
            gauge_fix="none",
            early_stop=False,
        )
        volume, _, info = align(geometry, grid, detector, data, config=cfg)
        results.append((np.asarray(volume), info["L"] / scale**2))
    for volume, bound in results[1:]:
        np.testing.assert_allclose(volume, results[0][0], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(bound, results[0][1], rtol=3e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("public_fallback", [False, True])
def test_repeated_alignment_reconstruction_keeps_updating_voxels_and_resumes(
    monkeypatch, public_fallback
):
    # The rays each sum two identical unknown voxels: A.T A has eigenvalue 2
    # on this constant mode. With L=4 every one-step update halves the error.
    # Repeatedly inflating L instead leaves a finite error even after 40 steps.
    grid = Grid(2, 2, 2, 1, 1, 1)
    detector = Detector(2, 2, 1, 1)
    geometry = ParallelGeometry(grid, detector, np.array([0], np.float32))
    data = jnp.full((1, 2, 2), 2.0)
    cfg = AlignConfig(
        outer_iters=40,
        recon_iters=1,
        recon_L=4,
        lambda_tv=0.01,
        huber_delta=0.12,
        projector_backend="jax",
        gather_dtype="fp32",
        optimise_dofs=("dx",),
        opt_method="gd",
        lr_trans=0,
        gauge_fix="none",
        early_stop=False,
    )
    if public_fallback:
        monkeypatch.setattr(
            reconstruction_stage,
            "_run_huber_fista_core_reconstruction",
            lambda step: reconstruction_stage._run_public_fista_core_bypass(
                step, fallback_reason="test_public_fista_fallback"
            ),
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
    assert checkpoints[-1].L == 4
    volume, _, info = align(
        geometry, grid, detector, data, config=cfg, resume_state=checkpoints[-1]
    )
    np.testing.assert_allclose(volume, np.ones((2, 2, 2)), atol=1e-6)
    assert info["L"] == 4
    assert all(stat["L_next"] == 4 for stat in info["outer_stats"])


def test_align_multires_public_execution_emits_observer_and_resume_metadata() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)
    config = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        projector_backend="jax",
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
        params5: jnp.ndarray,
        stat: dict[str, object],
    ) -> str:
        assert x.shape == (2, 2, 2)
        assert params5.shape == (2, 6)
        observer_stats.append(dict(stat))
        return "continue"

    x, params5, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=(1,),
        config=config,
        observer=observer,
        checkpoint_callback=checkpoint_states.append,
    )

    assert x.shape == (2, 2, 2)
    assert params5.shape == (2, 6)
    assert bool(jnp.all(jnp.isfinite(x)))
    assert bool(jnp.all(jnp.isfinite(params5)))
    assert info["loss"]
    assert info["outer_stats"]
    assert info["observer_action"] == "continue"
    assert info["stopped_by_observer"] is False
    assert info["total_outer_iters"] == 1
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
    assert final_state.global_outer_iters_completed == 1
    assert final_state.stage_name == "direct_pose"
    assert final_state.stage_completed is True
    assert final_state.x.shape == (2, 2, 2)
    assert final_state.params5.shape == (2, 6)
    assert final_state.loss == info["loss"]
    assert final_state.outer_stats == info["outer_stats"]
    assert isinstance(final_state.geometry_calibration_state, dict)
    assert "detector" in final_state.geometry_calibration_state


def test_align_multires_executes_a_setup_then_pose_schedule_with_real_stages() -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    projections = jnp.ones((2, 2, 2), dtype=jnp.float32)
    config = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        recon_L=12.0,
        projector_backend="jax",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        schedule=cor_then_polish_schedule(),
        gauge_policy="anchor_mean",
    )
    checkpoint_states = []

    x, params5, info = align_multires(
        geometry,
        grid,
        detector,
        projections,
        factors=(1,),
        config=config,
        checkpoint_callback=checkpoint_states.append,
    )

    assert x.shape == (2, 2, 2)
    assert params5.shape == (2, 6)
    assert bool(jnp.all(jnp.isfinite(x)))
    assert bool(jnp.all(jnp.isfinite(params5)))

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
    assert outer_stats[0]["train_reconstruction_iters"] == 2
    assert outer_stats[1]["schedule_stage_active_dofs"] == "alpha,beta,phi,dx,dz"
    assert outer_stats[1]["quality_tier"] == "reference"
    # Pose stages alternate with reconstruction instead of reusing a stale volume.
    assert not outer_stats[1].get("fixed_volume_reconstruction_skipped", False)
    assert outer_stats[1]["recon_actual_backend"] == "jax"
    assert outer_stats[1]["recon_fallback_reason"] is None

    assert info["total_outer_iters"] == 2
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
    assert final_state.completed_outer_iters_in_stage == 0
    assert final_state.global_outer_iters_completed == 2
    assert final_state.outer_stats == outer_stats


def test_pose_adjusted_pallas_fallback_keeps_folded_detector_grid_out_of_jax_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    base_det_grid = reconstruction_stage.get_detector_grid_device(detector)
    shifted_det_grid = (base_det_grid[0] + jnp.float32(0.25), base_det_grid[1])
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        projector_backend="pallas",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        optimise_dofs=("dx",),
        gauge_policy="anchor_mean",
    )
    captured: dict[str, object] = {}

    def resolve_backend(*, det_grid, **_kwargs):
        return ("pallas", None) if det_grid is None else ("jax", "det_grid_unsupported")

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
                params5=jnp.zeros((2, 5), dtype=jnp.float32),
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


def test_alignment_pallas_backend_probe_uses_options_api(
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
        if name == "pallas_projector_sinogram_unsupported_reason":
            return support_fn, None
        if name == "PallasProjectorOptions":
            return Options, None
        return None, missing_reason

    monkeypatch.setattr(reconstruction_stage.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(reconstruction_stage, "resolve_pallas_callable", resolve)

    backend, reason = reconstruction_stage._resolve_reconstruction_projector_backend(
        requested_backend="pallas",
        T_all=jnp.eye(4, dtype=jnp.float32)[None, ...],
        grid=grid,
        detector=detector,
        volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
        det_grid=det_grid,
        gather_dtype="fp32",
        fallback_policy="fallback",
    )

    assert backend == "pallas"
    assert reason is None
    assert isinstance(captured["options"], Options)
    options_kwargs = captured["options_kwargs"]
    assert options_kwargs["gather_dtype"] == "fp32"
    assert options_kwargs["det_grid"] is det_grid
    assert options_kwargs["state_mode"] == "cached"


def test_fixed_geometry_reconstruction_reports_effective_auto_pallas_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=np.asarray([0.0, 90.0], dtype=np.float32),
    )
    cfg = AlignConfig(
        outer_iters=1,
        recon_iters=1,
        projector_backend="jax",
        views_per_batch=1,
        checkpoint_projector=False,
        early_stop=False,
        optimise_dofs=("dx",),
        gauge_policy="anchor_mean",
    )

    def run_fixed(x0, *_args, **_kwargs):
        return (
            x0,
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    monkeypatch.setattr(
        reconstruction_stage,
        "_run_huber_fista_core_fixed_geometry",
        run_fixed,
    )
    monkeypatch.setattr(
        reconstruction_stage,
        "effective_fista_core_backend",
        lambda *_args, **_kwargs: "pallas",
    )

    _x, info = reconstruction_stage._run_huber_fista_core_reconstruction(
        reconstruction_stage._ReconstructionStepInputs(
            recon_geometry=geometry,
            grid=grid,
            detector=detector,
            projections=jnp.ones((2, 2, 2), dtype=jnp.float32),
            det_grid=reconstruction_stage.get_detector_grid_device(detector),
            x=jnp.ones((2, 2, 2), dtype=jnp.float32),
            cfg=cfg,
            L_prev=1.0,
            outer_idx=1,
        )
    )

    assert info["requested_backend"] == "jax"
    assert info["actual_backend"] == "pallas"


@pytest.mark.parametrize("shape", [(64, 61, 32), (33, 29, 17)])
def test_finite_volume_is_not_rejected_by_fraction_rounding(shape) -> None:
    volume = jnp.ones(shape, dtype=jnp.float32)
    assert reconstruction_stage._finite_fraction(volume) == 1.0
    damaged = volume.at[0, 0, 0].set(jnp.nan).at[-1, -1, -1].set(jnp.inf)
    assert reconstruction_stage._finite_fraction(damaged) == 1.0 - 2 / np.prod(shape)
