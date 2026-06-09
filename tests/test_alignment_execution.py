from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align import AlignConfig, align_multires

# check-public-imports: allow-private
from tomojax.align._objectives.recon_layer import PoseAdjustedGeometry

# check-public-imports: allow-private
from tomojax.align._stages import _reconstruction_stage as reconstruction_stage
from tomojax.geometry import Detector, Grid, ParallelGeometry


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
        assert params5.shape == (2, 5)
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
    assert params5.shape == (2, 5)
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
    assert final_state.params5.shape == (2, 5)
    assert final_state.loss == info["loss"]
    assert final_state.outer_stats == info["outer_stats"]
    assert isinstance(final_state.geometry_calibration_state, dict)
    assert "detector" in final_state.geometry_calibration_state


def test_align_multires_executes_cor_then_pose_schedule_with_real_stages() -> None:
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
        schedule="cor_then_pose",
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
    assert params5.shape == (2, 5)
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
    assert outer_stats[1]["fixed_volume_reconstruction_skipped"] is True
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
