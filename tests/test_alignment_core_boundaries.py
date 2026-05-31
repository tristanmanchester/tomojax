from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tomojax.align import AlignConfig

# check-public-imports: allow-private
from tomojax.align._config import _active_dof_mask_for_cfg, _active_dofs_for_cfg

# check-public-imports: allow-private
from tomojax.align._geometry.parametrizations import compose_R, se3_from_5d

# check-public-imports: allow-private
from tomojax.align._model.diagnostics import (
    GaugePolicyError,
    conditioning_diagnostics,
    validate_active_gauge_policy,
)

# check-public-imports: allow-private
from tomojax.align._model.schedules import resolve_alignment_schedule

# check-public-imports: allow-private
from tomojax.align._model.state import AlignmentState

# check-public-imports: allow-private
from tomojax.align._objectives.loss_specs import (
    L2LossSpec,
    L2OtsuLossSpec,
    loss_spec_name,
    loss_spec_params,
    parse_loss_schedule,
    parse_loss_spec,
    resolve_loss_for_level,
    validate_loss_schedule_levels,
)

# check-public-imports: allow-private
from tomojax.align._observer import _normalize_observer_action, adapt_observer_callback

# check-public-imports: allow-private
from tomojax.align._stages._stage_state import _prepare_multires_level_state

# check-public-imports: allow-private
from tomojax.align._stages._stage_types import StageRuntime
from tomojax.align.api import AlignMultiresResumeState, AlignResumeState


def test_pose_parametrization_composes_rotation_and_translation() -> None:
    rotation = compose_R(
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(jnp.pi / 2.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        np.asarray(rotation),
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )

    transform = se3_from_5d(jnp.asarray([0.0, 0.0, 0.0, 2.0, -3.0], dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(transform[:3, 3]), [2.0, 0.0, -3.0])
    np.testing.assert_allclose(np.asarray(transform[3]), [0.0, 0.0, 0.0, 1.0])


def test_gauge_policy_rejects_or_reports_coupled_dofs() -> None:
    with pytest.raises(GaugePolicyError, match="det_u_px_pose_translation"):
        validate_active_gauge_policy(("det_u_px", "dx"), policy="reject")

    decision = validate_active_gauge_policy(("det_u_px", "dx"), policy="anchor_mean")

    assert decision.status == "allowed_with_gauge_policy"
    assert decision.conflicts == ("det_u_px_pose_translation",)
    assert decision.warnings == ("det_u_px_pose_translation",)
    assert decision.to_dict()["status"] == "allowed_with_gauge_policy"


def test_conditioning_diagnostics_reports_near_null_dofs() -> None:
    diagnostics = conditioning_diagnostics(
        jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        dof_names=("alpha", "dx"),
        rtol=1e-4,
    )

    assert diagnostics["status"] == "weak"
    assert diagnostics["singular_values"] == [1.0, 0.0]
    assert diagnostics["near_null_vectors"] == [
        {"singular_value": 0.0, "terms": [{"dof": "dx", "weight": 1.0}]}
    ]


def test_align_config_normalizes_aliases_and_resolves_stage_dofs() -> None:
    cfg = AlignConfig(
        recon_algo="fista-tv",
        opt_method="l-bfgs-b",
        schedule="pose-dx-dz-after-phi",
        freeze_dofs=("dx",),
        projector_backend="jax",
        gather_dtype="FP32",
        gauge_policy="anchor-mean",
        gauge_fix="none",
        pose_model="per-view",
    )

    assert cfg.recon_algo == "fista"
    assert cfg.opt_method == "lbfgs"
    assert cfg.schedule == "pose_dx_dz_after_phi"
    assert cfg.projector_backend == "jax"
    assert cfg.gather_dtype == "fp32"
    assert cfg.gauge_policy == "anchor_mean"
    assert cfg.pose_model == "per_view"
    assert _active_dofs_for_cfg(cfg) == ("dz",)
    assert _active_dof_mask_for_cfg(cfg) == (False, False, False, False, True)


def test_loss_specs_round_trip_params_and_scheduled_levels() -> None:
    huber = parse_loss_spec("huber", {"delta": 2.5})

    assert loss_spec_name(huber) == "huber"
    assert loss_spec_params(huber) == {"delta": 2.5}

    schedule = parse_loss_schedule("4:phasecorr,2:ssim,1:l2_otsu", default=L2LossSpec())
    validate_loss_schedule_levels(schedule, factors=(1, 2, 4))

    assert loss_spec_name(resolve_loss_for_level(schedule, 4)) == "phasecorr"
    assert loss_spec_name(resolve_loss_for_level(schedule, 2)) == "ssim"
    assert isinstance(resolve_loss_for_level(schedule, 1), L2OtsuLossSpec)
    assert isinstance(resolve_loss_for_level(schedule, 8), L2LossSpec)

    with pytest.raises(ValueError, match="do not match configured levels"):
        validate_loss_schedule_levels(schedule, factors=(1, 2))


def test_stage_runtime_enriches_resume_stats_with_global_stage_context() -> None:
    schedule = resolve_alignment_schedule(schedule="cor_then_pose", outer_iters=3)
    stage = schedule.stages[1]
    runtime = StageRuntime(
        level_index=1,
        level_factor=2,
        global_before_level=5,
        global_elapsed_offset=10.0,
        active_loss_name="ssim",
        schedule_name=schedule.name,
        stats_before_level=[],
        loss_before_level=[],
        level_stats=[],
        level_losses=[],
        prev_factor=4,
        observer_fn=None,
        checkpoint_callback=None,
    )

    enriched = runtime.enrich_stats(
        [{"outer_idx": 2, "loss": 1.25, "cumulative_time": 0.5}],
        stage=stage,
        global_start=7,
    )[0]

    assert enriched["level_index"] == 1
    assert enriched["level_factor"] == 2
    assert enriched["global_outer_idx"] == 8
    assert enriched["global_elapsed_seconds"] == pytest.approx(10.5)
    assert enriched["schedule_name"] == "cor_then_pose"
    assert enriched["schedule_stage_name"] == "pose_polish"
    assert enriched["schedule_stage_active_dofs"] == "alpha,beta,phi,dx,dz"
    assert enriched["gauge_status"] == "ok"


def test_observer_action_adapter_normalizes_bool_and_string_results() -> None:
    assert _normalize_observer_action(None) == "continue"
    assert _normalize_observer_action(" ADVANCE_LEVEL ") == "advance_level"

    stop_observer = adapt_observer_callback(lambda _x, _params, _stat: True)
    continue_observer = adapt_observer_callback(lambda _x, _params, _stat: False)
    explicit_observer = adapt_observer_callback(lambda _x, _params, _stat: "stop_run")
    assert stop_observer is not None
    assert continue_observer is not None
    assert explicit_observer is not None

    x = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    params = jnp.zeros((1, 5), dtype=jnp.float32)
    stat = {"outer_idx": 1}

    assert stop_observer(x, params, stat) == "stop_run"
    assert continue_observer(x, params, stat) is None
    assert explicit_observer(x, params, stat) == "stop_run"

    with pytest.raises(ValueError, match="Unsupported observer action"):
        _normalize_observer_action("pause")


def test_multires_level_resume_plan_splits_preserved_and_active_stage_history() -> None:
    resume = AlignMultiresResumeState(
        x=jnp.zeros((2, 2, 2), dtype=jnp.float32),
        params5=jnp.zeros((3, 5), dtype=jnp.float32),
        level_index=1,
        level_factor=2,
        completed_outer_iters_in_level=3,
        global_outer_iters_completed=8,
        loss=[5.0, 4.0, 3.0, 2.0],
        outer_stats=[
            {"level_index": 0, "schedule_stage_index": 0, "outer_idx": 1},
            {"level_index": 1, "schedule_stage_index": 0, "outer_idx": 1},
            {"level_index": 1, "schedule_stage_index": 1, "outer_idx": 2},
            {"level_index": 1, "schedule_stage_index": 1, "outer_idx": 3},
        ],
        elapsed_offset=12.0,
        stage_index=1,
        stage_name="pose_polish",
        stage_completed=False,
        completed_outer_iters_in_stage=2,
    )

    plan = _prepare_multires_level_state(
        resume_state=resume,
        level_index=1,
        loss_hist=list(resume.loss),
        global_outer_stats=list(resume.outer_stats),
        executed_outer_iters=8,
    )

    assert plan.resuming is True
    assert plan.global_before_level == 5
    assert [stat["level_index"] for stat in plan.stats_before_level] == [0]
    assert [stat["schedule_stage_index"] for stat in plan.preserved_level_stats] == [0]
    assert [stat["outer_idx"] for stat in plan.resume_stage_stats] == [2, 3]
    assert plan.preserved_level_losses == [4.0]
    assert plan.resume_stage_losses == [3.0, 2.0]


def test_stage_runtime_checkpoint_preserves_schedule_resume_fields() -> None:
    schedule = resolve_alignment_schedule(schedule="cor_then_pose", outer_iters=3)
    stage = schedule.stages[1]
    emitted: list[AlignMultiresResumeState] = []
    runtime = StageRuntime(
        level_index=2,
        level_factor=1,
        global_before_level=6,
        global_elapsed_offset=20.0,
        active_loss_name="l2",
        schedule_name=schedule.name,
        stats_before_level=[],
        loss_before_level=[5.0],
        level_stats=[{"outer_idx": 1, "schedule_stage_index": 0}],
        level_losses=[4.5],
        prev_factor=2,
        observer_fn=None,
        checkpoint_callback=emitted.append,
    )
    callback = runtime.checkpoint_for_stage(
        stage=stage,
        global_start=7,
        setup_alignment_state=AlignmentState.zeros(n_views=2),
        active_geometry_dofs=("det_u_px",),
    )
    assert callback is not None

    callback(
        AlignResumeState(
            x=jnp.zeros((2, 2, 2), dtype=jnp.float32),
            params5=jnp.zeros((2, 5), dtype=jnp.float32),
            start_outer_iter=2,
            loss=[4.0, 3.5],
            outer_stats=[{"outer_idx": 1}, {"outer_idx": 2}],
            L=1.25,
            small_impr_streak=1,
            elapsed_offset=0.75,
        )
    )

    checkpoint = emitted[0]
    assert checkpoint.level_index == 2
    assert checkpoint.level_factor == 1
    assert checkpoint.stage_index == stage.index
    assert checkpoint.stage_name == "pose_polish"
    assert checkpoint.stage_completed is False
    assert checkpoint.completed_outer_iters_in_stage == 2
    assert checkpoint.global_outer_iters_completed == 9
    assert checkpoint.loss == [5.0, 4.5, 4.0, 3.5]
    assert checkpoint.geometry_calibration_state is not None
