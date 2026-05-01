from __future__ import annotations

import math

import jax.numpy as jnp

from tomojax.align.early_stop import (
    EarlyStopState,
    evaluate_early_stop,
    normalize_early_stop_profile,
    pose_evidence_from_stat,
    resolve_early_stop_policy,
    setup_evidence_from_stat,
)
from tomojax.align.pipeline import AlignConfig, align
from tomojax.core.geometry import Detector, Grid, ParallelGeometry


def test_compute_saving_setup_continues_for_meaningful_accepted_step():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    evidence = setup_evidence_from_stat(
        {
            "geometry_loss_before": 100.0,
            "geometry_loss_after": 99.0,
            "geometry_accepted": True,
            "geometry_step_norm": 0.02,
            "optimizer_selected_scale": 1.0,
            "optimizer_condition_number": 10.0,
        },
        active_dofs=("det_u_px",),
    )

    decision = evaluate_early_stop(
        evidence=evidence,
        policy=policy,
        state=EarlyStopState(),
        outer_idx=1,
    )

    assert decision.should_stop is False
    assert decision.reason == "warmup"
    assert decision.state.gain_streak == 0
    assert decision.state.step_streak == 0
    assert decision.telemetry()["accepted_rel_impr"] == 0.01


def test_compute_saving_stops_tiny_accepted_setup_gain_and_step_despite_loss_drift():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    state = EarlyStopState()
    prev_after = 100.0
    for outer_idx, loss_before, loss_after in (
        (1, 100.0, 99.999),
        (2, 99.0, 98.9995),
    ):
        evidence = setup_evidence_from_stat(
            {
                "geometry_loss_before": loss_before,
                "geometry_loss_after": loss_after,
                "geometry_accepted": True,
                "geometry_step_norm": 1e-4,
                "optimizer_selected_scale": 1.0,
                "optimizer_condition_number": 10.0,
            },
            active_dofs=("detector_roll_deg",),
            prev_loss_after=prev_after,
        )
        decision = evaluate_early_stop(
            evidence=evidence,
            policy=policy,
            state=state,
            outer_idx=outer_idx,
        )
        state = decision.state
        prev_after = float(loss_after)

    assert decision.should_stop is True
    assert decision.reason == "small_gain_and_step"
    assert decision.state.gain_streak == 2
    assert decision.state.step_streak == 2
    assert decision.evidence.loss_drift_from_prev_after is not None


def test_robust_profile_waits_longer_than_compute_saving_on_same_setup_evidence():
    compute_policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    robust_policy = resolve_early_stop_policy(
        enabled=True,
        profile="robust",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    evidence = setup_evidence_from_stat(
        {
            "geometry_loss_before": 100.0,
            "geometry_loss_after": 99.999,
            "geometry_accepted": True,
            "geometry_step_norm": 1e-4,
            "optimizer_selected_scale": 1.0,
            "optimizer_condition_number": 10.0,
        },
        active_dofs=("detector_roll_deg",),
    )

    compute_state = EarlyStopState()
    robust_state = EarlyStopState()
    for outer_idx in range(1, 3):
        compute_decision = evaluate_early_stop(
            evidence=evidence,
            policy=compute_policy,
            state=compute_state,
            outer_idx=outer_idx,
        )
        robust_decision = evaluate_early_stop(
            evidence=evidence,
            policy=robust_policy,
            state=robust_state,
            outer_idx=outer_idx,
        )
        compute_state = compute_decision.state
        robust_state = robust_decision.state

    assert compute_decision.should_stop is True
    assert robust_decision.should_stop is False
    assert robust_policy.patience == 4


def test_rejected_setup_updates_stop_with_specific_reason():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    state = EarlyStopState()
    evidence = setup_evidence_from_stat(
        {
            "geometry_loss_before": 100.0,
            "geometry_loss_after": 100.0,
            "geometry_accepted": False,
            "geometry_step_norm": 0.0,
            "optimizer_selected_scale": 0.0,
            "optimizer_condition_number": 10.0,
        },
        active_dofs=("detector_roll_deg",),
    )

    for outer_idx in range(1, 3):
        decision = evaluate_early_stop(
            evidence=evidence,
            policy=policy,
            state=state,
            outer_idx=outer_idx,
        )
        state = decision.state

    assert decision.should_stop is True
    assert decision.reason == "rejected_updates"
    assert decision.state.rejected_streak == 2


def test_missing_optional_evidence_does_not_crash_and_records_unknown_step():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    evidence = setup_evidence_from_stat(
        {
            "geometry_loss_before": 100.0,
            "geometry_loss_after": 99.999,
            "geometry_accepted": True,
        },
        active_dofs=("det_u_px",),
    )

    decision = evaluate_early_stop(
        evidence=evidence,
        policy=policy,
        state={},
        outer_idx=2,
    )

    assert decision.should_stop is False
    assert decision.step_is_small is None
    assert decision.telemetry()["early_stop_step_is_small"] is None


def test_disabled_profile_never_stops():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="off",
        rel_impr_threshold=1e-3,
        patience=1,
    )
    evidence = setup_evidence_from_stat(
        {
            "geometry_loss_before": 100.0,
            "geometry_loss_after": math.inf,
            "geometry_accepted": False,
            "geometry_step_norm": 0.0,
        },
        active_dofs=("det_u_px",),
    )

    decision = evaluate_early_stop(
        evidence=evidence,
        policy=policy,
        state=EarlyStopState(nonfinite_streak=10),
        outer_idx=10,
    )

    assert decision.should_stop is False
    assert decision.reason == "disabled"
    assert decision.state == EarlyStopState()


def test_pose_translation_only_uses_translation_movement_evidence():
    policy = resolve_early_stop_policy(
        enabled=True,
        profile="compute_saving",
        rel_impr_threshold=1e-3,
        patience=2,
    )
    state = EarlyStopState()
    evidence = pose_evidence_from_stat(
        {
            "loss_before": 50.0,
            "loss_after": 49.999,
            "rel_impr": 2e-5,
            "trans_mean": 1e-6,
        },
        active_dofs=("dx", "dz"),
    )

    for outer_idx in range(1, 3):
        decision = evaluate_early_stop(
            evidence=evidence,
            policy=policy,
            state=state,
            outer_idx=outer_idx,
        )
        state = decision.state

    assert decision.should_stop is True
    assert decision.reason == "small_gain_and_step"
    assert decision.step_is_small is True


def test_profile_aliases_normalize():
    assert normalize_early_stop_profile("compute-saving") == "compute_saving"
    assert normalize_early_stop_profile("conservative") == "robust"
    assert normalize_early_stop_profile("disabled") == "off"


def test_pose_alignment_emits_early_stop_telemetry_on_outer_stats():
    grid = Grid(nx=4, ny=4, nz=4, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=4, nv=4, du=1.0, dv=1.0)
    geometry = ParallelGeometry(
        grid=grid,
        detector=detector,
        thetas_deg=jnp.linspace(0.0, 180.0, 4, endpoint=False),
    )
    projections = jnp.zeros((4, 4, 4), dtype=jnp.float32)

    _x, _params, info = align(
        geometry,
        grid,
        detector,
        projections,
        cfg=AlignConfig(
            outer_iters=2,
            recon_iters=1,
            tv_prox_iters=1,
            optimise_dofs=("dx", "dz"),
            early_stop=True,
            early_stop_profile="compute_saving",
            views_per_batch=1,
            checkpoint_projector=False,
            gather_dtype="fp32",
            recon_positivity=False,
        ),
    )

    assert info["outer_stats"]
    assert info["outer_stats"][-1]["early_stop_profile"] == "compute_saving"
    assert info["outer_stats"][-1]["early_stop_decision"] in {"continue", "stop"}
    assert "early_stop_state" in info
