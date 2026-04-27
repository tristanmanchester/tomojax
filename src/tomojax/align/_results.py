from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, TypedDict

import jax.numpy as jnp

from ._observer import ObserverAction, OuterStat
from .model.schedules import ResolvedAlignmentStage


type GaugeFixSummary = dict[str, float | str | list[str]]
type MetadataDict = dict[str, object]


class AlignInfo(TypedDict):
    loss: list[float]
    loss_kind: str
    recon_algo: str
    L: float | None
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    wall_time_total: float
    pose_model: str
    pose_model_variables: int
    per_view_variables: int
    pose_model_basis_shape: list[int]
    active_dofs: list[str]
    active_pose_dofs: list[str]
    active_geometry_dofs: list[str]
    objective_kind: str
    objective_kinds: list[str]
    objective_provenance: MetadataDict | None
    optimizer_kind: str
    completed_outer_iters: int
    small_impr_streak: int
    motion_coeffs: jnp.ndarray | None
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: GaugeFixSummary


class AlignMultiresInfo(TypedDict):
    loss: list[float]
    factors: list[int]
    loss_kind: str | None
    recon_algo: str
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    total_outer_iters: int
    wall_time_total: float
    pose_model: str
    pose_model_variables: int | None
    per_view_variables: int | None
    pose_model_basis_shape: list[int] | None
    active_dofs: list[str]
    active_pose_dofs: list[str]
    active_geometry_dofs: list[str]
    schedule: MetadataDict
    schedule_name: str
    schedule_stages: list[MetadataDict]
    gauge_policy: str
    gauge_decision: MetadataDict
    objective_kind: str | None
    objective_kinds: list[str]
    objective_provenance: MetadataDict | None
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: GaugeFixSummary | None
    geometry_dofs: list[str]
    geometry_calibration_state: MetadataDict | None
    geometry_calibration_diagnostics: MetadataDict | None


@dataclass
class AlignResumeState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    start_outer_iter: int = 0
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    L: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0


@dataclass
class AlignMultiresResumeState:
    x: jnp.ndarray
    params5: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    level_index: int = 0
    level_factor: int = 1
    completed_outer_iters_in_level: int = 0
    global_outer_iters_completed: int = 0
    prev_factor: int | None = None
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    L: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0
    level_complete: bool = False
    run_complete: bool = False
    geometry_calibration_state: dict[str, object] | None = None
    stage_index: int = 0
    stage_name: str | None = None
    stage_completed: bool = False
    completed_outer_iters_in_stage: int = 0


AlignCheckpointCallback = Callable[[AlignResumeState], None]
AlignMultiresCheckpointCallback = Callable[[AlignMultiresResumeState], None]


def _record_stat_conversion_error(
    stat: OuterStat,
    key: str,
    exc: Exception,
    *,
    errors_key: str = "stat_errors",
) -> None:
    errors = stat.setdefault(errors_key, [])
    if isinstance(errors, list):
        errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})


def _set_float_stat(stat: OuterStat, key: str, value: object) -> None:
    try:
        stat[key] = float(value)  # type: ignore[typeddict-unknown-key]
    except (TypeError, ValueError, OverflowError) as exc:
        _record_stat_conversion_error(stat, key, exc)


def record_reconstruction_info(
    stat: OuterStat,
    *,
    info_rec: Mapping[str, object],
    recon_algo: str,
    cfg: object,
    outer_idx: int,
    L_prev: float | None,
) -> float | None:
    if recon_algo == "fista":
        try:
            L_meas = float(info_rec.get("L", 0.0))
            if L_meas > 0.0:
                L_prev = 1.2 * L_meas
                stat["L_meas"] = L_meas
                stat["L_next"] = L_prev
        except (TypeError, ValueError, OverflowError) as exc:
            _record_stat_conversion_error(stat, "L", exc)
    losses = info_rec.get("loss")
    if isinstance(losses, Iterable):
        try:
            lhist = list(losses)
            if lhist:
                stat["recon_loss_first"] = float(lhist[0])
                stat["recon_loss_last"] = float(lhist[-1])
                stat["recon_loss_min"] = float(min(lhist))
                if recon_algo == "fista":
                    stat["fista_first"] = float(lhist[0])
                    stat["fista_last"] = float(lhist[-1])
                    stat["fista_min"] = float(min(lhist))
        except (TypeError, ValueError, OverflowError) as exc:
            _record_stat_conversion_error(stat, "loss", exc)
    if recon_algo == "spdhg":
        for src, dst in (
            ("tau", "spdhg_tau"),
            ("sigma_data", "spdhg_sigma_data"),
            ("sigma_tv", "spdhg_sigma_tv"),
            ("views_per_batch", "spdhg_views_per_batch"),
            ("num_blocks", "spdhg_num_blocks"),
            ("A_norm", "spdhg_A_norm"),
        ):
            value = info_rec.get(src)
            if value is not None:
                stat[dst] = (
                    int(value)
                    if dst in {"spdhg_views_per_batch", "spdhg_num_blocks"}
                    else float(value)
                )
        stat["spdhg_seed"] = int(getattr(cfg, "spdhg_seed")) + int(outer_idx) - 1
    return L_prev


def enrich_multires_stage_stat(
    stat: Mapping[str, object],
    *,
    level_factor: int,
    level_index: int,
    global_outer_idx: int,
    elapsed_offset: float,
    loss_name: str,
    schedule_name: str,
    stage: ResolvedAlignmentStage | None,
) -> OuterStat:
    enriched = dict(stat)
    enriched["level_factor"] = int(level_factor)
    enriched["level_index"] = int(level_index)
    enriched["global_outer_idx"] = int(global_outer_idx)
    enriched["loss_kind"] = str(enriched.get("loss_kind") or loss_name)
    if stage is not None:
        enriched.setdefault("schedule_name", schedule_name)
        enriched.setdefault("schedule_stage_index", int(stage.index))
        enriched.setdefault("schedule_stage_name", stage.name)
        enriched.setdefault("schedule_stage_active_dofs", ",".join(stage.active_dofs))
        enriched.setdefault("gauge_policy", stage.gauge_policy)
        enriched.setdefault("gauge_status", stage.gauge_decision.status)
        enriched.setdefault("gauge_decision", stage.gauge_decision.to_dict())
    level_elapsed = stat.get("cumulative_time")
    try:
        level_elapsed_f = float(level_elapsed) if level_elapsed is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        _record_stat_conversion_error(
            enriched,
            "cumulative_time",
            exc,
            errors_key="level_stats_errors",
        )
        level_elapsed_f = None
    enriched["level_elapsed_seconds"] = level_elapsed_f
    enriched["global_elapsed_seconds"] = (
        float(elapsed_offset + level_elapsed_f) if level_elapsed_f is not None else None
    )
    return enriched


__all__ = [
    "AlignInfo",
    "AlignCheckpointCallback",
    "AlignMultiresInfo",
    "AlignMultiresCheckpointCallback",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "enrich_multires_stage_stat",
    "record_reconstruction_info",
]
