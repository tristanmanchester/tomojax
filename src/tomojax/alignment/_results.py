from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    import jax.numpy as jnp

    from ._model.schedules import ResolvedAlignmentStage
    from ._observer import ObserverAction, OuterStat


type GaugeFixSummary = dict[str, float | str | list[str]]
type MetadataDict = dict[str, object]


class AlignInfo(TypedDict):
    """Diagnostics from a single-resolution alignment solve.

    Attributes
    ----------
    loss : list[float]
        Objective values recorded in execution order.
    loss_kind : str
        Name of the active projection-domain loss.
    reconstruction : str
        Inner reconstruction method.
    lipschitz : float | None
        Most recent data-gradient Lipschitz estimate, or None.
    outer_stats : list[OuterStat]
        Per-outer-iteration diagnostic dictionaries.
    stopped_by_observer : bool
        Whether an observer stopped the solve.
    observer_action : ObserverAction
        Last normalized observer action.
    wall_time_total : float
        Total elapsed solver time in seconds.
    quality : str
        Alignment quality policy.
    profile_policy : MetadataDict
        Effective reconstruction/projector policy and its provenance.
    pose_model : str
        Per-view, polynomial or spline motion model.
    pose_translation_frame : str
        Object or detector frame for physical translations.
    pose_model_variables : int
        Number of variables in the chosen motion model.
    per_view_variables : int
        Number of variables in an independent per-view model.
    pose_model_basis_shape : list[int]
        Shape of the motion-model basis.
    active_dofs : list[str]
        All active parameter names.
    active_pose_dofs : list[str]
        Active per-view pose parameter names.
    active_geometry_dofs : list[str]
        Active setup-geometry parameter names.
    objective_kind : str
        Main objective model used by the run.
    objective_kinds : list[str]
        Objective models used across the executed stages.
    objective_provenance : MetadataDict | None
        Objective implementation provenance, when available.
    backend_provenance : MetadataDict | None
        Backend implementation provenance, when available.
    optimizer_kind : str
        Pose optimizer used.
    completed_outer_iterations : int
        Number of completed outer iterations.
    small_impr_streak : int
        Successive outer iterations with small improvement.
    motion_coeffs : jax.Array | None
        Motion-model coefficients, or None for independent per-view motion.
    gauge_fix : str
        Gauge-fixing rule applied to the result.
    gauge_fix_dofs : list[str]
        Parameter names affected by gauge fixing.
    gauge_fix_final : GaugeFixSummary
        Final gauge-fixing diagnostic record.
    gauge : NotRequired[MetadataDict]
        Common rigid motion removed from the returned volume/pose pair.
    """

    loss: list[float]
    loss_kind: str
    reconstruction: str
    lipschitz: float | None
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    wall_time_total: float
    quality: str
    profile_policy: MetadataDict
    pose_model: str
    pose_translation_frame: str
    pose_model_variables: int
    per_view_variables: int
    pose_model_basis_shape: list[int]
    active_dofs: list[str]
    active_pose_dofs: list[str]
    active_geometry_dofs: list[str]
    objective_kind: str
    objective_kinds: list[str]
    objective_provenance: MetadataDict | None
    backend_provenance: MetadataDict | None
    optimizer_kind: str
    completed_outer_iterations: int
    small_impr_streak: int
    motion_coeffs: jnp.ndarray | None
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: GaugeFixSummary
    gauge: NotRequired[MetadataDict]


class AlignMultiresInfo(TypedDict):
    """Diagnostics from a multiresolution alignment solve.

    Attributes
    ----------
    loss : list[float]
        Objective values recorded in execution order.
    factors : list[int]
        Resolution factors actually executed.
    factors_skipped : list[int]
        Finer resolution factors skipped for device-memory capacity.
    factors_skipped_reason : str | None
        Reason finer levels were skipped, or None.
    loss_kind : str | None
        Name of the active projection-domain loss.
    reconstruction : str
        Inner reconstruction method.
    outer_stats : list[OuterStat]
        Per-outer-iteration diagnostic dictionaries.
    stopped_by_observer : bool
        Whether an observer stopped the solve.
    observer_action : ObserverAction
        Last normalized observer action.
    total_outer_iterations : int
        Total completed outer iterations across levels.
    wall_time_total : float
        Total elapsed solver time in seconds.
    quality : str
        Alignment quality policy.
    profile_policy : MetadataDict
        Effective reconstruction/projector policy and its provenance.
    pose_model : str
        Per-view, polynomial or spline motion model.
    pose_translation_frame : str
        Object or detector frame for physical translations.
    pose_model_variables : int | None
        Number of variables in the chosen motion model.
    per_view_variables : int | None
        Number of variables in an independent per-view model.
    pose_model_basis_shape : list[int] | None
        Shape of the motion-model basis.
    active_dofs : list[str]
        All active parameter names.
    active_pose_dofs : list[str]
        Active per-view pose parameter names.
    active_geometry_dofs : list[str]
        Active setup-geometry parameter names.
    schedule : MetadataDict
        Resolved schedule metadata.
    schedule_name : str
        Resolved schedule name.
    schedule_stages : list[MetadataDict]
        Resolved stage descriptions in execution order.
    gauge_policy : str
        Requested policy for parameter gauge couplings.
    gauge_decision : MetadataDict
        Validated decision on gauge couplings.
    objective_kind : str | None
        Main objective model used by the run.
    objective_kinds : list[str]
        Objective models used across the executed stages.
    objective_provenance : MetadataDict | None
        Objective implementation provenance, when available.
    backend_provenance : MetadataDict | None
        Backend implementation provenance, when available.
    gauge_fix : str
        Gauge-fixing rule applied to the result.
    gauge_fix_dofs : list[str]
        Parameter names affected by gauge fixing.
    gauge_fix_final : GaugeFixSummary | None
        Final gauge-fixing diagnostic record.
    geometry_dofs : list[str]
        Setup-geometry parameter names reported by the run.
    geometry_calibration_state : MetadataDict | None
        Final persisted setup-geometry state, when present.
    geometry_calibration_diagnostics : MetadataDict | None
        Diagnostics supporting the setup-geometry estimate.
    gauge : MetadataDict | None
        Common rigid motion removed from the returned volume/pose pair.
    """

    loss: list[float]
    factors: list[int]
    # Finer factors left out because they would not fit in device memory.
    factors_skipped: list[int]
    factors_skipped_reason: str | None
    loss_kind: str | None
    reconstruction: str
    outer_stats: list[OuterStat]
    stopped_by_observer: bool
    observer_action: ObserverAction
    total_outer_iterations: int
    wall_time_total: float
    quality: str
    profile_policy: MetadataDict
    pose_model: str
    pose_translation_frame: str
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
    backend_provenance: MetadataDict | None
    gauge_fix: str
    gauge_fix_dofs: list[str]
    gauge_fix_final: GaugeFixSummary | None
    geometry_dofs: list[str]
    geometry_calibration_state: MetadataDict | None
    geometry_calibration_diagnostics: MetadataDict | None
    # The common rigid motion removed from the poses (see alignment._gauge).
    gauge: MetadataDict | None


@dataclass
class AlignResumeState:
    """Saved single-resolution alignment progress.

    Parameters
    ----------
    x : jax.Array
        Current reconstruction in (x, y, z) order on the current level grid.
    pose_params : jax.Array
        Per-view corrections, columns (alpha, beta, phi, dx, dz, dy); angles are radians
        and translations physical lengths.
    motion_coeffs : jax.Array | None, optional
        Coefficients of a polynomial or spline motion model; None for independent per-view
         parameters.
    start_outer_iter : int, optional
        Number of completed outer iterations to resume after. Default is 0.
    loss : list[float], optional
        Recorded objective values in iteration order; defaults to an empty list.
    outer_stats : list[OuterStat], optional
        Per-outer-iteration diagnostics in execution order.
    lipschitz : float | None, optional
        Last inner reconstruction Lipschitz estimate, or None if not yet estimated.
    small_impr_streak : int, optional
        Number of successive outer iterations with small improvement. Default is 0.
    elapsed_offset : float, optional
        Elapsed time in seconds before resuming this state. Default is 0.0.
    pose_translation_frame : str, optional
        Basis for physical translations: "object" or "detector". Default is 'object'.
    ray_integrator : str, optional
        Projection model used by the saved run. Default is 'sampled'.
    """

    x: jnp.ndarray
    pose_params: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    start_outer_iter: int = 0
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    lipschitz: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0
    pose_translation_frame: str = "object"
    ray_integrator: str = "sampled"


@dataclass
class AlignMultiresResumeState:
    """Saved multiresolution alignment progress.

    Parameters
    ----------
    x : jax.Array
        Current reconstruction in (x, y, z) order on the current level grid.
    pose_params : jax.Array
        Per-view corrections, columns (alpha, beta, phi, dx, dz, dy); angles are radians
        and translations physical lengths.
    motion_coeffs : jax.Array | None, optional
        Coefficients of a polynomial or spline motion model; None for independent per-view
         parameters.
    level_index : int, optional
        Zero-based index of the current resolution level. Default is 0.
    level_factor : int, optional
        Current coarsening factor relative to the native grid and detector. Default is
        1.
    completed_outer_iterations_in_level : int, optional
        Number of outer iterations already completed in this level. Default is 0.
    global_outer_iterations_completed : int, optional
        Number of outer iterations already completed across levels. Default is 0.
    prev_factor : int | None, optional
        Previously completed level factor, or None before the first level.
    loss : list[float], optional
        Recorded objective values in iteration order; defaults to an empty list.
    outer_stats : list[OuterStat], optional
        Per-outer-iteration diagnostics in execution order.
    lipschitz : float | None, optional
        Last inner reconstruction Lipschitz estimate, or None if not yet estimated.
    small_impr_streak : int, optional
        Number of successive outer iterations with small improvement. Default is 0.
    elapsed_offset : float, optional
        Elapsed time in seconds before resuming this state. Default is 0.0.
    level_complete : bool, optional
        Whether the current level has completed. Default is False.
    run_complete : bool, optional
        Whether the complete alignment run has finished. Default is False.
    factors_skipped_reason : str | None, optional
        Reason finer levels were skipped, or None when none were skipped.
    geometry_calibration_state : dict[str, object] | None, optional
        Serialized setup-geometry calibration, or None when absent.
    stage_index : int, optional
        Zero-based current schedule-stage index. Default is 0.
    stage_name : str | None, optional
        Name of the current stage, or None before choosing a stage.
    stage_completed : bool, optional
        Whether the current schedule stage has completed. Default is False.
    completed_outer_iterations_in_stage : int, optional
        Outer iterations already completed in this stage. Default is 0.
    pose_translation_frame : str, optional
        Basis for physical translations: "object" or "detector". Default is 'object'.
    ray_integrator : str, optional
        Projection model used by the saved run. Default is 'sampled'.
    """

    x: jnp.ndarray
    pose_params: jnp.ndarray
    motion_coeffs: jnp.ndarray | None = None
    level_index: int = 0
    level_factor: int = 1
    completed_outer_iterations_in_level: int = 0
    global_outer_iterations_completed: int = 0
    prev_factor: int | None = None
    loss: list[float] = field(default_factory=list)
    outer_stats: list[OuterStat] = field(default_factory=list)
    lipschitz: float | None = None
    small_impr_streak: int = 0
    elapsed_offset: float = 0.0
    level_complete: bool = False
    run_complete: bool = False
    factors_skipped_reason: str | None = None
    geometry_calibration_state: dict[str, object] | None = None
    stage_index: int = 0
    stage_name: str | None = None
    stage_completed: bool = False
    completed_outer_iterations_in_stage: int = 0
    pose_translation_frame: str = "object"
    ray_integrator: str = "sampled"


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
    reconstruction: str,
    cfg: object,
    outer_idx: int,
    L_prev: float | None,
) -> float | None:
    if reconstruction == "fista":
        try:
            L_meas = float(info_rec.get("lipschitz", 0.0))
            if math.isfinite(L_meas) and L_meas > 0.0:
                # Solvers report the bound actually used, including any
                # safety margin. Reapplying a factor here on every outer
                # iteration exponentially shrinks all subsequent voxel steps.
                L_prev = L_meas
                stat["lipschitz_measured"] = L_meas
                stat["lipschitz_next"] = L_prev
        except (TypeError, ValueError, OverflowError) as exc:
            _record_stat_conversion_error(stat, "lipschitz", exc)
    losses = info_rec.get("loss")
    if isinstance(losses, Iterable):
        try:
            lhist = list(losses)
            if lhist and bool(info_rec.get("iteration_loss_computed", True)):
                stat["recon_loss_first"] = float(lhist[0])
                stat["recon_loss_last"] = float(lhist[-1])
                stat["recon_loss_min"] = float(min(lhist))
        except (TypeError, ValueError, OverflowError) as exc:
            _record_stat_conversion_error(stat, "loss", exc)
    if reconstruction == "spdhg":
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
        stat["spdhg_seed"] = int(cfg.seed) + int(outer_idx) - 1
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
        enriched.setdefault("schedule_stage_role", stage.stage_role)
        enriched.setdefault("schedule_stage_quality_tier", stage.quality_tier)
        enriched.setdefault("schedule_stage_differentiability", stage.differentiability)
        enriched.setdefault("schedule_stage_speed_claim_eligible", stage.speed_claim_eligible)
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
    "AlignCheckpointCallback",
    "AlignInfo",
    "AlignMultiresCheckpointCallback",
    "AlignMultiresInfo",
    "AlignMultiresResumeState",
    "AlignResumeState",
    "enrich_multires_stage_stat",
    "record_reconstruction_info",
]
