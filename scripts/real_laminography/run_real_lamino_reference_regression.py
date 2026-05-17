#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import numpy as np

from tomojax.align.api import (
    AlignConfig,
    AlignmentSchedule,
    AlignmentStage,
    AlignmentState,
    GeometryCalibrationState,
    PoseState,
    SetupGeometryState,
    align,
    geometry_with_axis_state,
    level_detector_grid,
    loss_spec_name,
    resolve_loss_for_level,
    se3_from_5d,
)
from tomojax.bench import (
    RealLaminoGpuMonitor,
    append_real_lamino_csv,
    apply_real_lamino_projection_background,
    optimize_reference_setup_geometry_bilevel_for_level,
    parse_real_lamino_z_range,
    real_lamino_commit_info,
    real_lamino_global_z_to_local_index,
    real_lamino_global_z_to_phys,
    real_lamino_json_safe,
    real_lamino_pose_params_summary,
    real_lamino_projection_stats,
    real_lamino_xy_at_global_z,
    resize_nearest_2d,
    save_real_lamino_z_stack,
    save_uint8_png,
    update_real_lamino_status,
    validate_real_lamino_loaded_input,
    write_real_lamino_json,
    write_real_lamino_params_csv,
    write_real_lamino_stage_products,
)
from tomojax.core.geometry import Detector, Grid, LaminographyGeometry
from tomojax.io import load_real_laminography_input
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import FistaConfig, fista_tv
from tomojax.recon.multires import bin_projections, scale_detector, scale_grid, upsample_volume

_json_safe = real_lamino_json_safe
_write_json = write_real_lamino_json
_append_csv = append_real_lamino_csv
_status = update_real_lamino_status
GpuMonitor = RealLaminoGpuMonitor
_commit_info = real_lamino_commit_info
_validate_loaded_input = validate_real_lamino_loaded_input
_projection_stats = real_lamino_projection_stats
_apply_projection_background = apply_real_lamino_projection_background
_save_png = save_uint8_png
_resize_nearest_2d = resize_nearest_2d
_save_z_stack = save_real_lamino_z_stack


def _parse_bounds(text: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not text:
        return out
    for part in str(text).split(","):
        if not part.strip():
            continue
        name, raw = part.split("=", 1)
        lo_raw, hi_raw = raw.split(":", 1)
        lo = float(lo_raw)
        hi = float(hi_raw)
        if hi < lo:
            raise ValueError(f"invalid bounds for {name}: {raw}")
        out[str(name).strip()] = (lo, hi)
    return out


def _apply_setup_bounds(
    state: GeometryCalibrationState,
    bounds: str,
) -> tuple[GeometryCalibrationState, dict[str, Any]]:
    parsed = _parse_bounds(bounds)
    if not parsed:
        return state, {}
    updates: dict[str, float] = {}
    clipped: dict[str, dict[str, float | bool]] = {}
    for name, (lo, hi) in parsed.items():
        before = float(getattr(state, name))
        after = float(np.clip(before, lo, hi))
        updates[name] = after
        clipped[name] = {"before": before, "after": after, "lo": lo, "hi": hi, "clipped": before != after}
    return replace(state, **updates), clipped


def _alignment_state_from_geometry_state(
    state: GeometryCalibrationState,
    *,
    params5: np.ndarray,
    volume: jnp.ndarray | None,
) -> AlignmentState:
    return AlignmentState(
        setup=SetupGeometryState.from_degrees(
            det_u_px=state.det_u_px,
            det_v_px=state.det_v_px,
            detector_roll_deg=state.detector_roll_deg,
            axis_rot_x_deg=state.axis_rot_x_deg,
            axis_rot_y_deg=state.axis_rot_y_deg,
            nominal_axis_unit=state.nominal_axis_unit,
        ),
        pose=PoseState(jnp.asarray(params5, dtype=jnp.float32)),
        volume=volume,
    )


def _geometry_state_from_alignment_state(
    state: AlignmentState,
    *,
    active_geometry_dofs: tuple[str, ...],
) -> GeometryCalibrationState:
    degrees = state.setup.degrees_dict()
    return GeometryCalibrationState(
        det_u_px=float(state.setup.det_u_px),
        det_v_px=float(state.setup.det_v_px),
        detector_roll_deg=float(degrees["detector_roll_deg"]),
        axis_rot_x_deg=float(degrees["axis_rot_x_deg"]),
        axis_rot_y_deg=float(degrees["axis_rot_y_deg"]),
        nominal_axis_unit=tuple(float(v) for v in np.asarray(state.setup.nominal_axis_unit)),
        active_geometry_dofs=active_geometry_dofs,
    )


def _parse_shape3(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text).lower().replace("x", ",").split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("expected three positive integers as n_views,nv,nu")
    try:
        shape = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three positive integers as n_views,nv,nu") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("expected three positive integers as n_views,nv,nu")
    return shape


class RunContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_root = Path(args.out)
        self.status_path = self.run_root / "status.json"
        self.preview_global_z = int(args.preview_z)
        self.stack_z_range = parse_real_lamino_z_range(args.stack_z_range)
        self.stage_records: list[dict[str, Any]] = []
        self.naive_slice: np.ndarray | None = None
        self.final_volume: np.ndarray | None = None
        self.final_grid: Grid | None = None

    def stage_dir(self, name: str) -> Path:
        return self.run_root / name

    def save_stage_products(
        self,
        *,
        stage_dir: Path,
        volume: np.ndarray,
        grid: Grid,
        full_nz: int,
        input_reference: np.ndarray | None,
        suffix: str = "aligned",
    ) -> dict[str, str]:
        return write_real_lamino_stage_products(
            stage_dir=stage_dir,
            volume=volume,
            grid=grid,
            full_nz=full_nz,
            preview_global_z=self.preview_global_z,
            stack_z_range=self.stack_z_range,
            snapshot_max_cols=int(self.args.snapshot_max_cols),
            input_reference=input_reference,
            fallback_reference=self.naive_slice,
            suffix=suffix,
        )


def _schedule_dict(schedule: AlignmentSchedule) -> dict[str, Any]:
    return {
        "name": schedule.name,
        "stages": [
            {
                "name": stage.name,
                "active_dofs": list(stage.active_dofs),
                "objective_kind": stage.objective_kind,
                "optimizer_kind": stage.optimizer,
                "gauge_policy": stage.gauge_policy,
                "maxiter": int(stage.maxiter),
                "early_stop": bool(stage.early_stop),
                "stage_role": stage.stage_role,
                "differentiability": stage.differentiability,
                "quality_tier": stage.quality_tier,
                "speed_claim_eligible": bool(stage.speed_claim_eligible),
            }
            for stage in schedule.stages
        ],
        "metadata": dict(schedule.metadata),
    }


def _write_params_csv(path: Path, params: np.ndarray) -> None:
    write_real_lamino_params_csv(path, params)


def _make_cfg(
    args: argparse.Namespace,
    *,
    active_pose: tuple[str, ...] = (),
    active_setup: tuple[str, ...] = (),
    bounds: str = "",
    outer_iters: int | None = None,
) -> AlignConfig:
    cfg_bounds = "" if active_setup else bounds
    return AlignConfig(
        align_profile=str(args.align_profile),
        outer_iters=int(args.outer_iters if outer_iters is None else outer_iters),
        recon_iters=int(args.recon_iters),
        lambda_tv=float(args.lambda_tv),
        regulariser=str(args.regulariser),
        tv_prox_iters=int(args.tv_prox_iters),
        recon_algo="fista",
        views_per_batch=int(args.views_per_batch),
        checkpoint_projector=True,
        gather_dtype=str(args.gather_dtype),
        projector_backend=str(args.projector_backend),
        quality_tier=str(args.quality_tier),
        fallback_policy=str(args.fallback_policy),
        fold_rigid_detector_grid=bool(getattr(args, "fold_rigid_detector_grid", True)),
        opt_method="gn",
        gn_damping=float(args.gn_damping),
        optimise_dofs=active_pose or None,
        geometry_dofs=active_setup,
        bounds=cfg_bounds,
        pose_model=str(getattr(args, "pose_model", "spline")),
        knot_spacing=int(getattr(args, "knot_spacing", 8)),
        degree=int(getattr(args, "pose_degree", 3)),
        gauge_fix="mean_translation" if {"dx", "dz"} & set(active_pose) else "none",
        mask_vol="cyl",
        recon_positivity=bool(args.recon_positivity),
        seed_translations=False,
        early_stop=bool(args.early_stop),
        early_stop_rel_impr=float(args.early_stop_rel),
        early_stop_patience=int(args.early_stop_patience),
        log_summary=True,
        log_compact=True,
    )


def run_baseline(
    ctx: RunContext,
    *,
    geometry: LaminographyGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    full_nz: int,
) -> np.ndarray:
    stage_dir = ctx.stage_dir("00_baseline")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _status(ctx.status_path, state="running", stage="00_baseline", message="baseline_fbp")
    t0 = time.perf_counter()
    vol = np.asarray(
        fbp(
            geometry,
            grid,
            detector,
            jnp.asarray(projections, dtype=jnp.float32),
            filter_name=str(ctx.args.filter_name),
            views_per_batch=int(ctx.args.views_per_batch),
            checkpoint_projector=True,
            gather_dtype=str(ctx.args.gather_dtype),
        ),
        dtype=np.float32,
    )
    jax.block_until_ready(vol)
    elapsed = time.perf_counter() - t0
    np.save(stage_dir / "naive_fbp.npy", vol)
    ctx.naive_slice = real_lamino_xy_at_global_z(vol, grid=grid, full_nz=full_nz, global_z=ctx.preview_global_z)
    _save_png(stage_dir / f"naive_or_input_xy_global_z{ctx.preview_global_z:03d}.png", ctx.naive_slice)
    ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=vol,
        grid=grid,
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
        suffix="aligned",
    )
    _save_png(stage_dir / "measured_sinogram.png", projections[:, projections.shape[1] // 2, :])
    manifest = {
        "stage": "00_baseline",
        "status": "completed",
        "elapsed_seconds": elapsed,
        "volume_shape": list(vol.shape),
        "preview_z": int(ctx.preview_global_z),
        "z_stack_range": list(ctx.stack_z_range),
    }
    _write_json(stage_dir / "stage_manifest.json", manifest)
    _write_json(stage_dir / "align_info.json", {"stage": "baseline", "outer_stats": []})
    _append_csv(
        stage_dir / "stage_summary.csv",
        {"stage": "00_baseline", "status": "completed", "elapsed_seconds": elapsed},
        ["stage", "status", "elapsed_seconds"],
    )
    return vol


def _save_checkpoint(path: Path, *, x: np.ndarray, params: np.ndarray, setup: GeometryCalibrationState, stat: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.asarray(x, dtype=np.float32),
        params5=np.asarray(params, dtype=np.float32),
        geometry_calibration_state=json.dumps(_json_safe(setup.to_calibration_state().to_dict()), sort_keys=True),
        stat=json.dumps(_json_safe(stat), sort_keys=True),
    )
    latest = path.parent / "latest.npz"
    try:
        latest.write_bytes(path.read_bytes())
    except Exception:
        pass
    return str(path)


def run_setup_stage(
    ctx: RunContext,
    *,
    stage_dir: Path,
    stage_name: str,
    active_setup: tuple[str, ...],
    geometry: LaminographyGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    full_nz: int,
    setup_state: GeometryCalibrationState,
    params5: np.ndarray,
    levels: tuple[int, ...],
    bounds: str,
    level_outer_counts: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, GeometryCalibrationState, list[dict[str, Any]]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = _make_cfg(ctx.args, active_setup=active_setup, bounds=bounds, outer_iters=1)
    stage_stats: list[dict[str, Any]] = []
    x_init = None
    prev_factor: int | None = None
    setup_state = replace(setup_state, active_geometry_dofs=active_setup)
    for level_index, factor in enumerate(levels):
        g = scale_grid(grid, factor)
        d = scale_detector(detector, factor)
        y = bin_projections(jnp.asarray(projections, dtype=jnp.float32), factor)
        if x_init is not None and prev_factor is not None and prev_factor > factor:
            x_level = upsample_volume(jnp.asarray(x_init), prev_factor // factor, (g.nx, g.ny, g.nz))
        else:
            x_level = None
        last_loss = math.inf
        stale = 0
        parity_outer_count = (
            None if level_outer_counts is None else level_outer_counts.get(int(factor))
        )
        outer_limit = int(parity_outer_count or ctx.args.outer_iters)
        for outer in range(1, outer_limit + 1):
            _status(
                ctx.status_path,
                state="running",
                stage=stage_name,
                level_factor=int(factor),
                outer_iter=int(outer),
                active_dofs=list(active_setup),
            )
            loss_spec = resolve_loss_for_level(cfg_base.loss, int(factor))
            loss_name = loss_spec_name(loss_spec)
            t0 = time.perf_counter()
            setup_result = optimize_reference_setup_geometry_bilevel_for_level(
                geometry=geometry,
                grid=g,
                detector=d,
                projections=y,
                init_x=x_level,
                init_params5=jnp.asarray(params5, dtype=jnp.float32),
                state=_alignment_state_from_geometry_state(
                    setup_state,
                    params5=params5,
                    volume=x_level,
                ),
                active_geometry_dofs=active_setup,
                factor=int(factor),
                cfg=cfg_base,
                loss_spec=loss_spec,
                loss_name=loss_name,
            )
            x_level = setup_result.x
            setup_state = _geometry_state_from_alignment_state(
                setup_result.state,
                active_geometry_dofs=active_setup,
            )
            setup_state, clipped = _apply_setup_bounds(setup_state, bounds)
            elapsed = time.perf_counter() - t0
            x_np = np.asarray(x_level, dtype=np.float32)
            stat = dict(
                setup_result.checkpoint_outer_stats[-1]
                if setup_result.checkpoint_outer_stats
                else {}
            )
            stat.update(
                {
                    "stage": stage_name,
                    "level_factor": int(factor),
                    "level_index": int(level_index),
                    "outer_iter": int(outer),
                    "elapsed_seconds": float(elapsed),
                    "active_dofs": ",".join(active_setup),
                    "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
                    "setup_bounds": bounds,
                    "setup_bounds_clipped": clipped,
                }
            )
            stage_stats.append(stat)
            stem = f"outer_{len(stage_stats):03d}_level{factor:02d}_iter{outer:02d}"
            preview_dir = stage_dir / "timeline_z"
            preview_grid = g
            xy = real_lamino_xy_at_global_z(x_np, grid=preview_grid, full_nz=full_nz, global_z=ctx.preview_global_z)
            ref = _resize_nearest_2d(ctx.naive_slice, xy.shape) if ctx.naive_slice is not None else None
            _save_png(preview_dir / "slices" / f"{stem}_global_z{ctx.preview_global_z:03d}.png", xy)
            if ref is not None:
                _save_png(preview_dir / "delta_vs_naive" / f"{stem}_delta.png", xy - ref)
            _save_z_stack(
                preview_dir / "z_stacks" / f"{stem}_global_z{ctx.stack_z_range[0]:03d}_{ctx.stack_z_range[1]:03d}.png",
                x_np,
                grid=preview_grid,
                full_nz=full_nz,
                z_range=ctx.stack_z_range,
                max_cols=int(ctx.args.snapshot_max_cols),
            )
            _save_checkpoint(
                stage_dir / "checkpoints" / f"{stem}.npz",
                x=x_np,
                params=params5,
                setup=setup_state,
                stat=stat,
            )
            fields = [
                "stage",
                "level_factor",
                "outer_iter",
                "geometry_loss_before",
                "geometry_loss_after",
                "geometry_accepted",
                "geometry_status",
                "optimizer_selected_scale",
                "optimizer_condition_number",
                "elapsed_seconds",
            ]
            _append_csv(stage_dir / "stage_summary.csv", stat, fields)
            loss_after = float(stat.get("geometry_loss_after", last_loss))
            rel = (last_loss - loss_after) / max(abs(last_loss), 1e-6) if math.isfinite(last_loss) else math.inf
            stale = stale + 1 if rel < float(ctx.args.early_stop_rel) else 0
            last_loss = loss_after
            if (
                parity_outer_count is None
                and bool(ctx.args.early_stop)
                and stale >= int(ctx.args.early_stop_patience)
            ):
                break
        x_init = np.asarray(x_level, dtype=np.float32)
        prev_factor = int(factor)
    setup_state = replace(setup_state, active_geometry_dofs=active_setup)
    _write_json(stage_dir / "align_info.json", {"outer_stats": stage_stats, "active_geometry_dofs": list(active_setup)})
    _write_json(stage_dir / "geometry_calibration_state.json", setup_state.to_calibration_state().to_dict())
    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=np.asarray(x_init, dtype=np.float32),
        grid=scale_grid(grid, int(prev_factor or 1)),
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
    )
    _write_json(
        stage_dir / "stage_manifest.json",
        {
            "stage": stage_name,
            "status": "completed",
            "active_dofs": list(active_setup),
            "levels": list(levels),
            "bounds": bounds,
            "stats_count": len(stage_stats),
            "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
            "artifacts": products,
        },
    )
    return np.asarray(x_init, dtype=np.float32), setup_state, stage_stats


def run_pose_stage(
    ctx: RunContext,
    *,
    stage_dir: Path,
    stage_name: str,
    active_pose: tuple[str, ...],
    geometry: LaminographyGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    full_nz: int,
    setup_state: GeometryCalibrationState,
    params5: np.ndarray,
    levels: tuple[int, ...],
    bounds: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = _make_cfg(ctx.args, active_pose=active_pose, bounds=bounds)
    stage_stats: list[dict[str, Any]] = []
    x_init = None
    prev_factor: int | None = None
    params_current = jnp.asarray(params5, dtype=jnp.float32)

    def make_observer(level_grid: Grid, level_factor: int) -> Callable[[Any, Any, dict[str, Any]], str]:
        def observer(x_obs: Any, params_obs: Any, stat_obs: dict[str, Any]) -> str:
            stat = dict(stat_obs)
            stat["stage"] = stage_name
            stat["active_dofs"] = ",".join(active_pose)
            stat["level_factor"] = int(level_factor)
            stat["global_stage_outer_idx"] = len(stage_stats) + 1
            x_np = np.asarray(x_obs, dtype=np.float32)
            params_np = np.asarray(params_obs, dtype=np.float32)
            checkpoint_failures: list[str] = []
            x_finite_fraction = float(np.isfinite(x_np).mean()) if x_np.size else 0.0
            params_finite_fraction = float(np.isfinite(params_np).mean()) if params_np.size else 0.0
            stat["checkpoint_x_finite_fraction"] = x_finite_fraction
            stat["checkpoint_params_finite_fraction"] = params_finite_fraction
            if x_finite_fraction < 1.0:
                checkpoint_failures.append(f"x finite fraction is {x_finite_fraction:.6g}")
            if params_finite_fraction < 1.0:
                checkpoint_failures.append(
                    f"params finite fraction is {params_finite_fraction:.6g}"
                )
            for key in ("loss_before", "loss_after"):
                value = stat.get(key)
                if value is not None:
                    try:
                        if not math.isfinite(float(value)):
                            checkpoint_failures.append(f"{key} is non-finite")
                    except (TypeError, ValueError):
                        checkpoint_failures.append(f"{key} is not numeric")
            if checkpoint_failures:
                stat["checkpoint_validation_failed"] = True
                stat["checkpoint_validation_failures"] = checkpoint_failures
            xy = real_lamino_xy_at_global_z(x_np, grid=level_grid, full_nz=full_nz, global_z=ctx.preview_global_z)
            ref = _resize_nearest_2d(ctx.naive_slice, xy.shape) if ctx.naive_slice is not None else None
            stem = (
                f"outer_{len(stage_stats)+1:03d}_level{level_factor:02d}_"
                f"iter{int(stat.get('outer_idx', len(stage_stats)+1)):02d}"
            )
            preview_dir = stage_dir / "timeline_z"
            _save_png(preview_dir / "slices" / f"{stem}_global_z{ctx.preview_global_z:03d}.png", xy)
            if ref is not None:
                _save_png(preview_dir / "delta_vs_naive" / f"{stem}_delta.png", xy - ref)
            _save_z_stack(
                preview_dir / "z_stacks" / f"{stem}_global_z{ctx.stack_z_range[0]:03d}_{ctx.stack_z_range[1]:03d}.png",
                x_np,
                grid=level_grid,
                full_nz=full_nz,
                z_range=ctx.stack_z_range,
                max_cols=int(ctx.args.snapshot_max_cols),
            )
            _save_checkpoint(
                stage_dir / "checkpoints" / f"{stem}.npz",
                x=x_np,
                params=params_np,
                setup=setup_state,
                stat=stat,
            )
            stage_stats.append(stat)
            _append_csv(
                stage_dir / "stage_summary.csv",
                stat,
                [
                    "stage",
                    "level_factor",
                    "outer_idx",
                    "loss_before",
                    "loss_after",
                    "rms_update",
                    "accepted",
                    "active_dofs",
                    "cumulative_time",
                ],
            )
            if checkpoint_failures:
                return "stop_run"
            return "continue"

        return observer

    for level_index, factor in enumerate(levels):
        g = scale_grid(grid, factor)
        d = scale_detector(detector, factor)
        y = bin_projections(jnp.asarray(projections, dtype=jnp.float32), factor)
        if x_init is not None and prev_factor is not None and prev_factor > factor:
            x0 = upsample_volume(jnp.asarray(x_init), prev_factor // factor, (g.nx, g.ny, g.nz))
        else:
            x0 = None
        _status(
            ctx.status_path,
            state="running",
            stage=stage_name,
            level_factor=int(factor),
            active_dofs=list(active_pose),
        )
        geom_eff = geometry_with_axis_state(geometry, g, d, setup_state)
        det_grid = (
            None
            if bool(ctx.args.canonical_det_grid)
            else level_detector_grid(d, state=setup_state, factor=int(factor))
        )
        cfg_level = replace(cfg_base, recon_L=None, loss=resolve_loss_for_level(cfg_base.loss, int(factor)))
        t0 = time.perf_counter()
        x_lvl, params_current, info = align(
            geom_eff,
            g,
            d,
            y,
            cfg=cfg_level,
            init_x=x0,
            init_params5=params_current,
            observer=make_observer(g, int(factor)),
            det_grid_override=det_grid,
        )
        elapsed = time.perf_counter() - t0
        x_init = np.asarray(x_lvl, dtype=np.float32)
        prev_factor = int(factor)
        _write_json(stage_dir / f"level_{factor:02d}_align_info.json", info)
        if info.get("stopped_by_observer") and info.get("observer_action") == "stop_run":
            break
        if not info.get("outer_stats"):
            stage_stats.append(
                {
                    "stage": stage_name,
                    "level_factor": int(factor),
                    "active_dofs": ",".join(active_pose),
                    "elapsed_seconds": float(elapsed),
                    "note": "no outer stats emitted",
                }
            )
    params_np = np.asarray(params_current, dtype=np.float32)
    _write_params_csv(stage_dir / "params.csv", params_np)
    _write_json(stage_dir / "align_info.json", {"outer_stats": stage_stats, "params_summary": real_lamino_pose_params_summary(params_np)})
    _write_json(stage_dir / "geometry_calibration_state.json", setup_state.to_calibration_state().to_dict())
    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=np.asarray(x_init, dtype=np.float32),
        grid=scale_grid(grid, int(prev_factor or 1)),
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
    )
    _write_json(
        stage_dir / "stage_manifest.json",
        {
            "stage": stage_name,
            "status": "completed",
            "active_dofs": list(active_pose),
            "levels": list(levels),
            "bounds": bounds,
            "stats_count": len(stage_stats),
            "params_summary": real_lamino_pose_params_summary(params_np),
            "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
            "artifacts": products,
        },
    )
    return np.asarray(x_init, dtype=np.float32), params_np, stage_stats


def _final_reconstruct(
    ctx: RunContext,
    *,
    geometry: LaminographyGeometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    full_nz: int,
    setup_state: GeometryCalibrationState,
    params5: np.ndarray,
) -> np.ndarray:
    stage_dir = ctx.stage_dir("05_final")
    stage_dir.mkdir(parents=True, exist_ok=True)
    _status(ctx.status_path, state="running", stage="05_final", message="final_fista_tv")
    geom_eff = geometry_with_axis_state(geometry, grid, detector, setup_state)
    det_grid = None if bool(ctx.args.canonical_det_grid) else level_detector_grid(detector, state=setup_state, factor=1)
    params_jax = jnp.asarray(params5, dtype=jnp.float32)

    class _PoseAugmentedGeometry:
        def pose_for_view(self, i):
            T_nom = jnp.asarray(geom_eff.pose_for_view(i), dtype=jnp.float32)
            return tuple(map(tuple, T_nom @ se3_from_5d(params_jax[i])))

        def rays_for_view(self, i):
            return geom_eff.rays_for_view(i)

    t0 = time.perf_counter()
    vol, info = fista_tv(
        _PoseAugmentedGeometry(),
        grid,
        detector,
        jnp.asarray(projections, dtype=jnp.float32),
        config=FistaConfig(
            iters=max(1, int(ctx.args.recon_iters)),
            lambda_tv=float(ctx.args.lambda_tv),
            regulariser=str(ctx.args.regulariser),
            tv_prox_iters=int(ctx.args.tv_prox_iters),
            views_per_batch=None if int(ctx.args.views_per_batch) == 0 else max(1, int(ctx.args.views_per_batch)),
            checkpoint_projector=True,
            gather_dtype=str(ctx.args.gather_dtype),
            positivity=bool(ctx.args.recon_positivity),
        ),
        det_grid=det_grid,
    )
    vol_np = np.asarray(vol, dtype=np.float32)
    elapsed = time.perf_counter() - t0
    np.save(stage_dir / "final_setup_aligned_slab.npy", vol_np)
    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=vol_np,
        grid=grid,
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
        suffix="aligned",
    )
    _write_json(
        stage_dir / "stage_manifest.json",
        {
            "stage": "05_final",
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "recon_info": info,
            "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
            "params_summary": real_lamino_pose_params_summary(params5),
            "artifacts": products,
        },
    )
    _write_json(stage_dir / "align_info.json", {"recon_info": info, "params_summary": real_lamino_pose_params_summary(params5)})
    _write_json(stage_dir / "geometry_calibration_state.json", setup_state.to_calibration_state().to_dict())
    return vol_np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native staged setup+pose alignment for real k11-54014 laminography.")
    parser.add_argument("--input", required=True, help="Input NXtomo/HDF5 file with entry/imaging/data projections.")
    parser.add_argument(
        "--expected-projection-shape",
        type=_parse_shape3,
        default=None,
        metavar="N,NV,NU",
        help=(
            "Optional loaded projection shape contract. Accepts N,NV,NU or NxNVxNU; "
            "when omitted, the runner derives dimensions from the input stack."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--tilt-deg", type=float, default=34.4)
    parser.add_argument("--tilt-about", choices=["x", "z"], default="x")
    parser.add_argument("--flip-u", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-v", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transpose-detector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preview-z", type=int, default=209)
    parser.add_argument("--slab-center-z", type=int, default=209)
    parser.add_argument("--slab-nz", type=int, default=96)
    parser.add_argument("--stack-z-range", default="198:220")
    parser.add_argument("--levels-setup", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--levels-phi", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--levels-dx-dz", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--levels-polish", nargs="+", type=int, default=[2, 1])
    parser.add_argument("--outer-iters", type=int, default=8)
    parser.add_argument("--recon-iters", type=int, default=40)
    parser.add_argument("--tv-prox-iters", type=int, default=16)
    parser.add_argument("--lambda-tv", type=float, default=0.008)
    parser.add_argument("--align-profile", choices=["lightning", "tortoise"], default="lightning")
    parser.add_argument("--projector-backend", choices=["pallas", "jax"], default="pallas")
    parser.add_argument("--regulariser", choices=["huber_tv", "tv"], default="huber_tv")
    parser.add_argument("--quality-tier", choices=["fast", "reference"], default="fast")
    parser.add_argument("--fallback-policy", choices=["fallback", "strict"], default="fallback")
    parser.add_argument(
        "--pose-model",
        choices=["per_view", "polynomial", "spline"],
        default="spline",
    )
    parser.add_argument("--knot-spacing", type=int, default=8)
    parser.add_argument("--pose-degree", type=int, default=3)
    parser.add_argument(
        "--canonical-det-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the canonical detector grid in pose/final stages. This is a Pallas diagnostic "
            "mode: it can ignore calibrated detector-grid offsets/roll in those stages."
        ),
    )
    parser.add_argument("--projection-background", choices=["none", "view_median", "edge_median"], default="edge_median")
    parser.add_argument("--background-edge-px", type=int, default=16)
    parser.add_argument("--recon-positivity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--views-per-batch", type=int, default=0)
    parser.add_argument("--gather-dtype", default="bf16")
    parser.add_argument("--gn-damping", type=float, default=1e-3)
    parser.add_argument("--filter", dest="filter_name", default="ramp")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-rel", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--snapshot-max-cols", type=int, default=6)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced real-data diagnostic that still exercises setup, pose, and final reconstruction.",
    )
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if bool(args.smoke):
        args.slab_nz = min(int(args.slab_nz), 48)
        args.levels_setup = [8]
        args.levels_phi = [8]
        args.levels_dx_dz = [8]
        args.levels_polish = [8]
        args.outer_iters = min(int(args.outer_iters), 1)
        args.recon_iters = min(int(args.recon_iters), 3)
        args.tv_prox_iters = min(int(args.tv_prox_iters), 2)
        args.snapshot_max_cols = min(int(args.snapshot_max_cols), 4)
        if int(args.views_per_batch) == 0:
            args.views_per_batch = 16
    return args


def main() -> int:
    args = _parse_args()
    run_root = Path(args.out)
    if run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists and is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "runner.pid").write_text(str(os.getpid()) + "\n")
    monitor = GpuMonitor(run_root / "gpu_memory.csv")
    monitor.start()
    ctx = RunContext(args)
    started = datetime.now().isoformat(timespec="seconds")
    _status(ctx.status_path, state="starting", started_at=started)
    try:
        loaded = load_real_laminography_input(
            Path(args.input),
            flip_u=bool(args.flip_u),
            flip_v=bool(args.flip_v),
            transpose_detector=bool(args.transpose_detector),
        )
        raw_projections = loaded.projections
        thetas = loaded.thetas_deg
        raw_projections, thetas = _validate_loaded_input(
            raw_projections,
            thetas,
            expected_projection_shape=args.expected_projection_shape,
        )
        projections, background_offsets = _apply_projection_background(
            raw_projections,
            mode=str(args.projection_background),
            edge_px=int(args.background_edge_px),
        )
        np.save(run_root / "projection_background_offsets.npy", background_offsets.astype(np.float32))
        n_views, nv, nu = projections.shape
        full_nz = int(nv)
        center_phys_z = real_lamino_global_z_to_phys(int(args.slab_center_z), full_nz=full_nz)
        grid = Grid(nx=int(nu), ny=int(nu), nz=int(args.slab_nz), vx=1.0, vy=1.0, vz=1.0, vol_center=(0.0, 0.0, center_phys_z))
        detector = Detector(nu=int(nu), nv=int(nv), du=1.0, dv=1.0, det_center=(0.0, 0.0))
        preview_local_z = real_lamino_global_z_to_local_index(int(args.preview_z), full_nz=full_nz, grid=grid)
        if not 0 <= preview_local_z < int(grid.nz):
            raise ValueError(f"preview z {args.preview_z} maps outside slab local z={preview_local_z}")
        geometry = LaminographyGeometry(grid=grid, detector=detector, thetas_deg=thetas, tilt_deg=float(args.tilt_deg), tilt_about=str(args.tilt_about))

        setup_schedule = AlignmentSchedule(
            name="real_lamino_setup_geometry",
            stages=(
                AlignmentStage("cor", ("det_u_px",), "bilevel_cv", "validation_lm", maxiter=int(args.outer_iters), early_stop=bool(args.early_stop)),
                AlignmentStage("detector_roll", ("detector_roll_deg",), "bilevel_cv", "validation_lm", maxiter=int(args.outer_iters), early_stop=bool(args.early_stop)),
                AlignmentStage("axis_direction", ("axis_rot_x_deg", "axis_rot_y_deg"), "bilevel_cv", "validation_lm", gauge_policy="diagnose_only", maxiter=int(args.outer_iters), early_stop=bool(args.early_stop)),
            ),
            metadata={"real_data": True, "slab_nz": int(args.slab_nz), "preview_z": int(args.preview_z)},
        ).validate()
        pose_schedule = [
            {"name": "02_pose_phi", "active": ("phi",), "levels": tuple(args.levels_phi), "bounds": "phi=-0.0872665:0.0872665"},
            {"name": "03_pose_dx_dz", "active": ("dx", "dz"), "levels": tuple(args.levels_dx_dz), "bounds": "dx=-16:16,dz=-16:16"},
            {
                "name": "04_pose_polish",
                "active": ("alpha", "beta", "phi", "dx", "dz"),
                "levels": tuple(args.levels_polish),
                "bounds": "alpha=-0.0349066:0.0349066,beta=-0.0349066:0.0349066,phi=-0.0872665:0.0872665,dx=-16:16,dz=-16:16",
            },
        ]
        _write_json(
            run_root / "run_manifest.json",
            {
                "status": "running",
                "started_at": started,
                "input": str(args.input),
                "expected_projection_shape": (
                    None
                    if args.expected_projection_shape is None
                    else list(args.expected_projection_shape)
                ),
                "input_shape": list(projections.shape),
                "raw_projection_stats": _projection_stats(raw_projections),
                "working_projection_stats": _projection_stats(projections),
                "projection_preprocessing": {
                    "background_mode": str(args.projection_background),
                    "background_edge_px": int(args.background_edge_px),
                    "background_offsets_percentiles": [
                        float(v) for v in np.nanpercentile(background_offsets, [0, 5, 50, 95, 100])
                    ],
                    "background_offsets_file": "projection_background_offsets.npy",
                    "baseline_reconstruction_uses": "raw_projections",
                    "alignment_and_fista_use": "background_corrected_projections",
                },
                "reconstruction": {
                    "algorithm": "fista_tv",
                    "lambda_tv": float(args.lambda_tv),
                    "regulariser": str(args.regulariser),
                    "tv_prox_iters": int(args.tv_prox_iters),
                    "positivity": bool(args.recon_positivity),
                    "gather_dtype": str(args.gather_dtype),
                    "views_per_batch": int(args.views_per_batch),
                    "canonical_det_grid": bool(args.canonical_det_grid),
                },
                "alignment_policy": {
                    "align_profile": str(args.align_profile),
                    "projector_backend_requested": str(args.projector_backend),
                    "regulariser": str(args.regulariser),
                    "quality_tier": str(args.quality_tier),
                    "fallback_policy": str(args.fallback_policy),
                    "gather_dtype": str(args.gather_dtype),
                    "views_per_batch": int(args.views_per_batch),
                    "canonical_det_grid": bool(args.canonical_det_grid),
                    "smoke": bool(args.smoke),
                    "speed_claim_requires_stage_actual_backend": True,
                    "helper_only_timings_are_not_workflow_claims": True,
                },
                "worktree": _commit_info(Path.cwd()),
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "convention": {
                    "flip_u": bool(args.flip_u),
                    "flip_v": bool(args.flip_v),
                    "transpose_detector": bool(args.transpose_detector),
                    "tilt_deg": float(args.tilt_deg),
                    "tilt_about": str(args.tilt_about),
                },
                "slab": {
                    "slab_nz": int(args.slab_nz),
                    "slab_center_global_z": int(args.slab_center_z),
                    "preview_global_z": int(args.preview_z),
                    "preview_local_z": int(preview_local_z),
                    "z_stack_range": list(ctx.stack_z_range),
                    "grid": grid.to_dict(),
                },
                "setup_schedule": _schedule_dict(setup_schedule),
                "pose_schedule": pose_schedule,
                "early_stop": {
                    "enabled": bool(args.early_stop),
                    "rel_impr": float(args.early_stop_rel),
                    "patience": int(args.early_stop_patience),
                },
            },
        )
        run_baseline(ctx, geometry=geometry, grid=grid, detector=detector, projections=raw_projections, full_nz=full_nz)
        params5 = np.zeros((n_views, 5), dtype=np.float32)
        setup_state = GeometryCalibrationState.from_geometry(geometry, active_geometry_dofs=())

        setup_bounds = {
            ("det_u_px",): "det_u_px=-24:24",
            ("detector_roll_deg",): "detector_roll_deg=-10:10",
            ("axis_rot_x_deg", "axis_rot_y_deg"): "axis_rot_x_deg=-15:15,axis_rot_y_deg=-15:15",
        }
        for idx, stage in enumerate(setup_schedule.stages, start=1):
            _, setup_state, stats = run_setup_stage(
                ctx,
                stage_dir=ctx.stage_dir(f"01_setup_geometry/{idx:02d}_{stage.name}"),
                stage_name=f"01_setup_geometry/{idx:02d}_{stage.name}",
                active_setup=tuple(stage.active_dofs),
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                full_nz=full_nz,
                setup_state=setup_state,
                params5=params5,
                levels=tuple(int(v) for v in args.levels_setup),
                bounds=setup_bounds.get(tuple(stage.active_dofs), ""),
            )
            ctx.stage_records.append({"stage": stage.name, "stats_count": len(stats), "geometry_calibration_state": setup_state.to_calibration_state().to_dict()})

        if not bool(args.skip_pose):
            for stage in pose_schedule:
                _, params5, stats = run_pose_stage(
                    ctx,
                    stage_dir=ctx.stage_dir(str(stage["name"])),
                    stage_name=str(stage["name"]),
                    active_pose=tuple(stage["active"]),
                    geometry=geometry,
                    grid=grid,
                    detector=detector,
                    projections=projections,
                    full_nz=full_nz,
                    setup_state=setup_state,
                    params5=params5,
                    levels=tuple(int(v) for v in stage["levels"]),
                    bounds=str(stage["bounds"]),
                )
                ctx.stage_records.append({"stage": stage["name"], "stats_count": len(stats), "params_summary": real_lamino_pose_params_summary(params5)})

        final_volume = _final_reconstruct(
            ctx,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=setup_state,
            params5=params5,
        )
        _write_params_csv(run_root / "05_final" / "params.csv", params5)
        final_payload = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "stage_records": ctx.stage_records,
            "final_setup_estimates": setup_state.to_calibration_state().to_dict(),
            "final_pose_summary": real_lamino_pose_params_summary(params5),
            "final_volume_shape": list(final_volume.shape),
        }
        _write_json(run_root / "run_manifest.json", {**json.loads((run_root / "run_manifest.json").read_text()), **final_payload})
        _status(ctx.status_path, state="completed", stage="complete", **final_payload)
        return 0
    except Exception as exc:
        _status(ctx.status_path, state="failed", stage="error", error=repr(exc))
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
