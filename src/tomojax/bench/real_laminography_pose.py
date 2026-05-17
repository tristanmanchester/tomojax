"""Pose-stage adapters for real-laminography benchmark workflows."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import json
import math
import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import (
    AlignConfig,
    align,
    geometry_with_axis_state,
    level_detector_grid,
    resolve_loss_for_level,
)
from tomojax.bench._real_laminography_visuals import resize_nearest_2d, save_uint8_png
from tomojax.bench.real_laminography_artifacts import save_real_lamino_z_stack
from tomojax.bench.real_laminography_planning import real_lamino_xy_at_global_z
from tomojax.bench.real_laminography_report import real_lamino_pose_params_summary
from tomojax.bench.real_laminography_runtime import (
    append_real_lamino_csv,
    real_lamino_json_safe,
    update_real_lamino_status,
    write_real_lamino_json,
    write_real_lamino_params_csv,
)
from tomojax.recon.multires import bin_projections, scale_detector, scale_grid, upsample_volume

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from tomojax.align.api import GeometryCalibrationState
    from tomojax.core.geometry import Detector, Grid, LaminographyGeometry


def _make_pose_cfg(
    args: Any,
    *,
    active_pose: tuple[str, ...],
    bounds: str,
) -> AlignConfig:
    return AlignConfig(
        align_profile=str(args.align_profile),
        outer_iters=int(args.outer_iters),
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
        geometry_dofs=(),
        bounds=bounds,
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


def _save_pose_checkpoint(
    path: Path,
    *,
    x: np.ndarray,
    params: np.ndarray,
    setup: GeometryCalibrationState,
    stat: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=np.asarray(x, dtype=np.float32),
        params5=np.asarray(params, dtype=np.float32),
        geometry_calibration_state=json.dumps(
            real_lamino_json_safe(setup.to_calibration_state().to_dict()),
            sort_keys=True,
        ),
        stat=json.dumps(real_lamino_json_safe(stat), sort_keys=True),
    )
    latest = path.parent / "latest.npz"
    with suppress(Exception):
        latest.write_bytes(path.read_bytes())
    return str(path)


def run_real_lamino_pose_stage(
    ctx: Any,
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
    """Run one real-laminography pose stage and write its artifact bundle."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = _make_pose_cfg(ctx.args, active_pose=active_pose, bounds=bounds)
    stage_stats: list[dict[str, Any]] = []
    x_init = None
    prev_factor: int | None = None
    params_current = jnp.asarray(params5, dtype=jnp.float32)

    def make_observer(
        level_grid: Grid,
        level_factor: int,
    ) -> Callable[[Any, Any, dict[str, Any]], str]:
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
            xy = real_lamino_xy_at_global_z(
                x_np,
                grid=level_grid,
                full_nz=full_nz,
                global_z=ctx.preview_global_z,
            )
            ref = (
                resize_nearest_2d(ctx.naive_slice, xy.shape)
                if ctx.naive_slice is not None
                else None
            )
            stem = (
                f"outer_{len(stage_stats) + 1:03d}_level{level_factor:02d}_"
                f"iter{int(stat.get('outer_idx', len(stage_stats) + 1)):02d}"
            )
            preview_dir = stage_dir / "timeline_z"
            save_uint8_png(
                preview_dir / "slices" / f"{stem}_global_z{ctx.preview_global_z:03d}.png",
                xy,
            )
            if ref is not None:
                save_uint8_png(preview_dir / "delta_vs_naive" / f"{stem}_delta.png", xy - ref)
            save_real_lamino_z_stack(
                preview_dir
                / "z_stacks"
                / f"{stem}_global_z{ctx.stack_z_range[0]:03d}_{ctx.stack_z_range[1]:03d}.png",
                x_np,
                grid=level_grid,
                full_nz=full_nz,
                z_range=ctx.stack_z_range,
                max_cols=int(ctx.args.snapshot_max_cols),
            )
            _save_pose_checkpoint(
                stage_dir / "checkpoints" / f"{stem}.npz",
                x=x_np,
                params=params_np,
                setup=setup_state,
                stat=stat,
            )
            stage_stats.append(stat)
            append_real_lamino_csv(
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

    for factor in levels:
        g = scale_grid(grid, factor)
        d = scale_detector(detector, factor)
        y = bin_projections(jnp.asarray(projections, dtype=jnp.float32), factor)
        if x_init is not None and prev_factor is not None and prev_factor > factor:
            x0 = upsample_volume(jnp.asarray(x_init), prev_factor // factor, (g.nx, g.ny, g.nz))
        else:
            x0 = None
        update_real_lamino_status(
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
        cfg_level = replace(
            cfg_base,
            recon_L=None,
            loss=resolve_loss_for_level(cfg_base.loss, int(factor)),
        )
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
        write_real_lamino_json(stage_dir / f"level_{factor:02d}_align_info.json", info)
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
    write_real_lamino_params_csv(stage_dir / "params.csv", params_np)
    write_real_lamino_json(
        stage_dir / "align_info.json",
        {
            "outer_stats": stage_stats,
            "params_summary": real_lamino_pose_params_summary(params_np),
        },
    )
    write_real_lamino_json(
        stage_dir / "geometry_calibration_state.json",
        setup_state.to_calibration_state().to_dict(),
    )
    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=np.asarray(x_init, dtype=np.float32),
        grid=scale_grid(grid, int(prev_factor or 1)),
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
    )
    write_real_lamino_json(
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


__all__ = ["run_real_lamino_pose_stage"]
