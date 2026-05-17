"""Setup-stage adapters for real-laminography diagnostics."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import replace
import json
import math
import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

# check-public-imports: allow-private
from tomojax.align._setup_stage import (
    _optimize_setup_geometry_bilevel_for_level as optimize_reference_setup_geometry_bilevel_for_level,
)
from tomojax.align.api import (
    AlignConfig,
    AlignmentState,
    GeometryCalibrationState,
    PoseState,
    SetupGeometryState,
    loss_spec_name,
    resolve_loss_for_level,
)
from tomojax.bench._real_laminography_visuals import resize_nearest_2d, save_uint8_png
from tomojax.bench.real_laminography_artifacts import save_real_lamino_z_stack
from tomojax.bench.real_laminography_planning import real_lamino_xy_at_global_z
from tomojax.bench.real_laminography_runtime import (
    append_real_lamino_csv,
    real_lamino_json_safe,
    update_real_lamino_status,
    write_real_lamino_json,
)
from tomojax.recon.multires import bin_projections, scale_detector, scale_grid, upsample_volume

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tomojax.core.geometry import Detector, Grid, LaminographyGeometry

__all__ = [
    "optimize_reference_setup_geometry_bilevel_for_level",
    "run_real_lamino_setup_stage",
]


def _parse_setup_bounds(text: str) -> dict[str, tuple[float, float]]:
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
    parsed = _parse_setup_bounds(bounds)
    if not parsed:
        return state, {}
    updates: dict[str, float] = {}
    clipped: dict[str, dict[str, float | bool]] = {}
    for name, (lo, hi) in parsed.items():
        before = float(getattr(state, name))
        after = float(np.clip(before, lo, hi))
        updates[name] = after
        clipped[name] = {
            "before": before,
            "after": after,
            "lo": lo,
            "hi": hi,
            "clipped": before != after,
        }
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


def _make_setup_cfg(
    args: Any,
    *,
    active_setup: tuple[str, ...],
) -> AlignConfig:
    return AlignConfig(
        align_profile=str(args.align_profile),
        outer_iters=1,
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
        optimise_dofs=None,
        geometry_dofs=active_setup,
        bounds="",
        pose_model=str(getattr(args, "pose_model", "spline")),
        knot_spacing=int(getattr(args, "knot_spacing", 8)),
        degree=int(getattr(args, "pose_degree", 3)),
        gauge_fix="none",
        mask_vol="cyl",
        recon_positivity=bool(args.recon_positivity),
        seed_translations=False,
        early_stop=bool(args.early_stop),
        early_stop_rel_impr=float(args.early_stop_rel),
        early_stop_patience=int(args.early_stop_patience),
        log_summary=True,
        log_compact=True,
    )


def _save_setup_checkpoint(
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


def run_real_lamino_setup_stage(
    ctx: Any,
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
    """Run one real-laminography setup-geometry stage and write its artifacts."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = _make_setup_cfg(ctx.args, active_setup=active_setup)
    stage_stats: list[dict[str, Any]] = []
    x_init = None
    prev_factor: int | None = None
    setup_state = replace(setup_state, active_geometry_dofs=active_setup)
    for level_index, factor in enumerate(levels):
        g = scale_grid(grid, factor)
        d = scale_detector(detector, factor)
        y = bin_projections(jnp.asarray(projections, dtype=jnp.float32), factor)
        if x_init is not None and prev_factor is not None and prev_factor > factor:
            x_level = upsample_volume(
                jnp.asarray(x_init),
                prev_factor // factor,
                (g.nx, g.ny, g.nz),
            )
        else:
            x_level = None
        last_loss = math.inf
        stale = 0
        parity_outer_count = (
            None if level_outer_counts is None else level_outer_counts.get(int(factor))
        )
        outer_limit = int(parity_outer_count or ctx.args.outer_iters)
        for outer in range(1, outer_limit + 1):
            update_real_lamino_status(
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
            xy = real_lamino_xy_at_global_z(
                x_np,
                grid=preview_grid,
                full_nz=full_nz,
                global_z=ctx.preview_global_z,
            )
            ref = (
                resize_nearest_2d(ctx.naive_slice, xy.shape)
                if ctx.naive_slice is not None
                else None
            )
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
                grid=preview_grid,
                full_nz=full_nz,
                z_range=ctx.stack_z_range,
                max_cols=int(ctx.args.snapshot_max_cols),
            )
            _save_setup_checkpoint(
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
            append_real_lamino_csv(stage_dir / "stage_summary.csv", stat, fields)
            loss_after = float(stat.get("geometry_loss_after", last_loss))
            rel = (
                (last_loss - loss_after) / max(abs(last_loss), 1e-6)
                if math.isfinite(last_loss)
                else math.inf
            )
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
    write_real_lamino_json(
        stage_dir / "align_info.json",
        {"outer_stats": stage_stats, "active_geometry_dofs": list(active_setup)},
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
            "active_dofs": list(active_setup),
            "levels": list(levels),
            "bounds": bounds,
            "stats_count": len(stage_stats),
            "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
            "artifacts": products,
        },
    )
    return np.asarray(x_init, dtype=np.float32), setup_state, stage_stats
