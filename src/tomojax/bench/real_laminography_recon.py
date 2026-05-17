"""Reconstruction-stage glue for real-laminography benchmark workflows."""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.align.api import geometry_with_axis_state, level_detector_grid, se3_from_5d
from tomojax.bench._real_laminography_visuals import save_uint8_png
from tomojax.bench.real_laminography_planning import (
    real_lamino_xy_at_global_z,
    select_real_lamino_final_candidates,
)
from tomojax.bench.real_laminography_report import (
    REAL_LAMINO_COR_ONLY_STAGE,
    mark_real_lamino_stage_failed,
    real_lamino_loss_summary,
    real_lamino_pose_params_summary,
    validate_real_lamino_stage_output,
)
from tomojax.bench.real_laminography_runtime import (
    append_real_lamino_csv,
    update_real_lamino_status,
    write_real_lamino_json,
)
from tomojax.io import read_json_object
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import FistaConfig, fista_tv

if TYPE_CHECKING:
    from tomojax.bench.real_laminography_context import RealLaminoRunContext


def run_baseline_stage(
    ctx: RealLaminoRunContext,
    *,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
) -> np.ndarray:
    """Run the raw-projection baseline FBP and write the staged artifact bundle."""
    stage_dir = ctx.stage_dir("00_baseline")
    stage_dir.mkdir(parents=True, exist_ok=True)
    update_real_lamino_status(
        ctx.status_path,
        state="running",
        stage="00_baseline",
        message="baseline_fbp",
    )
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
    ctx.naive_slice = real_lamino_xy_at_global_z(
        vol,
        grid=grid,
        full_nz=full_nz,
        global_z=ctx.preview_global_z,
    )
    save_uint8_png(
        stage_dir / f"naive_or_input_xy_global_z{ctx.preview_global_z:03d}.png",
        ctx.naive_slice,
    )
    ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=vol,
        grid=grid,
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
        suffix="aligned",
    )
    save_uint8_png(
        stage_dir / "measured_sinogram.png", projections[:, projections.shape[1] // 2, :]
    )
    manifest = {
        "stage": "00_baseline",
        "status": "completed",
        "elapsed_seconds": elapsed,
        "volume_shape": list(vol.shape),
        "preview_z": int(ctx.preview_global_z),
        "z_stack_range": list(ctx.stack_z_range),
    }
    write_real_lamino_json(stage_dir / "stage_manifest.json", manifest)
    write_real_lamino_json(stage_dir / "align_info.json", {"stage": "baseline", "outer_stats": []})
    append_real_lamino_csv(
        stage_dir / "stage_summary.csv",
        {"stage": "00_baseline", "status": "completed", "elapsed_seconds": elapsed},
        ["stage", "status", "elapsed_seconds"],
    )
    return vol


def run_cor_only_fista_stage(
    ctx: RealLaminoRunContext,
    *,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    setup_state: Any,
) -> np.ndarray:
    """Run the COR-only FISTA comparator and write its staged artifacts."""
    stage_dir = ctx.stage_dir(REAL_LAMINO_COR_ONLY_STAGE)
    stage_dir.mkdir(parents=True, exist_ok=True)
    update_real_lamino_status(
        ctx.status_path,
        state="running",
        stage=REAL_LAMINO_COR_ONLY_STAGE,
    )
    geom_eff = geometry_with_axis_state(geometry, grid, detector, setup_state)
    det_grid = (
        None
        if bool(ctx.args.canonical_det_grid)
        else level_detector_grid(detector, state=setup_state, factor=1)
    )
    t0 = time.perf_counter()
    vol, info = fista_tv(
        geom_eff,
        grid,
        detector,
        jnp.asarray(projections, dtype=jnp.float32),
        config=FistaConfig(
            iters=max(1, int(ctx.args.recon_iters)),
            lambda_tv=float(ctx.args.lambda_tv),
            regulariser=str(ctx.args.regulariser),
            tv_prox_iters=int(ctx.args.tv_prox_iters),
            views_per_batch=max(1, int(ctx.args.views_per_batch)),
            checkpoint_projector=True,
            gather_dtype=str(ctx.args.gather_dtype),
            positivity=bool(ctx.args.recon_positivity),
        ),
        det_grid=det_grid,
    )
    vol_np = np.asarray(vol, dtype=np.float32)
    elapsed = time.perf_counter() - t0
    np.save(stage_dir / "cor_only_fista_fullres_slab.npy", vol_np)
    products = ctx.save_stage_products(
        stage_dir=stage_dir,
        volume=vol_np,
        grid=grid,
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
        suffix="aligned",
    )
    calibration_state = setup_state.to_calibration_state().to_dict()
    manifest = {
        "stage": REAL_LAMINO_COR_ONLY_STAGE,
        "status": "completed",
        "elapsed_seconds": float(elapsed),
        "active_dofs": ["det_u_px"],
        "volume_shape": list(vol_np.shape),
        "fista_info": info,
        "geometry_calibration_state": calibration_state,
        "setup_state": calibration_state,
        "artifacts": products,
    }
    write_real_lamino_json(stage_dir / "stage_manifest.json", manifest)
    write_real_lamino_json(stage_dir / "align_info.json", {"fista_info": info})
    write_real_lamino_json(stage_dir / "geometry_calibration_state.json", calibration_state)
    loss = real_lamino_loss_summary(info)
    append_real_lamino_csv(
        stage_dir / "stage_summary.csv",
        {
            "stage": REAL_LAMINO_COR_ONLY_STAGE,
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "loss_first": loss["first"],
            "loss_last": loss["last"],
        },
        ["stage", "status", "elapsed_seconds", "loss_first", "loss_last"],
    )
    return vol_np


def run_final_reconstruction_stage(
    ctx: Any,
    *,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    setup_state: Any,
    params5: np.ndarray,
) -> np.ndarray:
    """Run the final setup-and-pose-aware FISTA reconstruction stage."""
    stage_dir = ctx.stage_dir("05_final")
    stage_dir.mkdir(parents=True, exist_ok=True)
    update_real_lamino_status(
        ctx.status_path,
        state="running",
        stage="05_final",
        message="final_fista_tv",
    )
    geom_eff = geometry_with_axis_state(geometry, grid, detector, setup_state)
    det_grid = (
        None
        if bool(ctx.args.canonical_det_grid)
        else level_detector_grid(detector, state=setup_state, factor=1)
    )
    params_jax = jnp.asarray(params5, dtype=jnp.float32)

    class _PoseAugmentedGeometry:
        def pose_for_view(self, i: Any) -> tuple[tuple[Any, ...], ...]:
            t_nom = jnp.asarray(geom_eff.pose_for_view(i), dtype=jnp.float32)
            return tuple(map(tuple, t_nom @ se3_from_5d(params_jax[i])))

        def rays_for_view(self, i: Any) -> Any:
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
            views_per_batch=(
                None
                if int(ctx.args.views_per_batch) == 0
                else max(1, int(ctx.args.views_per_batch))
            ),
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
    calibration_state = setup_state.to_calibration_state().to_dict()
    params_summary = real_lamino_pose_params_summary(params5)
    write_real_lamino_json(
        stage_dir / "stage_manifest.json",
        {
            "stage": "05_final",
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "recon_info": info,
            "geometry_calibration_state": calibration_state,
            "params_summary": params_summary,
            "artifacts": products,
        },
    )
    write_real_lamino_json(
        stage_dir / "align_info.json",
        {"recon_info": info, "params_summary": params_summary},
    )
    write_real_lamino_json(stage_dir / "geometry_calibration_state.json", calibration_state)
    return vol_np


def run_best_final_reconstruction_stage(
    ctx: Any,
    *,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    candidates: list[dict[str, Any]],
    native: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run final FISTA candidates and publish the lowest-loss final artifact."""
    del native
    if not candidates:
        raise ValueError("at least one final reconstruction candidate is required")
    candidate_policy = str(getattr(getattr(ctx, "args", object()), "final_candidate_policy", "all"))
    candidates_to_score = select_real_lamino_final_candidates(
        candidates,
        policy=candidate_policy,
    )
    root = Path(ctx.run_root)
    scratch_root = root / "05_final_candidates"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scored: list[dict[str, Any]] = []
    original_stage_dir = ctx.stage_dir
    try:
        for idx, candidate in enumerate(candidates_to_score):
            label = str(candidate["label"]).replace("/", "__")
            candidate_root = scratch_root / f"{idx:02d}_{label}"
            ctx.stage_dir = lambda name, candidate_root=candidate_root: candidate_root / name
            volume = run_final_reconstruction_stage(
                ctx,
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                full_nz=full_nz,
                setup_state=candidate["setup_state"],
                params5=np.asarray(candidate["params5"], dtype=np.float32),
            )
            manifest = dict(read_json_object(candidate_root / "05_final" / "stage_manifest.json"))
            loss_last = real_lamino_loss_summary(manifest.get("recon_info", {})).get("last")
            validation = validate_real_lamino_stage_output(
                candidate_root / "05_final",
                stage_name=f"05_final:{candidate['source_stage']}",
                volume=volume,
                params5=np.asarray(candidate["params5"], dtype=np.float32),
                stats=[],
                require_data_loss=False,
            )
            if loss_last is None:
                validation["passed"] = False
                validation["failures"].append("final candidate loss is missing or non-finite")
            if not validation["passed"]:
                mark_real_lamino_stage_failed(
                    candidate_root / "05_final",
                    stage_name="05_final",
                    validation=validation,
                )
                continue
            scored.append(
                {
                    **candidate,
                    "candidate_dir": candidate_root / "05_final",
                    "volume": volume,
                    "loss_last": float(loss_last),
                }
            )
    finally:
        ctx.stage_dir = original_stage_dir
    if not scored:
        raise RuntimeError("no finite final reconstruction candidates passed validation")
    best = min(scored, key=lambda item: float(item["loss_last"]))
    final_dir = root / "05_final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(best["candidate_dir"], final_dir)
    manifest = dict(read_json_object(final_dir / "stage_manifest.json"))
    manifest["volume_shape"] = list(np.asarray(best["volume"]).shape)
    manifest["selected_final_candidate"] = {
        "label": best["label"],
        "source_stage": best["source_stage"],
        "loss_last": best["loss_last"],
        "candidate_policy": candidate_policy,
        "candidates": [
            {
                "label": item["label"],
                "source_stage": item["source_stage"],
                "loss_last": item["loss_last"],
                "candidate_dir": str(item["candidate_dir"]),
            }
            for item in scored
        ],
    }
    write_real_lamino_json(final_dir / "stage_manifest.json", manifest)
    return np.asarray(best["volume"], dtype=np.float32), best


__all__ = [
    "run_baseline_stage",
    "run_best_final_reconstruction_stage",
    "run_cor_only_fista_stage",
    "run_final_reconstruction_stage",
]
