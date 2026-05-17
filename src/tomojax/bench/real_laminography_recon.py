"""Reconstruction-stage glue for real-laminography benchmark workflows."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from tomojax.align.api import geometry_with_axis_state, level_detector_grid
from tomojax.bench.real_laminography_report import (
    REAL_LAMINO_COR_ONLY_STAGE,
    real_lamino_loss_summary,
)
from tomojax.bench.real_laminography_runtime import (
    append_real_lamino_csv,
    update_real_lamino_status,
    write_real_lamino_json,
)
from tomojax.recon.fista_tv import FistaConfig, fista_tv


def run_cor_only_fista_stage(
    ctx: Any,
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


__all__ = ["run_cor_only_fista_stage"]
