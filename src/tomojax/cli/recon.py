from __future__ import annotations

import argparse
from dataclasses import replace
import logging
import numpy as np
import jax.numpy as jnp
import os
import sys

from ..data.geometry_meta import build_geometry_from_meta
from ..data.io_hdf5 import NXTomoMetadata, load_nxtomo, save_nxtomo
from ..core.geometry import Detector, Grid
from ..recon.fbp import fbp
from ..recon.fista_tv import FistaConfig, fista_tv
from ..recon.quicklook import save_quicklook_png
from ..recon.spdhg_tv import spdhg_tv, SPDHGConfig
from ..utils.logging import setup_logging, log_jax_env
from ..utils.memory import estimate_views_per_batch_info
from ..utils.axes import DISK_VOLUME_AXES
from ._runtime import transfer_guard_context
from .config import parse_args_with_config
from .manifest import build_manifest, save_manifest

from ..utils.fov import (
    compute_roi,
    grid_from_detector_fov_slices,
    grid_from_detector_fov,
)
from ..utils.fov import cylindrical_mask_xy


def _parse_views_per_batch(value: str) -> int | str:
    """Parse ``--views-per-batch`` as a positive/zero integer or ``auto``."""
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("--views-per-batch must be 'auto' or an integer")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("value must be a positive float")
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def _default_views_per_batch(algo: str) -> int:
    return 16 if str(algo).lower() == "spdhg" else 1


def _resolve_views_per_batch(
    requested: int | str | None,
    *,
    algo: str,
    n_views: int,
    grid: Grid,
    detector: Detector,
    gather_dtype: str,
    checkpoint_projector: bool,
) -> tuple[int, str]:
    """Resolve CLI batching after ROI/grid choices are known."""
    if requested is None:
        return _default_views_per_batch(algo), "default"

    if isinstance(requested, str) and requested.lower() == "auto":
        estimate = estimate_views_per_batch_info(
            n_views=int(n_views),
            grid_nxyz=(int(grid.nx), int(grid.ny), int(grid.nz)),
            det_nuv=(int(detector.nv), int(detector.nu)),
            gather_dtype=str(gather_dtype),
            projection_dtype="fp32",
            volume_dtype="fp32",
            checkpoint_projector=bool(checkpoint_projector),
            algo=str(algo),
            fallback_batch=1,
        )
        if estimate.fallback_used:
            logging.warning(
                "Could not determine available memory for --views-per-batch auto; "
                "using views_per_batch=%d",
                estimate.views_per_batch,
            )
        return int(estimate.views_per_batch), "auto"

    return max(1, int(requested)), "explicit"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconstruct volume from dataset (.nxs)")
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
    p.add_argument("--algo", choices=["fbp", "fista", "spdhg"], default="fbp")
    p.add_argument("--filter", default="ramp", help="FBP filter: ramp|shepp|hann")
    p.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Iterations for iterative algos (FISTA/SPDHG)",
    )
    p.add_argument(
        "--lambda-tv",
        type=float,
        default=0.005,
        help="TV regularization weight (FISTA/SPDHG)",
    )
    p.add_argument(
        "--regulariser",
        choices=["tv", "huber_tv"],
        default="tv",
        help="Regulariser for iterative algos: tv (default) or huber_tv",
    )
    p.add_argument(
        "--huber-delta",
        type=_positive_float,
        default=1e-2,
        help="Huber-TV transition radius for --regulariser huber_tv",
    )
    p.add_argument(
        "--tv-prox-iters",
        type=int,
        default=10,
        help="Inner iterations for TV proximal operator (FISTA)",
    )
    p.add_argument(
        "--L",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA (skip power-method)",
    )
    pos = p.add_mutually_exclusive_group()
    pos.add_argument(
        "--positivity",
        dest="positivity",
        action="store_true",
        help="Enable nonnegative projection for FISTA reconstructions",
    )
    pos.add_argument(
        "--no-positivity",
        dest="positivity",
        action="store_false",
        help="Disable nonnegative projection for FISTA reconstructions",
    )
    p.set_defaults(positivity=False)
    p.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="Optional lower voxel bound for FISTA reconstructions",
    )
    p.add_argument(
        "--upper-bound",
        type=float,
        default=None,
        help="Optional upper voxel bound for FISTA reconstructions",
    )
    # SPDHG-specific knobs
    p.add_argument(
        "--views-per-batch",
        type=_parse_views_per_batch,
        default=None,
        help=(
            "Views per projection batch, or 'auto' to estimate from available memory "
            "(default: 1 for FBP/FISTA, 16 for SPDHG)"
        ),
    )
    p.add_argument("--theta", type=float, default=1.0, help="SPDHG: extrapolation for xbar")
    p.add_argument("--spdhg-seed", type=int, default=0, help="SPDHG: RNG seed for block order")
    p.add_argument(
        "--spdhg-tau",
        type=float,
        default=None,
        help="SPDHG: override primal step size (auto if None)",
    )
    p.add_argument(
        "--spdhg-sigma-data",
        type=float,
        default=None,
        help="SPDHG: override data dual step (auto if None)",
    )
    p.add_argument(
        "--spdhg-sigma-tv",
        type=float,
        default=None,
        help="SPDHG: override TV dual step (auto if None)",
    )
    p.add_argument(
        "--warm-start",
        choices=["none", "fbp"],
        default="none",
        help="Initialize iterative algo from this method (spdhg only): none|fbp",
    )
    p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32; accumulation stays fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    ck.add_argument(
        "--checkpoint-projector",
        dest="checkpoint_projector",
        action="store_true",
        help="Enable projector checkpointing",
    )
    ck.add_argument(
        "--no-checkpoint-projector",
        dest="checkpoint_projector",
        action="store_false",
        help="Disable projector checkpointing",
    )
    p.set_defaults(checkpoint_projector=True)
    p.add_argument(
        "--out",
        help="Output .nxs containing recon (and copying projections)",
    )
    p.add_argument(
        "--quicklook",
        "--save-preview",
        dest="quicklook",
        metavar="PATH",
        default=None,
        help="Write a percentile-scaled central xy slice PNG preview to PATH.",
    )
    p.add_argument(
        "--save-manifest",
        metavar="PATH",
        default=None,
        help="Write a JSON reproducibility manifest for this reconstruction run.",
    )
    p.add_argument(
        "--roi",
        choices=["off", "auto", "cube", "bbox"],
        default="auto",
        help=(
            "Optional ROI cropping based on detector FOV (default: auto). "
            "auto: square x–y slices + z from detector height if detector < grid; "
            "cube: force cubic ROI (nx=ny=nz) inside FOV; "
            "bbox: rectangular FOV bbox; off: keep original grid"
        ),
    )
    p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Override reconstruction grid size (nx ny nz). Voxel sizes stay as in input metadata.",
    )
    p.add_argument(
        "--frame",
        choices=["sample", "lab"],
        default="sample",
        help="Frame to record for saved volume (default: sample).",
    )
    p.add_argument(
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer compatibility).",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
    )
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
    )
    p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume during forward projection (FISTA) or on output (FBP): "
            "off (default), cyl for cylindrical x–y mask broadcast along z."
        ),
    )
    return p


def main() -> None:
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = args.grid if (meta.grid is None and args.grid is not None) else None
    grid, detector, geom = build_geometry_from_meta(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    proj = jnp.asarray(meta.projections, dtype=jnp.float32)

    from ..utils.memory import default_gather_dtype as _default_gather_dtype

    _gather = str(args.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    # Optional ROI selection
    recon_grid = grid
    roi_mode = str(args.roi).lower()
    is_parallel = meta.geometry_type == "parallel"
    if roi_mode != "off":
        try:
            info = compute_roi(grid, detector, crop_y_to_u=is_parallel)
            # Only crop when detector FOV is smaller than current grid (auto)
            full_half_x = ((grid.nx / 2.0) - 0.5) * float(grid.vx)
            full_half_y = ((grid.ny / 2.0) - 0.5) * float(grid.vy)
            full_half_z = ((grid.nz / 2.0) - 0.5) * float(grid.vz)
            det_smaller = (
                (info.r_u + 1e-6) < full_half_x
                or (is_parallel and (info.r_u + 1e-6) < full_half_y)
                or (info.r_v + 1e-6) < full_half_z
            )
            if roi_mode == "auto" and det_smaller:
                if is_parallel:
                    recon_grid = grid_from_detector_fov_slices(grid, detector, crop_y_to_u=True)
                else:
                    recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
            elif roi_mode == "cube":
                # Same as align default policy for cubic volumes
                from ..utils.fov import grid_from_detector_fov_cube as _grid_cube

                recon_grid = _grid_cube(grid, detector, crop_y_to_u=is_parallel)
            elif roi_mode == "bbox":
                recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
        except Exception:
            # Fall back silently if FOV computation fails
            recon_grid = grid

    # Rebuild geometry if grid changed
    if args.grid is not None:
        NX, NY, NZ = map(int, args.grid)
        recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)

    if recon_grid is not grid:
        # Once ROI and explicit sizing resolve an effective grid, keep that grid's
        # origin/centre metadata authoritative when rebuilding geometry.
        _, _, geom = build_geometry_from_meta(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    # Prepare optional volume mask
    vol_mask = None
    try:
        if str(args.mask_vol).lower() in ("cyl", "cylindrical"):
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            vol_mask = jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception:
        vol_mask = None

    resolved_vpb, vpb_mode = _resolve_views_per_batch(
        args.views_per_batch,
        algo=str(args.algo),
        n_views=int(proj.shape[0]),
        grid=recon_grid,
        detector=detector,
        gather_dtype=_gather,
        checkpoint_projector=bool(args.checkpoint_projector),
    )
    logging.info(
        "Reconstruction views_per_batch=%d (mode=%s, algo=%s)",
        resolved_vpb,
        vpb_mode,
        args.algo,
    )

    algorithm_config: dict[str, object]
    if args.algo == "fbp":
        algorithm_config = {
            "filter": str(args.filter),
            "views_per_batch": int(resolved_vpb),
            "projector_unroll": 1,
            "checkpoint_projector": bool(args.checkpoint_projector),
            "gather_dtype": _gather,
        }
        with transfer_guard_context(args.transfer_guard):
            vol = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                filter_name=args.filter,
                views_per_batch=resolved_vpb,
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=_gather,
            )
        # For FBP, apply mask post-hoc if requested for parity
        if vol_mask is not None:
            vol = vol * vol_mask
    elif args.algo == "fista":
        cfg = FistaConfig(
            iters=int(args.iters),
            lambda_tv=float(args.lambda_tv),
            regulariser=str(args.regulariser),
            huber_delta=float(args.huber_delta),
            L=(float(args.L) if args.L is not None else None),
            views_per_batch=resolved_vpb,
            projector_unroll=1,
            checkpoint_projector=bool(args.checkpoint_projector),
            gather_dtype=_gather,
            tv_prox_iters=int(args.tv_prox_iters),
            support=vol_mask,
            positivity=bool(args.positivity),
            lower_bound=(
                float(args.lower_bound) if args.lower_bound is not None else None
            ),
            upper_bound=(
                float(args.upper_bound) if args.upper_bound is not None else None
            ),
        )
        algorithm_config = {
            "iters": int(cfg.iters),
            "lambda_tv": float(cfg.lambda_tv),
            "regulariser": str(cfg.regulariser),
            "huber_delta": float(cfg.huber_delta),
            "L": cfg.L,
            "views_per_batch": int(resolved_vpb),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
            "grad_mode": str(cfg.grad_mode),
            "tv_prox_iters": int(cfg.tv_prox_iters),
            "recon_rel_tol": cfg.recon_rel_tol,
            "recon_patience": int(cfg.recon_patience),
            "power_iters": int(cfg.power_iters),
            "support": "cylindrical" if vol_mask is not None else None,
            "positivity": bool(cfg.positivity),
            "lower_bound": cfg.lower_bound,
            "upper_bound": cfg.upper_bound,
        }
        with transfer_guard_context(args.transfer_guard):
            vol, info = fista_tv(
                geom,
                recon_grid,
                detector,
                proj,
                config=cfg,
            )
    else:  # spdhg
        # Build SPDHG config
        cfg = SPDHGConfig(
            iters=int(args.iters),
            lambda_tv=float(args.lambda_tv),
            regulariser=str(args.regulariser),
            huber_delta=float(args.huber_delta),
            theta=float(args.theta),
            views_per_batch=int(resolved_vpb),
            seed=int(args.spdhg_seed),
            tau=(float(args.spdhg_tau) if args.spdhg_tau is not None else None),
            sigma_data=(
                float(args.spdhg_sigma_data) if args.spdhg_sigma_data is not None else None
            ),
            sigma_tv=(float(args.spdhg_sigma_tv) if args.spdhg_sigma_tv is not None else None),
            projector_unroll=1,
            checkpoint_projector=bool(args.checkpoint_projector),
            gather_dtype=_gather,
            positivity=True,
            support=vol_mask if vol_mask is not None else None,
            log_every=1,
        )
        algorithm_config = {
            "iters": int(cfg.iters),
            "lambda_tv": float(cfg.lambda_tv),
            "regulariser": str(cfg.regulariser),
            "huber_delta": float(cfg.huber_delta),
            "theta": float(cfg.theta),
            "views_per_batch": int(cfg.views_per_batch),
            "seed": int(cfg.seed),
            "tau": cfg.tau,
            "sigma_data": cfg.sigma_data,
            "sigma_tv": cfg.sigma_tv,
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
            "positivity": bool(cfg.positivity),
            "support": "cylindrical" if vol_mask is not None else None,
            "log_every": int(cfg.log_every),
            "warm_start": str(args.warm_start),
        }
        # Optional warm-start for SPDHG
        init_x = None
        if str(args.warm_start).lower() == "fbp":
            # Compute a quick FBP and use as initialization
            warm_start_vpb = 1 if vpb_mode == "default" else int(resolved_vpb)
            init_x = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                filter_name=str(args.filter),
                views_per_batch=warm_start_vpb,
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=_gather,
            )
            if vol_mask is not None:
                init_x = init_x * vol_mask
            # Enforce positivity for a clean start (harmless if TV/signal expects nonnegative)
            init_x = jnp.maximum(init_x, 0.0)
        with transfer_guard_context(args.transfer_guard):
            vol, info = spdhg_tv(
                geom,
                recon_grid,
                detector,
                proj,
                init_x=init_x,
                config=cfg,
            )

    # Save the reconstruction in the object (sample) frame.
    # Reuse host projections from metadata to avoid a device-to-host copy.
    volume_np = np.asarray(vol)
    save_meta = meta.copy_metadata()
    save_meta.grid = recon_grid.to_dict()
    save_meta.volume = volume_np
    save_meta.frame = str(args.frame)
    save_meta.volume_axes_order = str(args.volume_axes)
    save_nxtomo(
        args.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved reconstruction to %s", args.out)
    if args.quicklook is not None:
        save_quicklook_png(args.quicklook, volume_np)
        logging.info("Saved reconstruction quicklook to %s", args.quicklook)
    if args.save_manifest is not None:
        manifest = build_manifest(
            "tomojax-recon",
            list(sys.argv),
            args,
            {
                "input_path": args.data,
                "output_path": args.out,
                "quicklook_path": args.quicklook,
                "manifest_path": args.save_manifest,
                "config_path": config_metadata["config_path"],
                "config_file_values": config_metadata["config_file_values"],
                "explicit_cli_keys": config_metadata["explicit_cli_keys"],
                "effective_options": config_metadata["effective_options"],
                "algorithm": str(args.algo),
                "algorithm_config": algorithm_config,
                "geometry_type": str(meta.geometry_type),
                "input_projection_shape": list(meta.projections.shape),
                "reconstruction_grid": recon_grid.to_dict(),
                "detector": detector.to_dict(),
                "roi": {
                    "requested": roi_mode,
                    "is_parallel": bool(is_parallel),
                    "grid_changed": recon_grid != grid,
                },
                "requested_views_per_batch": args.views_per_batch,
                "views_per_batch": int(resolved_vpb),
                "views_per_batch_mode": vpb_mode,
                "requested_gather_dtype": str(args.gather_dtype),
                "gather_dtype": _gather,
                "checkpoint_projector": bool(args.checkpoint_projector),
                "transfer_guard": str(args.transfer_guard),
                "mask_vol": str(args.mask_vol),
                "mask_applied": vol_mask is not None,
                "volume_shape": list(volume_np.shape),
                "volume_axes": str(args.volume_axes),
                "frame": str(args.frame),
            },
        )
        save_manifest(args.save_manifest, manifest)
        logging.info("Saved reproducibility manifest to %s", args.save_manifest)


if __name__ == "__main__":  # pragma: no cover
    main()
