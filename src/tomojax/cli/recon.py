from __future__ import annotations

import argparse
import logging
import numpy as np
import jax.numpy as jnp
import os
from contextlib import nullcontext as _nullcontext

from ..data.io_hdf5 import load_nxtomo, save_nxtomo
from ..core.geometry import Grid, Detector, ParallelGeometry, LaminographyGeometry
from ..recon.fbp import fbp
from ..recon.fista_tv import fista_tv
from ..recon.spdhg_tv import spdhg_tv, SPDHGConfig
from ..utils.logging import setup_logging, log_jax_env
from ..utils.axes import DISK_VOLUME_AXES
from ..utils.fov import (
    compute_roi,
    grid_from_detector_fov_slices,
    grid_from_detector_fov,
)
from ..utils.fov import cylindrical_mask_xy


def build_geometry(meta: dict):
    grid_d = meta["grid"]
    det_d = meta["detector"]
    thetas = meta["thetas_deg"]
    gtype = meta.get("geometry_type", "parallel")
    grid = Grid(**{k: grid_d[k] for k in ("nx", "ny", "nz", "vx", "vy", "vz")})
    detector = Detector(
        **{k: det_d[k] for k in ("nu", "nv", "du", "dv")},
        det_center=tuple(det_d.get("det_center", (0.0, 0.0))),
    )
    if gtype == "parallel":
        geom = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    else:
        tilt_deg = float(meta.get("tilt_deg", 30.0))
        tilt_about = str(meta.get("tilt_about", "x"))
        geom = LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=tilt_deg,
            tilt_about=tilt_about,
        )
    return grid, detector, geom


def _transfer_guard_ctx(mode: str | None = None):
    # Allow overriding via env var: off|log|disallow
    if mode is None:
        mode = os.environ.get("TOMOJAX_TRANSFER_GUARD", "log").lower()
    if mode in ("off", "none", "disable", "disabled"):
        return _nullcontext()
    try:
        import jax as _jax  # local import for flexibility

        tg = getattr(_jax, "transfer_guard", None)
        if tg is not None:
            return tg(mode)
        try:
            from jax.experimental import transfer_guard as _tg  # type: ignore

            return _tg(mode)
        except Exception:
            return _nullcontext()
    except Exception:
        return _nullcontext()


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruct volume from dataset (.nxs)")
    p.add_argument("--data", required=True, help="Input .nxs")
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
    # SPDHG-specific knobs
    p.add_argument(
        "--views-per-batch",
        type=int,
        default=16,
        help="SPDHG: views per stochastic block",
    )
    p.add_argument(
        "--theta", type=float, default=1.0, help="SPDHG: extrapolation for xbar"
    )
    p.add_argument(
        "--spdhg-seed", type=int, default=0, help="SPDHG: RNG seed for block order"
    )
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
        required=True,
        help="Output .nxs containing recon (and copying projections)",
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
    args = p.parse_args()

    setup_logging()
    log_jax_env()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_nxtomo(args.data)
    grid, detector, geom = build_geometry(meta)
    proj = jnp.asarray(meta["projections"], dtype=jnp.float32)

    # Hidden defaults: stream one view at a time; unroll=1
    vpb_val: int | None = 1

    from ..utils.memory import default_gather_dtype as _default_gather_dtype

    _gather = str(args.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    # Optional ROI selection
    recon_grid = grid
    roi_mode = str(args.roi).lower()
    is_parallel = meta.get("geometry_type", "parallel") == "parallel"
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
                    recon_grid = grid_from_detector_fov_slices(
                        grid, detector, crop_y_to_u=True
                    )
                else:
                    recon_grid = grid_from_detector_fov(
                        grid, detector, crop_y_to_u=False
                    )
            elif roi_mode == "cube":
                # Same as align default policy for cubic volumes
                from ..utils.fov import grid_from_detector_fov_cube as _grid_cube

                recon_grid = _grid_cube(grid, detector, crop_y_to_u=is_parallel)
            elif roi_mode == "bbox":
                recon_grid = grid_from_detector_fov(
                    grid, detector, crop_y_to_u=is_parallel
                )
        except Exception:
            # Fall back silently if FOV computation fails
            recon_grid = grid

    # Rebuild geometry if grid changed
    if args.grid is not None:
        NX, NY, NZ = map(int, args.grid)
        recon_grid = Grid(nx=NX, ny=NY, nz=NZ, vx=grid.vx, vy=grid.vy, vz=grid.vz)

    if recon_grid is not grid:
        if meta.get("geometry_type", "parallel") == "parallel":
            geom = ParallelGeometry(
                grid=recon_grid, detector=detector, thetas_deg=meta["thetas_deg"]
            )
        else:
            tilt_deg = float(meta.get("tilt_deg", 30.0))
            tilt_about = str(meta.get("tilt_about", "x"))
            geom = LaminographyGeometry(
                grid=recon_grid,
                detector=detector,
                thetas_deg=meta["thetas_deg"],
                tilt_deg=tilt_deg,
                tilt_about=tilt_about,
            )

    # Prepare optional volume mask
    vol_mask = None
    try:
        if str(args.mask_vol).lower() in ("cyl", "cylindrical"):
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            vol_mask = jnp.asarray(m_xy, dtype=jnp.float32)[:, :, None]
    except Exception:
        vol_mask = None

    if args.algo == "fbp":
        with _transfer_guard_ctx(args.transfer_guard):
            vol = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                filter_name=args.filter,
                views_per_batch=int(vpb_val),
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=_gather,
            )
        # For FBP, apply mask post-hoc if requested for parity
        if vol_mask is not None:
            vol = vol * vol_mask
    elif args.algo == "fista":
        vpb = int(vpb_val) if int(vpb_val) > 0 else None
        with _transfer_guard_ctx(args.transfer_guard):
            vol, info = fista_tv(
                geom,
                recon_grid,
                detector,
                proj,
                iters=args.iters,
                lambda_tv=args.lambda_tv,
                L=(float(args.L) if args.L is not None else None),
                views_per_batch=vpb,
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=_gather,
                tv_prox_iters=int(args.tv_prox_iters),
                vol_mask=vol_mask,
            )
    else:  # spdhg
        # Build SPDHG config
        cfg = SPDHGConfig(
            iters=int(args.iters),
            lambda_tv=float(args.lambda_tv),
            theta=float(args.theta),
            views_per_batch=int(max(1, args.views_per_batch)),
            seed=int(args.spdhg_seed),
            tau=(float(args.spdhg_tau) if args.spdhg_tau is not None else None),
            sigma_data=(
                float(args.spdhg_sigma_data)
                if args.spdhg_sigma_data is not None
                else None
            ),
            sigma_tv=(
                float(args.spdhg_sigma_tv) if args.spdhg_sigma_tv is not None else None
            ),
            projector_unroll=1,
            checkpoint_projector=bool(args.checkpoint_projector),
            gather_dtype=_gather,
            positivity=True,
            support=vol_mask if vol_mask is not None else None,
            log_every=1,
        )
        # Optional warm-start for SPDHG
        init_x = None
        if str(args.warm_start).lower() == "fbp":
            # Compute a quick FBP and use as initialization
            init_x = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                filter_name=str(args.filter),
                views_per_batch=int(vpb_val),
                projector_unroll=1,
                checkpoint_projector=bool(args.checkpoint_projector),
                gather_dtype=_gather,
            )
            if vol_mask is not None:
                init_x = init_x * vol_mask
            # Enforce positivity for a clean start (harmless if TV/signal expects nonnegative)
            init_x = jnp.maximum(init_x, 0.0)
        with _transfer_guard_ctx(args.transfer_guard):
            vol, info = spdhg_tv(
                geom,
                recon_grid,
                detector,
                proj,
                init_x=init_x,
                config=cfg,
            )

    # Note: Reconstructions are computed on the object (sample) frame. We persist that by default.
    # If a lab-frame export is desired, we currently only record metadata; a resampling export
    # may be added later if needed for interop.
    # Avoid copying projections back from device: reuse host array from metadata
    save_nxtomo(
        args.out,
        projections=meta["projections"],
        thetas_deg=np.asarray(meta["thetas_deg"]),
        image_key=meta.get("image_key"),
        grid=recon_grid.to_dict(),
        detector=meta.get("detector"),
        geometry_type=meta.get("geometry_type", "parallel"),
        geometry_meta=meta.get("geometry_meta"),
        volume=np.asarray(vol),
        angle_offset_deg=meta.get("angle_offset_deg"),
        misalign_spec=meta.get("misalign_spec"),
        frame=str(args.frame),
        sample_name=meta.get("sample_name"),
        source_name=meta.get("source_name"),
        source_type=meta.get("source_type"),
        source_probe=meta.get("source_probe"),
        volume_axes_order=str(args.volume_axes),
    )
    logging.info("Saved reconstruction to %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
