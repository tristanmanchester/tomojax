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
from ..align.dofs import active_dof_mask, normalize_dofs
from ..align.losses import parse_loss_spec
from ..align.params_export import save_alignment_params_csv, save_alignment_params_json
from ..core.geometry import Grid, Detector
from ..align.pipeline import align, AlignConfig
from ..utils.logging import setup_logging, log_jax_env
from ..utils.axes import DISK_VOLUME_AXES
from ._runtime import transfer_guard_context
from .config import parse_args_with_config
from .manifest import build_manifest, save_manifest

from ..utils.fov import (
    compute_roi,
    grid_from_detector_fov_slices,
    grid_from_detector_fov,
    cylindrical_mask_xy,
)


def _init_jax_compilation_cache() -> None:
    """Enable JAX persistent compilation cache for faster re-runs.

    Directory precedence:
    - TOMOJAX_JAX_CACHE_DIR if set
    - ${XDG_CACHE_HOME:-~/.cache}/tomojax/jax_cache
    """
    try:
        # Avoid noisy logs on environments without this feature
        from jax.experimental import compilation_cache as cc  # type: ignore
    except Exception:
        return
    try:
        cache_dir = os.environ.get("TOMOJAX_JAX_CACHE_DIR")
        if not cache_dir:
            base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
            cache_dir = os.path.join(base, "tomojax", "jax_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cc.initialize_cache(cache_dir)
        logging.info("JAX compilation cache: %s", cache_dir)
    except Exception:
        # Best-effort; skip on any failure silently
        pass


def _resolve_recon_grid_and_mask(
    grid: Grid,
    detector: Detector,
    *,
    is_parallel: bool = True,
    roi_mode: str,
    grid_override: tuple[int, int, int] | list[int] | None,
) -> tuple[Grid, bool]:
    try:
        roi = compute_roi(grid, detector, crop_y_to_u=is_parallel)
        full_half_x = ((grid.nx / 2.0) - 0.5) * float(grid.vx)
        full_half_y = ((grid.ny / 2.0) - 0.5) * float(grid.vy)
        full_half_z = ((grid.nz / 2.0) - 0.5) * float(grid.vz)
        det_smaller = (
            (roi.r_u + 1e-6) < full_half_x
            or (is_parallel and (roi.r_u + 1e-6) < full_half_y)
            or (roi.r_v + 1e-6) < full_half_z
        )
    except Exception:
        det_smaller = False

    recon_grid = grid
    apply_cyl_mask = False
    if roi_mode == "cube" or (roi_mode == "auto" and det_smaller):
        if roi_mode == "auto" and not is_parallel:
            recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
        else:
            recon_grid = grid_from_detector_fov_slices(grid, detector, crop_y_to_u=is_parallel)
    elif roi_mode == "bbox":
        recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
    elif roi_mode == "cyl":
        recon_grid = grid_from_detector_fov_slices(grid, detector, crop_y_to_u=is_parallel)
        apply_cyl_mask = True

    # Explicit grid overrides take full precedence over ROI-derived masking.
    if grid_override is not None:
        NX, NY, NZ = map(int, grid_override)
        recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)
        apply_cyl_mask = False

    return recon_grid, apply_cyl_mask


def _parse_dof_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    try:
        optimise_dofs = (
            None
            if args.optimise_dofs is None
            else normalize_dofs(args.optimise_dofs, option_name="--optimise-dofs")
        )
        freeze_dofs = normalize_dofs(args.freeze_dofs, option_name="--freeze-dofs")
        active_dof_mask(optimise_dofs=optimise_dofs, freeze_dofs=freeze_dofs)
    except ValueError as exc:
        parser.error(str(exc))
    return optimise_dofs, freeze_dofs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint reconstruction + alignment on dataset (.nxs)")
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
    p.add_argument("--outer-iters", type=int, default=5)
    p.add_argument("--recon-iters", type=int, default=10)
    p.add_argument("--lambda-tv", type=float, default=0.005)
    p.add_argument(
        "--tv-prox-iters",
        type=int,
        default=10,
        help="Inner iterations for TV proximal operator",
    )
    p.add_argument("--lr-rot", type=float, default=1e-3)
    p.add_argument("--lr-trans", type=float, default=1e-1)
    p.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="Optional multires factors, e.g., 4 2 1",
    )
    p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype (auto: bf16 on GPU/TPU, else fp32)",
    )
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--checkpoint-projector", dest="checkpoint_projector", action="store_true")
    ck.add_argument("--no-checkpoint-projector", dest="checkpoint_projector", action="store_false")
    p.set_defaults(checkpoint_projector=True)
    p.add_argument(
        "--opt-method",
        choices=["gd", "gn"],
        default="gn",
        help="Alignment optimizer: gd or gn (GN supported for L2-like losses: l2, l2_otsu, edge_l2, pwls)",
    )
    p.add_argument(
        "--gn-damping",
        type=float,
        default=1e-3,
        help="Levenberg-Marquardt damping for GN",
    )
    p.add_argument(
        "--optimise-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to optimise: alpha,beta,phi,dx,dz. Example: dx,dz",
    )
    p.add_argument(
        "--freeze-dofs",
        nargs="+",
        default=None,
        metavar="DOF[,DOF]",
        help="Named alignment DOFs to keep fixed at initial values. Example: phi",
    )
    p.add_argument("--w-rot", type=float, default=1e-3, help="Smoothness weight for rotations")
    p.add_argument("--w-trans", type=float, default=1e-3, help="Smoothness weight for translations")
    p.add_argument(
        "--seed-translations",
        action="store_true",
        help="Phase-correlation init for dx,dz at coarsest level",
    )
    p.add_argument(
        "--log-summary",
        action="store_true",
        help="Print per-outer summaries (FISTA loss, alignment loss before/after)",
    )
    p.add_argument(
        "--log-compact",
        dest="log_compact",
        action="store_true",
        default=True,
        help="Use compact one-line per-outer summary when --log-summary is set (default: on)",
    )
    p.add_argument("--no-log-compact", dest="log_compact", action="store_false")
    p.add_argument(
        "--recon-L",
        type=float,
        default=None,
        help="Fixed Lipschitz constant for FISTA inside alignment (skip power-method)",
    )
    # Data term / similarity
    p.add_argument(
        "--loss",
        choices=[
            "l2",
            "charbonnier",
            "huber",
            "cauchy",
            "lorentzian",
            "welsch",
            "leclerc",
            "barron",
            "student_t",
            "correntropy",
            "zncc",
            "ssim",
            "ms-ssim",
            "mi",
            "nmi",
            "renyi_mi",
            "grad_l1",
            "edge_l2",
            "ngf",
            "grad_orient",
            "phasecorr",
            "fft_mag",
            "chamfer_edge",
            "l2_otsu",
            "ssim_otsu",
            "tversky",
            "swd",
            "mind",
            "pwls",
            "poisson",
        ],
        default="l2_otsu",
        help="Data term / similarity to optimize (default: l2_otsu)",
    )
    p.add_argument(
        "--loss-param",
        action="append",
        default=[],
        help="Loss parameter as k=v (repeatable), e.g., delta=1.0, eps=1e-3, window=7, temp=0.5",
    )
    # Early stopping controls (alignment phase)
    es = p.add_mutually_exclusive_group()
    es.add_argument(
        "--early-stop",
        dest="early_stop",
        action="store_true",
        help="Enable early stopping across outers (default)",
    )
    es.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable early stopping across outers",
    )
    p.set_defaults(early_stop=True)
    p.add_argument(
        "--early-stop-rel",
        type=float,
        default=None,
        help="Relative improvement threshold for early stop (default 1e-3)",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Consecutive outers below threshold before stopping (default 2)",
    )
    p.add_argument(
        "--transfer-guard",
        choices=["off", "log", "disallow"],
        default=os.environ.get("TOMOJAX_TRANSFER_GUARD", "off"),
        help="JAX transfer guard mode during compute (default: off; use log/disallow when debugging)",
    )
    p.add_argument("--out", help="Output .nxs with recon and alignment params")
    p.add_argument(
        "--save-params-json",
        default=None,
        help="Optional JSON sidecar for final per-view alignment parameters",
    )
    p.add_argument(
        "--save-params-csv",
        default=None,
        help="Optional CSV sidecar for final per-view alignment parameters",
    )
    p.add_argument(
        "--save-manifest",
        metavar="PATH",
        default=None,
        help="Write a JSON reproducibility manifest for this alignment run.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars if tqdm is available",
    )
    p.add_argument(
        "--roi",
        choices=["auto", "off", "cube", "bbox", "cyl"],
        default="auto",
        help=(
            "Region to reconstruct: auto: square x–y slices + z from detector height; "
            "off: use full grid; cube: same as auto; bbox: rectangular FOV bbox; "
            "cyl: auto + zero outside cylindrical FOV"
        ),
    )
    p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume before forward projection in alignment: "
            "off (default), or cyl for cylindrical x–y mask broadcast along z."
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
        "--volume-axes",
        choices=["zyx", "xyz"],
        default=DISK_VOLUME_AXES,
        help="On-disk axis order for saved volumes (default: zyx for viewer compatibility).",
    )
    return p


def main() -> None:
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, required=("data", "out"))

    setup_logging()
    log_jax_env()
    _init_jax_compilation_cache()
    if args.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"
    # Parse loss params (k=v -> float)
    loss_params: dict[str, float] = {}
    for kv in args.loss_param:
        if "=" not in kv:
            raise SystemExit(f"--loss-param must be k=v, got: {kv}")
        k, v = kv.split("=", 1)
        try:
            loss_params[k.strip()] = float(v)
        except ValueError:
            raise SystemExit(f"--loss-param value must be numeric: {kv}")
    loss_spec = parse_loss_spec(str(args.loss), loss_params if loss_params else None)
    optimise_dofs, freeze_dofs = _parse_dof_args(args, p)

    meta = load_nxtomo(args.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = args.grid if (meta.grid is None and args.grid is not None) else None
    grid, detector, geom = build_geometry_from_meta(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=False,
    )
    proj = jnp.asarray(meta.projections, dtype=jnp.float32)

    # Hidden defaults: stream one view at a time; unroll=1
    vpb_est = 1

    # Resolve default gather dtype lazily at runtime
    from ..utils.memory import default_gather_dtype as _default_gather_dtype

    _gather = str(args.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    cfg = AlignConfig(
        outer_iters=args.outer_iters,
        recon_iters=args.recon_iters,
        lambda_tv=args.lambda_tv,
        tv_prox_iters=int(args.tv_prox_iters),
        lr_rot=args.lr_rot,
        lr_trans=args.lr_trans,
        views_per_batch=int(vpb_est),
        projector_unroll=1,
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=_gather,
        opt_method=str(args.opt_method),
        gn_damping=float(args.gn_damping),
        w_rot=float(args.w_rot),
        w_trans=float(args.w_trans),
        optimise_dofs=optimise_dofs,
        freeze_dofs=freeze_dofs,
        loss=loss_spec,
        seed_translations=bool(args.seed_translations),
        log_summary=bool(args.log_summary),
        log_compact=bool(args.log_compact),
        recon_L=(float(args.recon_L) if args.recon_L is not None else None),
        early_stop=bool(args.early_stop),
        early_stop_rel_impr=(
            float(args.early_stop_rel) if args.early_stop_rel is not None else 1e-3
        ),
        early_stop_patience=(
            int(args.early_stop_patience) if args.early_stop_patience is not None else 2
        ),
        mask_vol=str(args.mask_vol),
    )
    # ROI handling (align on realistic FOV)
    recon_grid, apply_cyl_mask = _resolve_recon_grid_and_mask(
        grid,
        detector,
        is_parallel=meta.geometry_type == "parallel",
        roi_mode=str(args.roi).lower(),
        grid_override=args.grid,
    )

    # Rebuild geometry if grid changed
    if recon_grid is not grid:
        # Once ROI and explicit sizing resolve an effective grid, keep that grid's
        # origin/centre metadata authoritative when rebuilding geometry.
        _, _, geom = build_geometry_from_meta(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=False,
        )

    if args.levels is not None and len(args.levels) > 0:
        from ..align.pipeline import align_multires

        with transfer_guard_context(args.transfer_guard):
            x, params5, info = align_multires(
                geom, recon_grid, detector, proj, factors=args.levels, cfg=cfg
            )
    else:
        with transfer_guard_context(args.transfer_guard):
            x, params5, info = align(geom, recon_grid, detector, proj, cfg=cfg)

    # Optional cylindrical mask in x–y
    if apply_cyl_mask:
        import numpy as _np

        try:
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            m = jnp.asarray(m_xy, dtype=x.dtype)[:, :, None]
            x = x * m
        except Exception:
            m_xy = cylindrical_mask_xy(recon_grid, detector)
            m = _np.asarray(m_xy, dtype=_np.float32)[:, :, None]
            x = jnp.asarray(_np.asarray(x) * m)

    # Avoid copying projections back from device: reuse host array from metadata
    params5_np = np.asarray(params5)
    save_meta = meta.copy_metadata()
    save_meta.grid = recon_grid.to_dict()
    save_meta.volume = np.asarray(x)
    save_meta.align_params = params5_np
    save_meta.frame = str(meta.frame or "sample")
    save_meta.volume_axes_order = str(args.volume_axes)
    save_nxtomo(
        args.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved alignment results to %s", args.out)
    if args.save_params_json is not None:
        save_alignment_params_json(
            args.save_params_json,
            params5_np,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        logging.info("Saved alignment parameter JSON to %s", args.save_params_json)
    if args.save_params_csv is not None:
        save_alignment_params_csv(
            args.save_params_csv,
            params5_np,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        logging.info("Saved alignment parameter CSV to %s", args.save_params_csv)
    if args.save_manifest is not None:
        loss_values = info.get("loss", []) if isinstance(info, dict) else []
        manifest = build_manifest(
            "tomojax-align",
            list(sys.argv),
            args,
            {
                "input_path": args.data,
                "output_path": args.out,
                "save_params_json": args.save_params_json,
                "save_params_csv": args.save_params_csv,
                "manifest_path": args.save_manifest,
                "config_path": config_metadata["config_path"],
                "config_file_values": config_metadata["config_file_values"],
                "explicit_cli_keys": config_metadata["explicit_cli_keys"],
                "effective_options": config_metadata["effective_options"],
                "geometry_type": str(meta.geometry_type),
                "input_projection_shape": list(meta.projections.shape),
                "reconstruction_grid": recon_grid.to_dict(),
                "detector": detector.to_dict(),
                "roi": {
                    "requested": str(args.roi),
                    "is_parallel": bool(meta.geometry_type == "parallel"),
                    "grid_changed": recon_grid != grid,
                    "cylindrical_output_mask": bool(apply_cyl_mask),
                },
                "requested_gather_dtype": str(args.gather_dtype),
                "gather_dtype": _gather,
                "views_per_batch": int(vpb_est),
                "projector_unroll": 1,
                "checkpoint_projector": bool(args.checkpoint_projector),
                "transfer_guard": str(args.transfer_guard),
                "levels": args.levels,
                "used_multires": bool(args.levels is not None and len(args.levels) > 0),
                "loss_params": loss_params,
                "loss_spec": loss_spec,
                "align_config": cfg,
                "alignment_params_shape": list(params5_np.shape),
                "volume_shape": list(np.asarray(x).shape),
                "volume_axes": str(args.volume_axes),
                "frame": str(save_meta.frame),
                "run_info": {
                    "loss_count": len(loss_values),
                    "final_loss": loss_values[-1] if len(loss_values) else None,
                    "stopped_by_observer": (
                        info.get("stopped_by_observer") if isinstance(info, dict) else None
                    ),
                    "observer_action": (
                        info.get("observer_action") if isinstance(info, dict) else None
                    ),
                },
            },
        )
        save_manifest(args.save_manifest, manifest)
        logging.info("Saved reproducibility manifest to %s", args.save_manifest)


if __name__ == "__main__":  # pragma: no cover
    main()
