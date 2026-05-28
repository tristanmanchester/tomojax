# ruff: noqa: E402
"""Run reconstruction workflows from the public TomoJAX CLI."""

# ruff: noqa: E402
# pyright: reportUnusedCallResult=false

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import logging
import os
import sys
from typing import TYPE_CHECKING, Literal, cast

from tomojax.cli._jax_allocator import configure_jax_allocator_defaults

configure_jax_allocator_defaults()

import jax.numpy as jnp
import numpy as np

from tomojax.backends import estimate_views_per_batch_info
from tomojax.cli._runtime import transfer_guard_context
from tomojax.cli.config import ConfigValue, parse_args_with_config
from tomojax.cli.manifest import build_manifest, save_manifest
from tomojax.core import log_jax_env, setup_logging
from tomojax.geometry import (
    DISK_VOLUME_AXES,
    Detector,
    Grid,
    compute_roi,
    cylindrical_mask_xy,
    detector_grid_from_geometry_inputs,
    grid_from_detector_fov,
    grid_from_detector_fov_slices,
)
from tomojax.io import (
    build_geometry_from_dataset_metadata,
    load_projection_payload,
    save_projection_payload,
)
from tomojax.recon.fbp import FBPConfig, fbp
from tomojax.recon.fista_tv import (
    FistaConfig,
    fista_tv,  # pyright: ignore[reportUnknownVariableType]
)
from tomojax.recon.quicklook import save_quicklook_png
from tomojax.recon.spdhg_tv import (
    SPDHGConfig,
    spdhg_tv,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tomojax.io import JsonValue
    from tomojax.recon.types import Regulariser

type ViewsPerBatch = int | Literal["auto"]
type ReconTransferGuardMode = Literal["off", "log", "disallow"]
type ReconAlgorithm = Literal["fbp", "fista", "spdhg"]
type ReconRoiMode = Literal["off", "auto", "cube", "bbox"]
type ReconMaskMode = Literal["off", "cyl"]
type ReconFrame = Literal["sample", "lab"]
type ReconVolumeAxes = Literal["zyx", "xyz"]


@dataclass(frozen=True)
class ReconCommand:
    """Typed command plan for the public reconstruction workflow."""

    config: str | None
    data: str
    out: str
    algo: ReconAlgorithm
    filter: str
    iters: int
    lambda_tv: float
    regulariser: str
    huber_delta: float
    tv_prox_iters: int
    lipschitz: float | None
    positivity: bool
    lower_bound: float | None
    upper_bound: float | None
    views_per_batch: ViewsPerBatch | None
    theta: float
    spdhg_seed: int
    spdhg_tau: float | None
    spdhg_sigma_data: float | None
    spdhg_sigma_tv: float | None
    warm_start: str
    gather_dtype: str
    checkpoint_projector: bool
    quicklook: str | None
    save_manifest: str | None
    roi: ReconRoiMode
    grid: tuple[int, int, int] | None
    frame: ReconFrame
    volume_axes: ReconVolumeAxes
    progress: bool
    transfer_guard: ReconTransferGuardMode
    mask_vol: ReconMaskMode
    apply_saved_alignment: bool
    det_u_px: float | None
    det_v_px: float | None


def _parse_views_per_batch(value: str) -> int | str:
    """Parse ``--views-per-batch`` as a positive/zero integer or ``auto``."""
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--views-per-batch must be 'auto' or an integer") from exc


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive float") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive float")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite float") from exc
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite float")
    return parsed


def _default_views_per_batch(algo: str) -> int:
    return 16 if str(algo).lower() == "spdhg" else 1


def _jnp_float32_array(value: object) -> jnp.ndarray:
    return jnp.asarray(value, dtype=np.float32)  # pyright: ignore[reportUnknownMemberType]


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


def _resolve_recon_grid_for_cli(  # noqa: PLR0911
    grid: Grid,
    detector: Detector,
    *,
    is_parallel: bool,
    roi_mode: str,
) -> Grid:
    if roi_mode == "off":
        return grid

    try:
        info = compute_roi(grid, detector, crop_y_to_u=is_parallel)
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
                return grid_from_detector_fov_slices(grid, detector, crop_y_to_u=True)
            return grid_from_detector_fov(grid, detector, crop_y_to_u=False)
        if roi_mode == "cube":
            from tomojax.geometry import grid_from_detector_fov_cube as _grid_cube

            return _grid_cube(grid, detector, crop_y_to_u=is_parallel)
        if roi_mode == "bbox":
            return grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
        return grid
    except Exception as exc:
        if roi_mode == "auto":
            logging.warning(
                "--roi=auto could not be applied; continuing without ROI crop: %s",
                exc,
                exc_info=True,
            )
            return grid
        raise ValueError(f"Failed to apply requested --roi={roi_mode!r}") from exc


def _resolve_volume_mask_for_cli(
    grid: Grid,
    detector: Detector,
    *,
    mask_vol: str,
) -> jnp.ndarray | None:
    mask_mode = str(mask_vol).lower()
    if mask_mode not in ("cyl", "cylindrical"):
        return None
    try:
        m_xy = cylindrical_mask_xy(grid, detector)
        return _jnp_float32_array(m_xy)[:, :, None]
    except Exception as exc:
        raise ValueError(f"Failed to apply requested --mask-vol={mask_mode!r}") from exc


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
            "auto: square x-y slices + z from detector height if detector < grid; "
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
        help="On-disk axis order for saved volumes (default: zyx for viewer convention).",
    )
    p.add_argument(
        "--det-u-px",
        type=_finite_float,
        default=None,
        help="Override detector centre u offset in detector pixels for COR sweeps.",
    )
    p.add_argument(
        "--det-v-px",
        type=_finite_float,
        default=None,
        help="Override detector centre v offset in detector pixels for COR sweeps.",
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
        help=(
            "JAX transfer guard mode during compute "
            "(default: off; use log/disallow for strict transfer checks)"
        ),
    )
    p.add_argument(
        "--mask-vol",
        choices=["off", "cyl"],
        default="off",
        help=(
            "Mask the volume during forward projection (FISTA) or on output (FBP): "
            "off (default), cyl for cylindrical x-y mask broadcast along z."
        ),
    )
    saved = p.add_mutually_exclusive_group()
    saved.add_argument(
        "--apply-saved-alignment",
        dest="apply_saved_alignment",
        action="store_true",
        help="Apply saved per-view alignment parameters from the input metadata.",
    )
    saved.add_argument(
        "--ignore-saved-alignment",
        dest="apply_saved_alignment",
        action="store_false",
        help="Ignore saved per-view alignment parameters from the input metadata (default).",
    )
    p.set_defaults(apply_saved_alignment=False)
    return p


def _parse_command(
    argv: Sequence[str] | None = None,
) -> tuple[ReconCommand, dict[str, ConfigValue]]:
    """Parse CLI/config defaults into a typed reconstruction command plan."""
    p = _build_parser()
    args, config_metadata = parse_args_with_config(p, argv, required=("data", "out"))
    grid = cast("list[int] | None", args.grid)
    return (
        ReconCommand(
            config=cast("str | None", args.config),
            data=cast("str", args.data),
            out=cast("str", args.out),
            algo=cast("ReconAlgorithm", args.algo),
            filter=cast("str", args.filter),
            iters=cast("int", args.iters),
            lambda_tv=cast("float", args.lambda_tv),
            regulariser=cast("str", args.regulariser),
            huber_delta=cast("float", args.huber_delta),
            tv_prox_iters=cast("int", args.tv_prox_iters),
            lipschitz=cast("float | None", args.L),
            positivity=cast("bool", args.positivity),
            lower_bound=cast("float | None", args.lower_bound),
            upper_bound=cast("float | None", args.upper_bound),
            views_per_batch=cast("ViewsPerBatch | None", args.views_per_batch),
            theta=cast("float", args.theta),
            spdhg_seed=cast("int", args.spdhg_seed),
            spdhg_tau=cast("float | None", args.spdhg_tau),
            spdhg_sigma_data=cast("float | None", args.spdhg_sigma_data),
            spdhg_sigma_tv=cast("float | None", args.spdhg_sigma_tv),
            warm_start=cast("str", args.warm_start),
            gather_dtype=cast("str", args.gather_dtype),
            checkpoint_projector=cast("bool", args.checkpoint_projector),
            quicklook=cast("str | None", args.quicklook),
            save_manifest=cast("str | None", args.save_manifest),
            roi=cast("ReconRoiMode", args.roi),
            grid=None if grid is None else (int(grid[0]), int(grid[1]), int(grid[2])),
            frame=cast("ReconFrame", args.frame),
            volume_axes=cast("ReconVolumeAxes", args.volume_axes),
            progress=cast("bool", args.progress),
            transfer_guard=cast("ReconTransferGuardMode", args.transfer_guard),
            mask_vol=cast("ReconMaskMode", args.mask_vol),
            apply_saved_alignment=cast("bool", args.apply_saved_alignment),
            det_u_px=cast("float | None", args.det_u_px),
            det_v_px=cast("float | None", args.det_v_px),
        ),
        config_metadata,
    )


def _apply_detector_center_override(
    detector: Detector,
    geometry_meta: dict[str, object],
    *,
    det_u_px: float | None,
    det_v_px: float | None,
) -> tuple[Detector, dict[str, JsonValue]]:
    """Apply CLI detector-centre overrides specified in detector pixels."""
    requested: dict[str, JsonValue] = {"det_u_px": det_u_px, "det_v_px": det_v_px}
    if det_u_px is None and det_v_px is None:
        u_px = float(detector.det_center[0]) / float(detector.du)
        v_px = float(detector.det_center[1]) / float(detector.dv)
        metadata_override: dict[str, JsonValue] = {
            "source": "metadata",
            "requested_px": requested,
            "effective_px": {"det_u_px": u_px, "det_v_px": v_px},
            "effective_world": {
                "det_u": float(detector.det_center[0]),
                "det_v": float(detector.det_center[1]),
            },
        }
        return detector, metadata_override
    u_px = float(detector.det_center[0]) / float(detector.du)
    v_px = float(detector.det_center[1]) / float(detector.dv)
    if det_u_px is not None:
        u_px = float(det_u_px)
    if det_v_px is not None:
        v_px = float(det_v_px)
    updated = replace(
        detector,
        det_center=(u_px * float(detector.du), v_px * float(detector.dv)),
    )
    detector_meta = dict(cast("dict[str, object]", geometry_meta.get("detector", {})))
    detector_meta["det_center"] = [float(updated.det_center[0]), float(updated.det_center[1])]
    geometry_meta["detector"] = detector_meta
    override: dict[str, JsonValue] = {
        "source": "cli_override",
        "requested_px": requested,
        "effective_px": {"det_u_px": u_px, "det_v_px": v_px},
        "effective_world": {
            "det_u": float(updated.det_center[0]),
            "det_v": float(updated.det_center[1]),
        },
    }
    logging.info(
        "Applying detector centre override: det_u_px=%.3f det_v_px=%.3f",
        u_px,
        v_px,
    )
    return updated, override


def _run_reconstruction(command: ReconCommand, config_metadata: dict[str, ConfigValue]) -> None:  # noqa: PLR0912, PLR0915
    """Run reconstruction from a typed command plan."""
    setup_logging()
    log_jax_env()
    if command.progress:
        os.environ["TOMOJAX_PROGRESS"] = "1"

    meta = load_projection_payload(command.data)
    geometry_meta = meta.geometry_inputs()
    initial_grid_override = (
        command.grid if (meta.grid is None and command.grid is not None) else None
    )
    grid, detector, geom = build_geometry_from_dataset_metadata(
        geometry_meta,
        grid_override=initial_grid_override,
        apply_saved_alignment=bool(command.apply_saved_alignment),
    )
    detector, detector_center_override = _apply_detector_center_override(
        detector,
        geometry_meta,
        det_u_px=command.det_u_px,
        det_v_px=command.det_v_px,
    )
    if command.det_u_px is not None or command.det_v_px is not None:
        grid, detector, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=initial_grid_override,
            apply_saved_alignment=bool(command.apply_saved_alignment),
        )
    if command.apply_saved_alignment and meta.align_params is not None:
        logging.info("Applying saved per-view alignment parameters from input metadata")
    proj = _jnp_float32_array(meta.projections)
    det_grid = detector_grid_from_geometry_inputs(detector, geometry_meta)
    if det_grid is not None:
        logging.info(
            "Applying saved detector_roll_deg=%s from geometry metadata",
            geometry_meta.get("detector_roll_deg"),
        )

    from tomojax.backends import default_gather_dtype as _default_gather_dtype

    _gather = str(command.gather_dtype)
    if _gather == "auto":
        _gather = _default_gather_dtype()

    # Optional ROI selection
    roi_mode = str(command.roi).lower()
    is_parallel = meta.geometry_type == "parallel"
    recon_grid = _resolve_recon_grid_for_cli(
        grid,
        detector,
        is_parallel=is_parallel,
        roi_mode=roi_mode,
    )

    # Rebuild geometry if grid changed
    if command.grid is not None:
        NX, NY, NZ = map(int, command.grid)
        recon_grid = replace(recon_grid, nx=NX, ny=NY, nz=NZ)

    if recon_grid is not grid:
        # Once ROI and explicit sizing resolve an effective grid, keep that grid's
        # origin/centre metadata authoritative when rebuilding geometry.
        _, _, geom = build_geometry_from_dataset_metadata(
            geometry_meta,
            grid_override=recon_grid,
            apply_saved_alignment=bool(command.apply_saved_alignment),
        )

    # Prepare optional volume mask
    vol_mask = _resolve_volume_mask_for_cli(
        recon_grid,
        detector,
        mask_vol=str(command.mask_vol),
    )

    resolved_vpb, vpb_mode = _resolve_views_per_batch(
        command.views_per_batch,
        algo=str(command.algo),
        n_views=int(proj.shape[0]),
        grid=recon_grid,
        detector=detector,
        gather_dtype=_gather,
        checkpoint_projector=bool(command.checkpoint_projector),
    )
    logging.info(
        "Reconstruction views_per_batch=%d (mode=%s, algo=%s)",
        resolved_vpb,
        vpb_mode,
        command.algo,
    )

    algorithm_config: dict[str, object]
    if command.algo == "fbp":
        cfg = FBPConfig(
            filter_name=str(command.filter),
            views_per_batch=int(resolved_vpb),
            projector_unroll=1,
            checkpoint_projector=bool(command.checkpoint_projector),
            gather_dtype=_gather,
        )
        algorithm_config = {
            "filter": str(cfg.filter_name),
            "views_per_batch": int(cfg.views_per_batch),
            "projector_unroll": int(cfg.projector_unroll),
            "checkpoint_projector": bool(cfg.checkpoint_projector),
            "gather_dtype": str(cfg.gather_dtype),
        }
        with transfer_guard_context(command.transfer_guard):
            vol = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                config=cfg,
                views_per_batch=int(resolved_vpb),
                det_grid=det_grid,
            )
        # For FBP, apply the requested output mask after reconstruction.
        if vol_mask is not None:
            vol = vol * vol_mask
    elif command.algo == "fista":
        cfg = FistaConfig(
            iters=int(command.iters),
            lambda_tv=float(command.lambda_tv),
            regulariser=cast("Regulariser", str(command.regulariser)),
            huber_delta=float(command.huber_delta),
            L=(float(command.lipschitz) if command.lipschitz is not None else None),
            views_per_batch=resolved_vpb,
            projector_unroll=1,
            checkpoint_projector=bool(command.checkpoint_projector),
            gather_dtype=_gather,
            tv_prox_iters=int(command.tv_prox_iters),
            support=vol_mask,
            positivity=bool(command.positivity),
            lower_bound=(float(command.lower_bound) if command.lower_bound is not None else None),
            upper_bound=(float(command.upper_bound) if command.upper_bound is not None else None),
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
        with transfer_guard_context(command.transfer_guard):
            vol = fista_tv(
                geom,
                recon_grid,
                detector,
                proj,
                config=cfg,
                det_grid=det_grid,
            )[0]
    else:  # spdhg
        # Build SPDHG config
        cfg = SPDHGConfig(
            iters=int(command.iters),
            lambda_tv=float(command.lambda_tv),
            regulariser=cast("Regulariser", str(command.regulariser)),
            huber_delta=float(command.huber_delta),
            theta=float(command.theta),
            views_per_batch=int(resolved_vpb),
            seed=int(command.spdhg_seed),
            tau=(float(command.spdhg_tau) if command.spdhg_tau is not None else None),
            sigma_data=(
                float(command.spdhg_sigma_data) if command.spdhg_sigma_data is not None else None
            ),
            sigma_tv=(
                float(command.spdhg_sigma_tv) if command.spdhg_sigma_tv is not None else None
            ),
            projector_unroll=1,
            checkpoint_projector=bool(command.checkpoint_projector),
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
            "warm_start": str(command.warm_start),
        }
        # Optional warm-start for SPDHG
        init_x = None
        if str(command.warm_start).lower() == "fbp":
            # Compute a quick FBP and use as initialization
            warm_start_vpb = 1 if vpb_mode == "default" else int(resolved_vpb)
            warm_start_cfg = FBPConfig(
                filter_name=str(command.filter),
                views_per_batch=warm_start_vpb,
                projector_unroll=1,
                checkpoint_projector=bool(command.checkpoint_projector),
                gather_dtype=_gather,
            )
            init_x = fbp(
                geom,
                recon_grid,
                detector,
                proj,
                config=warm_start_cfg,
                det_grid=det_grid,
            )
            if vol_mask is not None:
                init_x = init_x * vol_mask
            # Enforce positivity for a clean start (harmless if TV/signal expects nonnegative)
            init_x = jnp.maximum(init_x, 0.0)
        with transfer_guard_context(command.transfer_guard):
            vol = spdhg_tv(
                geom,
                recon_grid,
                detector,
                proj,
                init_x=init_x,
                config=cfg,
                det_grid=det_grid,
            )[0]

    # Save the reconstruction in the object (sample) frame.
    # Reuse host projections from metadata to avoid a device-to-host copy.
    volume_np = np.asarray(vol)
    save_meta = meta.copy_metadata()
    save_meta.grid = recon_grid.to_dict()
    save_meta.detector = detector
    save_meta.geometry_meta = dict(save_meta.geometry_meta or {})
    save_meta.geometry_meta["detector_center_override"] = detector_center_override
    save_meta.volume = volume_np
    save_meta.frame = str(command.frame)
    save_meta.volume_axes_order = str(command.volume_axes)
    save_projection_payload(
        command.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    logging.info("Saved reconstruction to %s", command.out)
    if command.quicklook is not None:
        _ = save_quicklook_png(command.quicklook, volume_np)
        logging.info("Saved reconstruction quicklook to %s", command.quicklook)
    if command.save_manifest is not None:
        manifest = build_manifest(
            "tomojax recon",
            list(sys.argv),
            asdict(command),
            {
                "input_path": command.data,
                "output_path": command.out,
                "quicklook_path": command.quicklook,
                "manifest_path": command.save_manifest,
                "config_path": config_metadata["config_path"],
                "config_file_values": config_metadata["config_file_values"],
                "explicit_cli_keys": config_metadata["explicit_cli_keys"],
                "effective_options": config_metadata["effective_options"],
                "algorithm": str(command.algo),
                "algorithm_config": algorithm_config,
                "geometry_type": str(meta.geometry_type),
                "input_projection_shape": list(meta.projections.shape),
                "reconstruction_grid": recon_grid.to_dict(),
                "detector": detector.to_dict(),
                "detector_center_override": detector_center_override,
                "detector_roll_deg": geometry_meta.get("detector_roll_deg"),
                "detector_grid_replayed": det_grid is not None,
                "roi": {
                    "requested": roi_mode,
                    "is_parallel": bool(is_parallel),
                    "grid_changed": recon_grid != grid,
                },
                "requested_views_per_batch": command.views_per_batch,
                "views_per_batch": int(resolved_vpb),
                "views_per_batch_mode": vpb_mode,
                "requested_gather_dtype": str(command.gather_dtype),
                "gather_dtype": _gather,
                "checkpoint_projector": bool(command.checkpoint_projector),
                "transfer_guard": str(command.transfer_guard),
                "mask_vol": str(command.mask_vol),
                "apply_saved_alignment": bool(command.apply_saved_alignment),
                "mask_applied": vol_mask is not None,
                "volume_shape": list(volume_np.shape),
                "volume_axes": str(command.volume_axes),
                "frame": str(command.frame),
            },
        )
        save_manifest(command.save_manifest, manifest)
        logging.info("Saved reproducibility manifest to %s", command.save_manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Run reconstruction from the public CLI."""
    command, config_metadata = _parse_command(argv)
    _run_reconstruction(command, config_metadata)


if __name__ == "__main__":  # pragma: no cover
    main()
