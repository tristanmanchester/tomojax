from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path

import numpy as np

from tomojax.calibration.center import (
    DetectorCenterCalibrationConfig,
    calibrate_detector_center,
)
from tomojax.data.geometry_meta import build_geometry_from_meta
from tomojax.data.io_hdf5 import load_nxtomo, save_nxtomo
from tomojax.recon.quicklook import save_quicklook_png
from tomojax.utils.axes import DISK_VOLUME_AXES
from tomojax.utils.fov import compute_roi, grid_from_detector_fov, grid_from_detector_fov_slices
from tomojax.utils.logging import log_jax_env, setup_logging

from .config import parse_args_with_config
from .manifest import save_manifest


def _parse_search_pass(value: str) -> tuple[float, float]:
    text = str(value).strip()
    if ":" not in text:
        raise argparse.ArgumentTypeError("search pass must be RADIUS:STEP, e.g. 10:2")
    radius_raw, step_raw = text.split(":", 1)
    try:
        radius = float(radius_raw)
        step = float(step_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("search pass radius and step must be numbers") from exc
    if not np.isfinite(radius) or radius < 0.0:
        raise argparse.ArgumentTypeError("search pass radius must be finite and >= 0")
    if not np.isfinite(step) or step <= 0.0:
        raise argparse.ArgumentTypeError("search pass step must be finite and > 0")
    return radius, step


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _heldout_stride(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("heldout stride must be >= 2")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate scanner/instrument geometry calibration parameters."
    )
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
    p.add_argument("--out", help="Output calibrated .nxs")
    p.add_argument(
        "--mode",
        choices=["detector-center"],
        default="detector-center",
        help="Calibration mode (currently only detector-center).",
    )
    p.add_argument(
        "--initial-det-u-px",
        type=float,
        default=0.0,
        help="Initial detector/ray-grid horizontal centre offset in native detector pixels.",
    )
    p.add_argument(
        "--det-v-px",
        type=float,
        default=0.0,
        help="Supplied frozen detector/ray-grid vertical centre offset in native detector pixels.",
    )
    p.add_argument(
        "--search-pass",
        action="append",
        type=_parse_search_pass,
        default=None,
        metavar="RADIUS:STEP",
        help=(
            "Add a detector-centre search pass in native pixels. Repeatable. "
            "Default: 10:2, 2:0.5, 0.5:0.1"
        ),
    )
    p.add_argument(
        "--heldout-stride",
        type=_heldout_stride,
        default=8,
        help="Use every Nth projection as held-out validation views.",
    )
    p.add_argument("--top-k", type=_positive_int, default=5, help="Number of top candidates.")
    p.add_argument("--filter", default="ramp", help="FBP filter: ramp|shepp|hann")
    p.add_argument(
        "--views-per-batch",
        type=_positive_int,
        default=1,
        help="Views per FBP backprojection batch.",
    )
    p.add_argument(
        "--gather-dtype",
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Projector gather dtype.",
    )
    ck = p.add_mutually_exclusive_group()
    ck.add_argument(
        "--checkpoint-projector",
        dest="checkpoint_projector",
        action="store_true",
        help="Enable projector checkpointing.",
    )
    ck.add_argument(
        "--no-checkpoint-projector",
        dest="checkpoint_projector",
        action="store_false",
        help="Disable projector checkpointing.",
    )
    p.set_defaults(checkpoint_projector=True)
    p.add_argument(
        "--roi",
        choices=["off", "auto", "bbox"],
        default="auto",
        help="Optional ROI cropping based on detector FOV.",
    )
    p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Override reconstruction grid size (nx ny nz).",
    )
    p.add_argument(
        "--workdir",
        default=None,
        help="Directory for calibration diagnostics (default: <out-stem>_calibration).",
    )
    p.add_argument(
        "--save-manifest",
        default=None,
        help="Write calibration manifest JSON (default: <workdir>/manifest.json).",
    )
    p.add_argument(
        "--quicklook",
        "--save-preview",
        dest="quicklook",
        default=None,
        help="Write a percentile-scaled central xy slice PNG preview.",
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
        help="On-disk axis order for saved volumes.",
    )
    p.add_argument("--progress", action="store_true", help="Show progress bars if available.")
    return p


def _resolve_recon_grid(args: argparse.Namespace, meta, grid, detector):
    recon_grid = grid
    roi_mode = str(args.roi).lower()
    is_parallel = meta.geometry_type == "parallel"
    if roi_mode != "off":
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
                    recon_grid = grid_from_detector_fov_slices(
                        grid, detector, crop_y_to_u=True
                    )
                else:
                    recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=False)
            elif roi_mode == "bbox":
                recon_grid = grid_from_detector_fov(grid, detector, crop_y_to_u=is_parallel)
        except Exception:
            recon_grid = grid

    if args.grid is not None:
        nx, ny, nz = map(int, args.grid)
        recon_grid = replace(recon_grid, nx=nx, ny=ny, nz=nz)
    return recon_grid


def _default_workdir(out_path: str) -> Path:
    out = Path(out_path)
    return out.with_name(f"{out.stem}_calibration")


def main() -> None:
    parser = _build_parser()
    args, config_metadata = parse_args_with_config(parser, required=("data", "out"))

    setup_logging()
    log_jax_env()

    meta = load_nxtomo(args.data)
    geometry_inputs = meta.geometry_inputs()
    grid, detector, _ = build_geometry_from_meta(
        geometry_inputs,
        grid_override=(args.grid if (meta.grid is None and args.grid is not None) else None),
        apply_saved_alignment=False,
    )
    recon_grid = _resolve_recon_grid(args, meta, grid, detector)
    workdir = Path(args.workdir) if args.workdir is not None else _default_workdir(args.out)
    manifest_path = (
        Path(args.save_manifest) if args.save_manifest is not None else workdir / "manifest.json"
    )
    det_v_status = (
        "supplied"
        if "det_v_px" in config_metadata["explicit_cli_keys"]
        or "det_v_px" in config_metadata["config_file_values"]
        else "frozen"
    )
    search_passes = (
        tuple(args.search_pass)
        if args.search_pass is not None
        else DetectorCenterCalibrationConfig().search_passes
    )
    cfg = DetectorCenterCalibrationConfig(
        initial_det_u_px=float(args.initial_det_u_px),
        det_v_px=float(args.det_v_px),
        det_v_status=det_v_status,
        search_passes=search_passes,
        heldout_stride=int(args.heldout_stride),
        top_k=int(args.top_k),
        filter_name=str(args.filter),
        views_per_batch=int(args.views_per_batch),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=str(args.gather_dtype),
    )
    result = calibrate_detector_center(
        geometry_inputs,
        grid=recon_grid,
        detector=detector,
        projections=meta.projections,
        config=cfg,
        workdir=workdir,
    )

    save_meta = meta.copy_metadata()
    save_meta.detector = result.calibrated_detector.to_dict()
    save_meta.grid = recon_grid.to_dict()
    save_meta.volume = result.final_volume
    save_meta.frame = str(args.frame)
    save_meta.volume_axes_order = str(args.volume_axes)
    save_meta.geometry_calibration = result.manifest
    save_nxtomo(
        args.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    save_manifest(manifest_path, result.manifest)
    if args.quicklook is not None:
        save_quicklook_png(args.quicklook, result.final_volume)

    logging.info(
        "Estimated detector/ray-grid centre det_u_px=%.4f, det_v_px=%.4f, confidence=%s",
        result.best_det_u_px,
        result.det_v_px,
        result.confidence.get("level"),
    )
    logging.info("Saved calibrated dataset to %s", args.out)
    logging.info("Saved calibration manifest to %s", manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()
