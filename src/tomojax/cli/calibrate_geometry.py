from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import logging
from pathlib import Path

from tomojax.calibration.axis import AxisDirectionCalibrationConfig, calibrate_axis_direction
from tomojax.calibration.axis_geometry import AXIS_DIRECTION_DOFS
from tomojax.calibration.center import (
    DETECTOR_CENTER_DOFS,
    DetectorCenterCalibrationConfig,
    calibrate_detector_center,
)
from tomojax.calibration.manifest import build_calibration_manifest
from tomojax.calibration.roll import DetectorRollCalibrationConfig, calibrate_detector_roll
from tomojax.calibration.state import CalibrationState
from tomojax.data.geometry_meta import build_geometry_from_meta
from tomojax.data.io_hdf5 import load_nxtomo, save_nxtomo
from tomojax.recon.quicklook import save_quicklook_png
from tomojax.utils.axes import DISK_VOLUME_AXES
from tomojax.utils.fov import compute_roi, grid_from_detector_fov, grid_from_detector_fov_slices
from tomojax.utils.logging import log_jax_env, setup_logging

from .config import parse_args_with_config
from .manifest import save_manifest


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


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative number") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a non-negative number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return parsed


def _parse_active_detector_dofs(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("active detector DOFs must not be empty")
    unknown = sorted(set(names) - set(DETECTOR_CENTER_DOFS))
    if unknown:
        allowed = ", ".join(DETECTOR_CENTER_DOFS)
        raise argparse.ArgumentTypeError(
            f"unknown detector DOF(s) {unknown}; expected one or more of: {allowed}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("active detector DOFs must not contain duplicates")
    return names


def _parse_active_axis_dofs(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("active axis DOFs must not be empty")
    unknown = sorted(set(names) - set(AXIS_DIRECTION_DOFS))
    if unknown:
        allowed = ", ".join(AXIS_DIRECTION_DOFS)
        raise argparse.ArgumentTypeError(
            f"unknown axis DOF(s) {unknown}; expected one or more of: {allowed}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("active axis DOFs must not contain duplicates")
    return names


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate scanner/instrument geometry calibration parameters."
    )
    p.add_argument("--config", help="Load command defaults from a TOML config file")
    p.add_argument("--data", help="Input .nxs")
    p.add_argument("--out", help="Output calibrated .nxs")
    p.add_argument(
        "--mode",
        choices=[
            "detector-center",
            "axis-direction",
            "detector-roll",
            "detector-center-axis",
            "detector-center-axis-roll",
        ],
        default="detector-center",
        help="Calibration mode.",
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
        "--active-detector-dofs",
        type=_parse_active_detector_dofs,
        default=("det_u_px",),
        help="Comma-separated detector-centre DOFs to optimize with GN. Default: det_u_px.",
    )
    p.add_argument(
        "--outer-iters",
        type=_positive_int,
        default=12,
        help="Maximum detector-centre Gauss-Newton outer iterations.",
    )
    p.add_argument(
        "--gn-damping",
        type=_nonnegative_float,
        default=1e-3,
        help="Levenberg-Marquardt damping for detector-centre GN.",
    )
    p.add_argument(
        "--gn-accept-tol",
        type=_nonnegative_float,
        default=0.0,
        help="Relative loss improvement required to accept a detector-centre GN step.",
    )
    p.add_argument(
        "--max-step-px",
        type=_positive_float,
        default=2.0,
        help="Maximum detector-centre GN step length in native detector pixels.",
    )
    p.add_argument(
        "--active-axis-dofs",
        type=_parse_active_axis_dofs,
        default=None,
        help=(
            "Comma-separated rotation-axis direction DOFs to optimize. "
            "Default is geometry-aware; laminography tilted about x uses axis_rot_x_deg."
        ),
    )
    p.add_argument(
        "--initial-axis-rot-x-deg",
        type=float,
        default=0.0,
        help="Initial lab-frame x-axis correction to the nominal rotation axis.",
    )
    p.add_argument(
        "--initial-axis-rot-y-deg",
        type=float,
        default=0.0,
        help="Initial lab-frame y-axis correction to the nominal rotation axis.",
    )
    p.add_argument(
        "--axis-outer-iters",
        type=_positive_int,
        default=12,
        help="Maximum rotation-axis Gauss-Newton outer iterations.",
    )
    p.add_argument(
        "--axis-gn-damping",
        type=_nonnegative_float,
        default=1e-3,
        help="Levenberg-Marquardt damping for rotation-axis GN.",
    )
    p.add_argument(
        "--axis-gn-accept-tol",
        type=_nonnegative_float,
        default=0.0,
        help="Relative loss improvement required to accept a rotation-axis GN step.",
    )
    p.add_argument(
        "--axis-max-step-deg",
        type=_positive_float,
        default=2.0,
        help="Maximum rotation-axis GN step length in degrees.",
    )
    p.add_argument(
        "--initial-detector-roll-deg",
        type=float,
        default=0.0,
        help="Initial detector-plane roll angle in degrees.",
    )
    p.add_argument(
        "--roll-outer-iters",
        type=_positive_int,
        default=12,
        help="Maximum detector-roll Gauss-Newton outer iterations.",
    )
    p.add_argument(
        "--roll-gn-damping",
        type=_nonnegative_float,
        default=1e-3,
        help="Levenberg-Marquardt damping for detector-roll GN.",
    )
    p.add_argument(
        "--roll-gn-accept-tol",
        type=_nonnegative_float,
        default=0.0,
        help="Relative loss improvement required to accept a detector-roll GN step.",
    )
    p.add_argument(
        "--roll-max-step-deg",
        type=_positive_float,
        default=1.0,
        help="Maximum detector-roll GN step length in degrees.",
    )
    refine = p.add_mutually_exclusive_group()
    refine.add_argument(
        "--refine-detector-center-after-axis",
        dest="refine_detector_center_after_axis",
        action="store_true",
        help="In combined mode, run a final detector-centre refinement after axis calibration.",
    )
    refine.add_argument(
        "--no-refine-detector-center-after-axis",
        dest="refine_detector_center_after_axis",
        action="store_false",
        help="In combined mode, skip the final detector-centre refinement after axis calibration.",
    )
    p.set_defaults(refine_detector_center_after_axis=True)
    refine_roll = p.add_mutually_exclusive_group()
    refine_roll.add_argument(
        "--refine-detector-center-after-roll",
        dest="refine_detector_center_after_roll",
        action="store_true",
        help="In roll staged mode, run a final detector-centre refinement after roll.",
    )
    refine_roll.add_argument(
        "--no-refine-detector-center-after-roll",
        dest="refine_detector_center_after_roll",
        action="store_false",
        help="In roll staged mode, skip the final detector-centre refinement after roll.",
    )
    p.set_defaults(refine_detector_center_after_roll=True)
    p.add_argument(
        "--heldout-stride",
        type=_heldout_stride,
        default=8,
        help="Use every Nth projection as held-out validation views.",
    )
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


def _detector_center_config(
    args: argparse.Namespace,
    config_metadata: dict,
) -> DetectorCenterCalibrationConfig:
    det_v_status = (
        "supplied"
        if "det_v_px" in config_metadata["explicit_cli_keys"]
        or "det_v_px" in config_metadata["config_file_values"]
        else "frozen"
    )
    return DetectorCenterCalibrationConfig(
        initial_det_u_px=float(args.initial_det_u_px),
        det_v_px=float(args.det_v_px),
        det_v_status=det_v_status,
        active_detector_dofs=tuple(args.active_detector_dofs),
        outer_iters=int(args.outer_iters),
        gn_damping=float(args.gn_damping),
        gn_accept_tol=float(args.gn_accept_tol),
        max_step_px=float(args.max_step_px),
        heldout_stride=int(args.heldout_stride),
        filter_name=str(args.filter),
        views_per_batch=int(args.views_per_batch),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=str(args.gather_dtype),
    )


def _axis_direction_config(args: argparse.Namespace) -> AxisDirectionCalibrationConfig:
    return AxisDirectionCalibrationConfig(
        active_axis_dofs=(
            None if args.active_axis_dofs is None else tuple(args.active_axis_dofs)
        ),
        initial_axis_rot_x_deg=float(args.initial_axis_rot_x_deg),
        initial_axis_rot_y_deg=float(args.initial_axis_rot_y_deg),
        outer_iters=int(args.axis_outer_iters),
        gn_damping=float(args.axis_gn_damping),
        gn_accept_tol=float(args.axis_gn_accept_tol),
        max_step_deg=float(args.axis_max_step_deg),
        heldout_stride=int(args.heldout_stride),
        filter_name=str(args.filter),
        views_per_batch=int(args.views_per_batch),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=str(args.gather_dtype),
    )


def _detector_roll_config(args: argparse.Namespace) -> DetectorRollCalibrationConfig:
    return DetectorRollCalibrationConfig(
        initial_detector_roll_deg=float(args.initial_detector_roll_deg),
        outer_iters=int(args.roll_outer_iters),
        gn_damping=float(args.roll_gn_damping),
        gn_accept_tol=float(args.roll_gn_accept_tol),
        max_step_deg=float(args.roll_max_step_deg),
        heldout_stride=int(args.heldout_stride),
        filter_name=str(args.filter),
        views_per_batch=int(args.views_per_batch),
        checkpoint_projector=bool(args.checkpoint_projector),
        gather_dtype=str(args.gather_dtype),
    )


def _geometry_inputs_with_detector(geometry_inputs: dict, detector) -> dict:
    updated = dict(geometry_inputs)
    updated["detector"] = detector.to_dict()
    return updated


def _geometry_inputs_with_axis(geometry_inputs: dict, axis_unit_lab) -> dict:
    updated = dict(geometry_inputs)
    updated["axis_unit_lab"] = [float(v) for v in axis_unit_lab]
    return updated


def _geometry_inputs_with_detector_roll(geometry_inputs: dict, detector_roll_deg: float) -> dict:
    updated = dict(geometry_inputs)
    updated["detector_roll_deg"] = float(detector_roll_deg)
    return updated


def _save_calibrated_dataset(
    *,
    args: argparse.Namespace,
    meta,
    recon_grid,
    detector,
    volume,
    manifest: dict,
    axis_unit_lab=None,
    detector_roll_deg=None,
) -> None:
    save_meta = meta.copy_metadata()
    save_meta.detector = detector.to_dict()
    save_meta.grid = recon_grid.to_dict()
    save_meta.volume = volume
    save_meta.frame = str(args.frame)
    save_meta.volume_axes_order = str(args.volume_axes)
    if axis_unit_lab is not None or detector_roll_deg is not None:
        geometry_meta = dict(save_meta.geometry_meta or {})
        if axis_unit_lab is not None:
            geometry_meta["axis_unit_lab"] = [float(v) for v in axis_unit_lab]
        if detector_roll_deg is not None:
            geometry_meta["detector_roll_deg"] = float(detector_roll_deg)
        save_meta.geometry_meta = geometry_meta
    save_meta.geometry_calibration = manifest
    save_nxtomo(
        args.out,
        projections=meta.projections,
        metadata=save_meta,
    )
    if args.quicklook is not None:
        save_quicklook_png(args.quicklook, volume)


def _combine_stage_manifests(
    *,
    detector_result,
    axis_result,
    refine_result=None,
    roll_result=None,
    refine_after_roll_result=None,
) -> dict:
    detector_state = (
        refine_after_roll_result.calibration_state
        if refine_after_roll_result is not None
        else refine_result.calibration_state
        if refine_result is not None
        else detector_result.calibration_state
    )
    detector_variables = tuple(detector_state.detector)
    if roll_result is not None:
        detector_variables = detector_variables + tuple(roll_result.calibration_state.detector)
    combined_state = CalibrationState(
        detector=detector_variables,
        scan=axis_result.calibration_state.scan,
        object_residual=axis_result.calibration_state.object_residual,
        reconstruction=axis_result.calibration_state.reconstruction,
    )
    calibrated_detector = (
        refine_after_roll_result.calibrated_detector
        if refine_after_roll_result is not None
        else refine_result.calibrated_detector
        if refine_result is not None
        else detector_result.calibrated_detector
    )
    stage_manifests = {
        "detector_center_initial": copy.deepcopy(detector_result.manifest),
        "axis_direction": copy.deepcopy(axis_result.manifest),
    }
    if refine_result is not None:
        stage_manifests["detector_center_refine"] = copy.deepcopy(refine_result.manifest)
    if roll_result is not None:
        stage_manifests["detector_roll"] = copy.deepcopy(roll_result.manifest)
    if refine_after_roll_result is not None:
        stage_manifests["detector_center_refine_after_roll"] = copy.deepcopy(
            refine_after_roll_result.manifest
        )
    mode = "detector-center-axis-roll" if roll_result is not None else "detector-center-axis"
    return build_calibration_manifest(
        calibration_state=combined_state,
        objective_card=(
            roll_result.objective_card if roll_result is not None else axis_result.objective_card
        ),
        calibrated_geometry={
            "detector": calibrated_detector.to_dict(),
            "axis_unit_lab": [float(v) for v in axis_result.axis_unit_lab],
            **(
                {"detector_roll_deg": float(roll_result.detector_roll_deg)}
                if roll_result is not None
                else {}
            ),
            "stages": list(stage_manifests),
        },
        source={
            "mode": mode,
            "detector_center_confidence": detector_result.confidence,
            "axis_direction_confidence": axis_result.confidence,
            "refined_detector_center": refine_result is not None,
            **(
                {
                    "detector_roll_confidence": roll_result.confidence,
                    "refined_detector_center_after_roll": refine_after_roll_result is not None,
                }
                if roll_result is not None
                else {}
            ),
        },
        extra={"stages": stage_manifests},
    )


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
    detector_cfg = _detector_center_config(args, config_metadata)
    axis_cfg = _axis_direction_config(args)
    roll_cfg = _detector_roll_config(args)

    if args.mode == "detector-center":
        result = calibrate_detector_center(
            geometry_inputs,
            grid=recon_grid,
            detector=detector,
            projections=meta.projections,
            config=detector_cfg,
            workdir=workdir,
        )
        final_detector = result.calibrated_detector
        final_volume = result.final_volume
        final_manifest = result.manifest
        final_axis_unit_lab = None
        final_detector_roll_deg = None
        logging.info(
            "Estimated detector/ray-grid centre det_u_px=%.4f, det_v_px=%.4f, confidence=%s",
            result.best_det_u_px,
            result.det_v_px,
            result.confidence.get("level"),
        )
    elif args.mode == "axis-direction":
        axis_result = calibrate_axis_direction(
            geometry_inputs,
            grid=recon_grid,
            detector=detector,
            projections=meta.projections,
            config=axis_cfg,
            workdir=workdir,
        )
        final_detector = detector
        final_volume = axis_result.final_volume
        final_manifest = axis_result.manifest
        final_axis_unit_lab = axis_result.axis_unit_lab
        final_detector_roll_deg = None
        logging.info(
            "Estimated rotation axis unit=(%.6f, %.6f, %.6f), confidence=%s",
            float(axis_result.axis_unit_lab[0]),
            float(axis_result.axis_unit_lab[1]),
            float(axis_result.axis_unit_lab[2]),
            axis_result.confidence.get("level"),
        )
    elif args.mode == "detector-roll":
        roll_result = calibrate_detector_roll(
            geometry_inputs,
            grid=recon_grid,
            detector=detector,
            projections=meta.projections,
            config=roll_cfg,
            workdir=workdir,
        )
        final_detector = detector
        final_volume = roll_result.final_volume
        final_manifest = roll_result.manifest
        final_axis_unit_lab = None
        final_detector_roll_deg = roll_result.detector_roll_deg
        logging.info(
            "Estimated detector roll %.6f deg, confidence=%s",
            float(roll_result.detector_roll_deg),
            roll_result.confidence.get("level"),
        )
    elif args.mode in {"detector-center-axis", "detector-center-axis-roll"}:
        detector_result = calibrate_detector_center(
            geometry_inputs,
            grid=recon_grid,
            detector=detector,
            projections=meta.projections,
            config=detector_cfg,
            workdir=workdir / "01_detector_center",
        )
        axis_inputs = _geometry_inputs_with_detector(
            geometry_inputs, detector_result.calibrated_detector
        )
        axis_result = calibrate_axis_direction(
            axis_inputs,
            grid=recon_grid,
            detector=detector_result.calibrated_detector,
            projections=meta.projections,
            config=axis_cfg,
            workdir=workdir / "02_axis_direction",
        )
        refine_result = None
        final_detector = detector_result.calibrated_detector
        final_volume = axis_result.final_volume
        final_detector_roll_deg = None
        if bool(args.refine_detector_center_after_axis):
            refine_inputs = _geometry_inputs_with_detector(
                _geometry_inputs_with_axis(axis_inputs, axis_result.axis_unit_lab),
                detector_result.calibrated_detector,
            )
            refine_result = calibrate_detector_center(
                refine_inputs,
                grid=recon_grid,
                detector=detector_result.calibrated_detector,
                projections=meta.projections,
                config=detector_cfg,
                workdir=workdir / "03_detector_center_refine",
            )
            final_detector = refine_result.calibrated_detector
            final_volume = refine_result.final_volume
        roll_result = None
        refine_after_roll_result = None
        if args.mode == "detector-center-axis-roll":
            roll_inputs = _geometry_inputs_with_detector(
                _geometry_inputs_with_axis(axis_inputs, axis_result.axis_unit_lab),
                final_detector,
            )
            roll_result = calibrate_detector_roll(
                roll_inputs,
                grid=recon_grid,
                detector=final_detector,
                projections=meta.projections,
                config=roll_cfg,
                workdir=workdir / "04_detector_roll",
            )
            final_volume = roll_result.final_volume
            final_detector_roll_deg = roll_result.detector_roll_deg
            if bool(args.refine_detector_center_after_roll):
                refine_roll_inputs = _geometry_inputs_with_detector_roll(
                    roll_inputs,
                    roll_result.detector_roll_deg,
                )
                refine_after_roll_result = calibrate_detector_center(
                    refine_roll_inputs,
                    grid=recon_grid,
                    detector=final_detector,
                    projections=meta.projections,
                    config=detector_cfg,
                    workdir=workdir / "05_detector_center_refine_after_roll",
                )
                final_detector = refine_after_roll_result.calibrated_detector
                final_volume = refine_after_roll_result.final_volume
        final_manifest = _combine_stage_manifests(
            detector_result=detector_result,
            axis_result=axis_result,
            refine_result=refine_result,
            roll_result=roll_result,
            refine_after_roll_result=refine_after_roll_result,
        )
        final_axis_unit_lab = axis_result.axis_unit_lab
        if roll_result is None:
            logging.info(
                "Estimated detector centre then axis: axis=(%.6f, %.6f, %.6f)",
                float(axis_result.axis_unit_lab[0]),
                float(axis_result.axis_unit_lab[1]),
                float(axis_result.axis_unit_lab[2]),
            )
        else:
            logging.info(
                "Estimated detector centre, axis, roll: axis=(%.6f, %.6f, %.6f), roll=%.6f deg",
                float(axis_result.axis_unit_lab[0]),
                float(axis_result.axis_unit_lab[1]),
                float(axis_result.axis_unit_lab[2]),
                float(roll_result.detector_roll_deg),
            )
    else:  # pragma: no cover - argparse choices keep this unreachable.
        raise ValueError(f"unknown calibration mode: {args.mode}")

    _save_calibrated_dataset(
        args=args,
        meta=meta,
        recon_grid=recon_grid,
        detector=final_detector,
        volume=final_volume,
        manifest=final_manifest,
        axis_unit_lab=final_axis_unit_lab,
        detector_roll_deg=final_detector_roll_deg,
    )
    save_manifest(manifest_path, final_manifest)
    logging.info("Saved calibrated dataset to %s", args.out)
    logging.info("Saved calibration manifest to %s", manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()
