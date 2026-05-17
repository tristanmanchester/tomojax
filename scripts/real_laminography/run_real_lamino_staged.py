#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Run the v2 real-laminography staged workflow."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.bench.real_laminography_planning import (
    binned_pixel_scale,
    parse_shape3,
    pose_dx_dz_bounds,
    pose_phi_bounds,
    pose_polish_bounds,
    resolve_fixture_bin_factor,
    setup_det_u_bounds,
    view_indices_for_smoke_shape,
)
from tomojax.bench.real_laminography_profiles import (
    REAL_LAMINO_PROFILE_CHOICES,
    REAL_LAMINO_STAGED_PATH,
    REFERENCE_REGRESSION_STAGE_MAP as _REFERENCE_REGRESSION_STAGE_MAP,
    apply_real_lamino_profile_args,
    apply_real_lamino_profile_contract_args,
    normalize_real_lamino_runtime_args,
    real_lamino_reference_regression_contract_payload,
)
from tomojax.bench.real_laminography_report import (
    build_real_lamino_report,
    real_lamino_artifact_validation_failures,
    real_lamino_finite_fraction,
    real_lamino_method_constraints,
    real_lamino_stat_validation_failures,
    validate_real_lamino_stage_output,
)

STAGED_PATH = REAL_LAMINO_STAGED_PATH
REFERENCE_REGRESSION_STAGE_MAP = _REFERENCE_REGRESSION_STAGE_MAP
_artifact_validation_failures = real_lamino_artifact_validation_failures
_finite_fraction = real_lamino_finite_fraction
_stat_validation_failures = real_lamino_stat_validation_failures
_validate_stage_output = validate_real_lamino_stage_output
_apply_profile_contract_args = apply_real_lamino_profile_contract_args
_apply_real_lamino_profile_args = apply_real_lamino_profile_args
_normalize_runtime_args = normalize_real_lamino_runtime_args
_reference_regression_contract_payload = real_lamino_reference_regression_contract_payload

def main(argv: list[str] | None = None) -> int:
    """Run the v2 real-laminography staged workflow."""
    args = _parse_args(argv)
    run_root = Path(args.out)
    if run_root.exists() and any(run_root.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"output exists and is not empty: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    native = _load_native_runner()
    monitor = native.GpuMonitor(run_root / "gpu_memory.csv")
    monitor.start()
    started = datetime.now().isoformat(timespec="seconds")
    try:
        summary = run_real_lamino_staged(args, native=native, started_at=started)
        print(f"real_lamino_report: {summary['artifacts']['summary_json']}")
        print(f"phase_complete: {summary['success']['passed']}")
        return 0
    finally:
        monitor.close()


def run_real_lamino_staged(  # noqa: PLR0915
    args: argparse.Namespace,
    *,
    native: Any | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Execute the v2 real-laminography workflow and write the staged report."""
    if native is None:
        native = _load_native_runner()
    _normalize_runtime_args(args)
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    started = started_at or datetime.now().isoformat(timespec="seconds")
    status_path = run_root / "status.json"
    native._status(status_path, state="starting", started_at=started)
    try:
        raw_projections, thetas = native._load_input(
            Path(args.input),
            flip_u=bool(args.flip_u),
            flip_v=bool(args.flip_v),
            transpose_detector=bool(args.transpose_detector),
        )
        raw_projections, thetas = native._validate_loaded_input(
            raw_projections,
            thetas,
            expected_projection_shape=args.expected_projection_shape,
        )
        raw_projections, thetas, geometry_inputs, binning_provenance = _prepare_binned_fixture(
            args,
            native=native,
            raw_projections=raw_projections,
            thetas=thetas,
        )
        ctx = native.RunContext(args)
        projections, background_offsets = native._apply_projection_background(
            raw_projections,
            mode=str(args.projection_background),
            edge_px=int(args.background_edge_px),
        )
        np.save(
            run_root / "projection_background_offsets.npy",
            background_offsets.astype(np.float32),
        )
        n_views, _nv, _nu = projections.shape
        grid = geometry_inputs["grid"]
        detector = geometry_inputs["detector"]
        full_nz = int(geometry_inputs["full_nz"])
        preview_local_z = native._global_z_to_local_index(
            int(args.preview_z),
            full_nz=full_nz,
            grid=grid,
        )
        if not 0 <= preview_local_z < int(grid.nz):
            raise ValueError(
                f"preview z {args.preview_z} maps outside slab local z={preview_local_z}"
            )
        geometry = native.LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(args.tilt_deg),
            tilt_about=str(args.tilt_about),
        )
        full_staged = bool(args.full_staged)
        implemented_stages = [
            "00_baseline",
            "01_setup_geometry/01_cor",
            "06_cor_only_fista",
        ]
        planned_stages = [
            "01_setup_geometry/02_detector_roll",
            "01_setup_geometry/03_axis_direction",
            "02_pose_phi",
            "03_pose_dx_dz",
            "04_pose_polish",
            "05_final",
        ]
        if full_staged:
            implemented_stages.extend(planned_stages)
            planned_stages = []
        run_manifest = {
            "schema": "tomojax.real_lamino_staged_run.v2",
            "status": "running",
            "started_at": started,
            "input": str(args.input),
            "reference_target_report": (
                str(args.reference_report) if args.reference_report else None
            ),
            "expected_projection_shape": (
                None
                if args.expected_projection_shape is None
                else list(args.expected_projection_shape)
            ),
            "input_shape": list(projections.shape),
            "binning": binning_provenance,
            "raw_projection_stats": native._projection_stats(raw_projections),
            "working_projection_stats": native._projection_stats(projections),
            "projection_preprocessing": {
                "background_mode": str(args.projection_background),
                "background_edge_px": int(args.background_edge_px),
                "background_offsets_file": "projection_background_offsets.npy",
                "baseline_reconstruction_uses": "raw_projections",
                "cor_and_fista_use": "background_corrected_projections",
            },
            "workflow": {
                "profile": str(args.profile),
                "implemented_stages": implemented_stages,
                "planned_stages": planned_stages,
                "full_staged_success_deferred": not full_staged,
                "staged_lamino": str(args.profile) == "staged-lamino",
                "reference_regression": str(args.profile) == "reference-regression",
                "reference_regression_contract": (
                    _reference_regression_contract_payload(args)
                    if str(args.profile) == "reference-regression"
                    else None
                ),
                "pose_bounds_profile": str(args.pose_bounds_profile),
                "binned_translation_bounds_scale": float(binned_pixel_scale(args)),
            },
            "reconstruction": {
                "algorithm": "fista_tv",
                "lambda_tv": float(args.lambda_tv),
                "regulariser": str(args.regulariser),
                "tv_prox_iters": int(args.tv_prox_iters),
                "positivity": bool(args.recon_positivity),
                "gather_dtype": str(args.gather_dtype),
                "views_per_batch": int(args.views_per_batch),
                "canonical_det_grid": bool(args.canonical_det_grid),
                "final_candidate_policy": str(args.final_candidate_policy),
            },
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "slab": {
                "slab_nz": int(args.slab_nz),
                "slab_center_global_z": int(args.slab_center_z),
                "preview_global_z": int(args.preview_z),
                "preview_local_z": int(preview_local_z),
                "z_stack_range": list(ctx.stack_z_range),
                "grid": grid.to_dict(),
            },
            "method_constraints": real_lamino_method_constraints(),
        }
        native._write_json(run_root / "run_manifest.json", run_manifest)

        baseline = native.run_baseline(
            ctx,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=raw_projections,
            full_nz=full_nz,
        )
        params5 = np.zeros((n_views, 5), dtype=np.float32)
        setup_state = native.GeometryCalibrationState.from_geometry(
            geometry,
            active_geometry_dofs=(),
        )
        _, setup_state, stats = native.run_setup_stage(
            ctx,
            stage_dir=ctx.stage_dir("01_setup_geometry/01_cor"),
            stage_name="01_setup_geometry/01_cor",
            active_setup=("det_u_px",),
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=setup_state,
            params5=params5,
            levels=tuple(int(v) for v in args.levels_setup),
            bounds=setup_det_u_bounds(args),
            level_outer_counts=_reference_regression_level_outer_counts(
                args,
                stage_name="01_setup_geometry/01_cor",
            ),
        )
        cor_setup_state = setup_state
        cor_only = run_cor_only_fista(
            ctx,
            native=native,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=cor_setup_state,
        )
        final_volume = None
        stage_records = [
            {"stage": "00_baseline", "volume_shape": list(baseline.shape)},
            {
                "stage": "01_setup_geometry/01_cor",
                "stats_count": len(stats),
                "geometry_calibration_state": cor_setup_state.to_calibration_state().to_dict(),
            },
            {"stage": "06_cor_only_fista", "volume_shape": list(cor_only.shape)},
        ]
        if full_staged:
            setup_state, params5, staged_records, final_candidates = run_remaining_stages(
                ctx,
                native=native,
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                full_nz=full_nz,
                setup_state=setup_state,
                params5=params5,
            )
            stage_records.extend(staged_records)
            final_volume, final_choice = run_best_final_reconstruction(
                ctx,
                native=native,
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                full_nz=full_nz,
                candidates=final_candidates,
            )
            setup_state = final_choice["setup_state"]
            params5 = final_choice["params5"]
            native._write_params_csv(run_root / "05_final" / "params.csv", params5)
            stage_records.append(
                {
                    "stage": "05_final",
                    "volume_shape": list(final_volume.shape),
                    "selected_candidate": final_choice["label"],
                    "selected_candidate_source_stage": final_choice["source_stage"],
                    "selected_candidate_loss": final_choice["loss_last"],
                }
            )
        else:
            _write_planned_stage_manifests(run_root, native=native)
        completed = datetime.now().isoformat(timespec="seconds")
        final_payload = {
            "status": "completed",
            "completed_at": completed,
            "stage_records": stage_records,
            "final_setup_estimates": setup_state.to_calibration_state().to_dict(),
            "final_pose_summary": native._params_summary(params5),
            "final_volume_shape": list(
                final_volume.shape if final_volume is not None else cor_only.shape
            ),
        }
        native._write_json(run_root / "run_manifest.json", {**run_manifest, **final_payload})
        native._status(ctx.status_path, state="completed", stage="complete", **final_payload)
        return build_real_lamino_staged_report(
            run_root,
            out_dir=run_root / "real_lamino_report",
            reference_report=Path(args.reference_report) if args.reference_report else None,
        )
    except Exception as exc:
        native._status(status_path, state="failed", stage="error", error=repr(exc))
        raise


def run_cor_only_fista(
    ctx: Any,
    *,
    native: Any,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    setup_state: Any,
) -> np.ndarray:
    """Run the COR-only final FISTA comparator and write stage artifacts."""
    stage_dir = ctx.stage_dir("06_cor_only_fista")
    stage_dir.mkdir(parents=True, exist_ok=True)
    native._status(ctx.status_path, state="running", stage="06_cor_only_fista")
    geom_eff = native.geometry_with_axis_state(geometry, grid, detector, setup_state)
    det_grid = (
        None
        if bool(ctx.args.canonical_det_grid)
        else native.level_detector_grid(detector, state=setup_state, factor=1)
    )
    t0 = time.perf_counter()
    vol, info = native.fista_tv(
        geom_eff,
        grid,
        detector,
        jnp.asarray(projections, dtype=jnp.float32),
        config=native.FistaConfig(
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
        stage_name="06_cor_only_fista",
        volume=vol_np,
        grid=grid,
        full_nz=full_nz,
        input_reference=ctx.naive_slice,
        suffix="aligned",
    )
    manifest = {
        "stage": "06_cor_only_fista",
        "status": "completed",
        "elapsed_seconds": float(elapsed),
        "active_dofs": ["det_u_px"],
        "volume_shape": list(vol_np.shape),
        "fista_info": info,
        "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
        "setup_state": setup_state.to_calibration_state().to_dict(),
        "artifacts": products,
    }
    native._write_json(stage_dir / "stage_manifest.json", manifest)
    native._write_json(stage_dir / "align_info.json", {"fista_info": info})
    native._write_json(
        stage_dir / "geometry_calibration_state.json",
        setup_state.to_calibration_state().to_dict(),
    )
    native._append_csv(
        stage_dir / "stage_summary.csv",
        {
            "stage": "06_cor_only_fista",
            "status": "completed",
            "elapsed_seconds": float(elapsed),
            "loss_first": _loss_summary(info)["first"],
            "loss_last": _loss_summary(info)["last"],
        },
        ["stage", "status", "elapsed_seconds", "loss_first", "loss_last"],
    )
    return vol_np


def run_remaining_stages(
    ctx: Any,
    *,
    native: Any,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    setup_state: Any,
    params5: np.ndarray,
) -> tuple[Any, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run detector roll, axis, and pose stages after the COR-only comparator."""
    records: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = [
        {
            "label": "01_cor",
            "source_stage": "01_setup_geometry/01_cor",
            "setup_state": setup_state,
            "params5": np.asarray(params5, dtype=np.float32).copy(),
        }
    ]
    setup_plan = (
        (
            "01_setup_geometry/02_detector_roll",
            ("detector_roll_deg",),
            "detector_roll_deg=-10:10",
        ),
        (
            "01_setup_geometry/03_axis_direction",
            ("axis_rot_x_deg", "axis_rot_y_deg"),
            "axis_rot_x_deg=-15:15,axis_rot_y_deg=-15:15",
        ),
    )
    for idx, (stage_name, active_setup, bounds) in enumerate(setup_plan):
        x_stage, setup_state, stats = native.run_setup_stage(
            ctx,
            stage_dir=ctx.stage_dir(stage_name),
            stage_name=stage_name,
            active_setup=active_setup,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=setup_state,
            params5=params5,
            levels=tuple(int(v) for v in ctx.args.levels_setup),
            bounds=bounds,
            level_outer_counts=_reference_regression_level_outer_counts(
                ctx.args,
                stage_name=stage_name,
            ),
        )
        validation = _validate_stage_output(
            ctx.stage_dir(stage_name),
            stage_name=stage_name,
            volume=x_stage,
            params5=params5,
            stats=stats,
            require_data_loss=False,
        )
        if not validation["passed"]:
            _mark_stage_failed(
                native,
                ctx.stage_dir(stage_name),
                stage_name=stage_name,
                validation=validation,
            )
            records.append(
                {
                    "stage": stage_name,
                    "status": "failed",
                    "stats_count": len(stats),
                    "failure_provenance": validation,
                }
            )
            _write_skipped_stage_manifests(
                ctx.run_root,
                native=native,
                stages=[spec[0] for spec in setup_plan[idx + 1 :]]
                + ["02_pose_phi", "03_pose_dx_dz", "04_pose_polish"],
                reason=f"upstream stage {stage_name} failed validation",
            )
            return setup_state, params5, records, final_candidates
        records.append(
            {
                "stage": stage_name,
                "status": "completed",
                "stats_count": len(stats),
                "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
            }
        )
        final_candidates.append(
            {
                "label": stage_name.rsplit("/", 1)[-1],
                "source_stage": stage_name,
                "setup_state": setup_state,
                "params5": np.asarray(params5, dtype=np.float32).copy(),
            }
        )
    pose_plan = (
        ("02_pose_phi", ("phi",), tuple(ctx.args.levels_phi), pose_phi_bounds(ctx.args)),
        ("03_pose_dx_dz", ("dx", "dz"), tuple(ctx.args.levels_dx_dz), pose_dx_dz_bounds(ctx.args)),
        (
            "04_pose_polish",
            ("alpha", "beta", "phi", "dx", "dz"),
            tuple(ctx.args.levels_polish),
            pose_polish_bounds(ctx.args),
        ),
    )
    for idx, (stage_name, active_pose, levels, bounds) in enumerate(pose_plan):
        x_stage, params5, stats = native.run_pose_stage(
            ctx,
            stage_dir=ctx.stage_dir(stage_name),
            stage_name=stage_name,
            active_pose=active_pose,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=setup_state,
            params5=params5,
            levels=tuple(int(v) for v in levels),
            bounds=bounds,
        )
        validation = _validate_stage_output(
            ctx.stage_dir(stage_name),
            stage_name=stage_name,
            volume=x_stage,
            params5=params5,
            stats=stats,
            require_data_loss=True,
        )
        if not validation["passed"]:
            _mark_stage_failed(
                native,
                ctx.stage_dir(stage_name),
                stage_name=stage_name,
                validation=validation,
            )
            records.append(
                {
                    "stage": stage_name,
                    "status": "failed",
                    "stats_count": len(stats),
                    "params_summary": _safe_params_summary(native, params5),
                    "failure_provenance": validation,
                }
            )
            _write_skipped_stage_manifests(
                ctx.run_root,
                native=native,
                stages=[spec[0] for spec in pose_plan[idx + 1 :]],
                reason=f"upstream pose stage {stage_name} failed validation",
            )
            return setup_state, params5, records, final_candidates
        records.append(
            {
                "stage": stage_name,
                "status": "completed",
                "stats_count": len(stats),
                "params_summary": native._params_summary(params5),
            }
        )
        final_candidates.append(
            {
                "label": stage_name,
                "source_stage": stage_name,
                "setup_state": setup_state,
                "params5": np.asarray(params5, dtype=np.float32).copy(),
            }
        )
    return setup_state, params5, records, final_candidates


def run_best_final_reconstruction(
    ctx: Any,
    *,
    native: Any,
    geometry: Any,
    grid: Any,
    detector: Any,
    projections: np.ndarray,
    full_nz: int,
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run final FISTA candidates and publish the lowest-loss final artifact."""
    if not candidates:
        raise ValueError("at least one final reconstruction candidate is required")
    candidate_policy = str(
        getattr(getattr(ctx, "args", object()), "final_candidate_policy", "all")
    )
    candidates_to_score = _select_final_candidates(candidates, policy=candidate_policy)
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
            volume = native._final_reconstruct(
                ctx,
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                full_nz=full_nz,
                setup_state=candidate["setup_state"],
                params5=np.asarray(candidate["params5"], dtype=np.float32),
            )
            manifest = _read_json(candidate_root / "05_final" / "stage_manifest.json")
            loss_last = _loss_summary(manifest.get("recon_info", {})).get("last")
            validation = _validate_stage_output(
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
                _mark_stage_failed(
                    native,
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
    manifest = _read_json(final_dir / "stage_manifest.json")
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
    native._write_json(final_dir / "stage_manifest.json", manifest)
    return np.asarray(best["volume"], dtype=np.float32), best


def _select_final_candidates(
    candidates: list[dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "all":
        return candidates
    if normalized == "last_valid":
        return [candidates[-1]]
    if normalized == "setup_only":
        setup_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("source_stage", "")).startswith("01_setup_geometry/")
            or str(candidate.get("source_stage", "")) == "01_setup_geometry/01_cor"
        ]
        return setup_candidates or [candidates[-1]]
    raise ValueError(
        "final candidate policy must be one of 'all', 'last_valid', or 'setup_only'; "
        f"got {policy!r}"
    )


def _mark_stage_failed(
    native: Any,
    stage_dir: Path,
    *,
    stage_name: str,
    validation: Mapping[str, Any],
) -> None:
    manifest_path = stage_dir / "stage_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {"stage": stage_name}
    manifest["status"] = "failed"
    manifest["failure_provenance"] = dict(validation)
    native._write_json(manifest_path, manifest)
    native._write_json(stage_dir / "failure_provenance.json", dict(validation))


def _write_skipped_stage_manifests(
    root: Path,
    *,
    native: Any,
    stages: list[str],
    reason: str,
) -> None:
    for stage in stages:
        stage_dir = Path(root) / stage
        manifest_path = stage_dir / "stage_manifest.json"
        if manifest_path.exists():
            continue
        stage_dir.mkdir(parents=True, exist_ok=True)
        native._write_json(
            manifest_path,
            {
                "stage": stage,
                "status": "skipped",
                "skip_reason": reason,
                "failure_provenance": {
                    "schema": "tomojax.real_lamino_stage_validation.v1",
                    "stage": stage,
                    "passed": False,
                    "failures": [reason],
                },
            },
        )


def _safe_params_summary(native: Any, params5: np.ndarray) -> dict[str, Any] | None:
    if _finite_fraction(params5) != 1.0:
        return None
    return native._params_summary(params5)


def build_real_lamino_staged_report(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    reference_report: Path | None = None,
) -> dict[str, Any]:
    """Write the staged real-laminography report via the bench-owned facade."""
    return build_real_lamino_report(
        Path(run_dir),
        out_dir=out_dir,
        reference_report=reference_report,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Run the v2 staged real-laminography workflow."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--expected-projection-shape", type=parse_shape3, default=None)
    parser.add_argument("--tilt-deg", type=float, default=34.4)
    parser.add_argument("--tilt-about", choices=["x", "z"], default="x")
    parser.add_argument("--flip-u", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-v", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--transpose-detector",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--preview-z", type=int, default=209)
    parser.add_argument("--slab-center-z", type=int, default=209)
    parser.add_argument("--slab-nz", type=int, default=96)
    parser.add_argument("--stack-z-range", default="198:220")
    parser.add_argument(
        "--bin-factor",
        type=int,
        default=1,
        help="Bin the real input projections and reconstruction grid by this factor.",
    )
    parser.add_argument(
        "--diagnostic-shape",
        dest="smoke_shape",
        type=parse_shape3,
        default=None,
        metavar="N,NV,NU",
        help=(
            "Optional real-data diagnostic target as N,NV,NU. Views are deterministically "
            "subselected and the bin factor is raised so binned detector dims fit."
        ),
    )
    parser.add_argument(
        "--smoke-shape",
        dest="smoke_shape",
        type=parse_shape3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--levels-setup", nargs="+", type=int, default=[8, 4, 2])
    parser.add_argument("--levels-phi", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--levels-dx-dz", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--levels-polish", nargs="+", type=int, default=[2, 1])
    parser.add_argument("--outer-iters", type=int, default=8)
    parser.add_argument("--recon-iters", type=int, default=40)
    parser.add_argument("--tv-prox-iters", type=int, default=16)
    parser.add_argument("--lambda-tv", type=float, default=0.008)
    parser.add_argument("--align-profile", choices=["lightning", "tortoise"], default="lightning")
    parser.add_argument("--projector-backend", choices=["pallas", "jax"], default="pallas")
    parser.add_argument("--regulariser", choices=["huber_tv", "tv"], default="huber_tv")
    parser.add_argument("--quality-tier", choices=["fast", "reference"], default="fast")
    parser.add_argument("--fallback-policy", choices=["fallback", "strict"], default="fallback")
    parser.add_argument(
        "--pose-model",
        choices=["per_view", "polynomial", "spline"],
        default="spline",
    )
    parser.add_argument("--knot-spacing", type=int, default=8)
    parser.add_argument("--pose-degree", type=int, default=3)
    parser.add_argument(
        "--canonical-det-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--projection-background",
        choices=["none", "view_median", "edge_median"],
        default="edge_median",
    )
    parser.add_argument("--background-edge-px", type=int, default=16)
    parser.add_argument("--recon-positivity", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--views-per-batch", type=int, default=0)
    parser.add_argument("--gather-dtype", default="bf16")
    parser.add_argument(
        "--final-candidate-policy",
        choices=["all", "last_valid", "setup_only"],
        default="all",
        help=(
            "Which staged geometry candidates to score for the final reconstruction. "
            "'all' preserves the exhaustive diagnostic sweep; 'last_valid' is the "
            "fast production confirmation path."
        ),
    )
    parser.add_argument("--gn-damping", type=float, default=1e-3)
    parser.add_argument("--filter", dest="filter_name", default="ramp")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-rel", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--snapshot-max-cols", type=int, default=6)
    parser.add_argument(
        "--pose-bounds-profile",
        choices=["reference_conservative", "wide"],
        default="reference_conservative",
    )
    parser.add_argument(
        "--profile",
        choices=REAL_LAMINO_PROFILE_CHOICES,
        default="manual",
        help=(
            "Resolved real-laminography profile. 'staged-lamino' runs the clean "
            "staged workflow; 'reference-regression' preserves the internal "
            "reference-run comparison contract; 'diagnostic-fast' enables the "
            "bounded diagnostic workflow."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--full-staged",
        action="store_true",
        help="Run detector-roll, axis, pose, polish, and final reconstruction after COR-only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    _apply_real_lamino_profile_args(args, parser)
    if bool(args.smoke):
        if int(args.bin_factor) <= 1 and args.smoke_shape is None:
            args.bin_factor = 4
        args.slab_nz = min(int(args.slab_nz), 48)
        args.levels_setup = [8]
        args.levels_phi = [8]
        args.levels_dx_dz = [8]
        args.levels_polish = [8]
        args.outer_iters = min(int(args.outer_iters), 1)
        args.recon_iters = min(int(args.recon_iters), 3)
        args.tv_prox_iters = min(int(args.tv_prox_iters), 2)
        args.snapshot_max_cols = min(int(args.snapshot_max_cols), 4)
        if int(args.views_per_batch) <= 0:
            args.views_per_batch = 1
    return args


def _reference_regression_level_outer_counts(
    args: argparse.Namespace,
    *,
    stage_name: str,
) -> dict[int, int] | None:
    """Return reference-run per-level setup row counts for strict replay."""
    if str(getattr(args, "profile", "")) != "reference-regression":
        return None
    reference_report = getattr(args, "reference_report", None)
    if not reference_report:
        return None
    reference_root = Path(reference_report).resolve().parents[1]
    summary_path = reference_root / stage_name / "stage_summary.csv"
    rows = _read_stage_summary(summary_path)
    counts: dict[int, int] = {}
    for row in rows:
        try:
            level = int(row.get("level_factor", ""))
        except (TypeError, ValueError):
            continue
        counts[level] = counts.get(level, 0) + 1
    return counts or None


def _map_global_z_to_binned(
    native: Any,
    global_z: int,
    *,
    original_full_nz: int,
    binned_full_nz: int,
    binned_grid: Any,
) -> int:
    phys_z = native._global_z_to_phys(int(global_z), full_nz=int(original_full_nz))
    local_z = native._phys_z_to_local_index(phys_z, binned_grid)
    local_z = int(np.clip(local_z, 0, int(binned_grid.nz) - 1))
    mapped = native._local_z_to_global_index(
        local_z,
        full_nz=int(binned_full_nz),
        grid=binned_grid,
    )
    return int(np.clip(mapped, 0, int(binned_full_nz) - 1))


def _prepare_binned_fixture(
    args: argparse.Namespace,
    *,
    native: Any,
    raw_projections: np.ndarray,
    thetas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Derive the optional binned real-data fixture and mutate working args."""
    original_shape = tuple(int(v) for v in raw_projections.shape)
    original_full_nz = int(original_shape[1])
    view_indices = view_indices_for_smoke_shape(original_shape[0], args.smoke_shape)
    raw_selected = np.asarray(raw_projections[view_indices], dtype=np.float32)
    thetas_selected = np.asarray(thetas[view_indices], dtype=np.float32)
    bin_factor = resolve_fixture_bin_factor(
        projection_shape=tuple(int(v) for v in raw_selected.shape),
        slab_nz=int(args.slab_nz),
        requested_bin_factor=int(args.bin_factor),
        smoke_shape=args.smoke_shape,
    )

    center_phys_z = native._global_z_to_phys(int(args.slab_center_z), full_nz=original_full_nz)
    base_grid = native.Grid(
        nx=int(original_shape[2]),
        ny=int(original_shape[2]),
        nz=int(args.slab_nz),
        vx=1.0,
        vy=1.0,
        vz=1.0,
        vol_center=(0.0, 0.0, center_phys_z),
    )
    base_detector = native.Detector(
        nu=int(original_shape[2]),
        nv=int(original_shape[1]),
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    if bin_factor > 1:
        working_raw = np.asarray(
            native.bin_projections(jnp.asarray(raw_selected, dtype=jnp.float32), bin_factor),
            dtype=np.float32,
        )
        grid = native.scale_grid(base_grid, bin_factor)
        detector = native.scale_detector(base_detector, bin_factor)
    else:
        working_raw = raw_selected
        grid = base_grid
        detector = base_detector

    binned_detector_nz = int(detector.nv)
    original_preview_z = int(args.preview_z)
    original_slab_center_z = int(args.slab_center_z)
    original_stack_z_range = tuple(native._parse_range(str(args.stack_z_range)))
    args.slab_nz = int(grid.nz)
    args.effective_bin_factor = int(bin_factor)
    args.effective_view_indices = [int(v) for v in view_indices.tolist()]

    provenance = {
        "enabled": bool(bin_factor > 1 or len(view_indices) != original_shape[0]),
        "requested_bin_factor": int(args.bin_factor),
        "effective_bin_factor": int(bin_factor),
        "requested_smoke_shape": None if args.smoke_shape is None else list(args.smoke_shape),
        "original_projection_shape": list(original_shape),
        "selected_projection_shape_before_binning": list(raw_selected.shape),
        "working_projection_shape": list(working_raw.shape),
        "view_indices": [int(v) for v in view_indices.tolist()],
        "original_slab_nz": int(base_grid.nz),
        "working_slab_nz": int(grid.nz),
        "coordinate_full_nz": int(original_full_nz),
        "working_detector_nz": int(binned_detector_nz),
        "original_preview_global_z": int(original_preview_z),
        "working_preview_global_z": int(original_preview_z),
        "original_slab_center_global_z": int(original_slab_center_z),
        "working_slab_center_global_z": int(original_slab_center_z),
        "original_stack_z_range": list(original_stack_z_range),
        "working_stack_z_range": list(original_stack_z_range),
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "detector_shift_bound_scale": float(binned_pixel_scale(args)),
        "pose_dx_dz_bound_scale": float(binned_pixel_scale(args)),
    }
    geometry_inputs = {"grid": grid, "detector": detector, "full_nz": int(original_full_nz)}
    return working_raw, thetas_selected, geometry_inputs, provenance


def _load_native_runner() -> Any:
    path = Path(__file__).with_name("run_real_lamino_reference_regression.py")
    spec = importlib.util.spec_from_file_location("run_real_lamino_reference_regression", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load native runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_planned_stage_manifests(root: Path, *, native: Any) -> None:
    for spec in STAGED_PATH:
        if spec["status"] != "planned":
            continue
        stage_dir = root / str(spec["stage"])
        stage_dir.mkdir(parents=True, exist_ok=True)
        native._write_json(
            stage_dir / "stage_manifest.json",
            {
                "stage": spec["stage"],
                "label": spec["label"],
                "status": "planned",
                "active_dofs": spec["active_dofs"],
                "planned_after": "v2 COR-only path works",
            },
        )


def _resolve_artifact_path(stage_dir: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return stage_dir / candidate


def _loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    losses = info.get("loss", [])
    if not isinstance(losses, list) or not losses:
        return {"first": None, "last": None, "iters": 0}
    first = float(losses[0])
    last = float(losses[-1])
    if not np.isfinite(first):
        first = None
    if not np.isfinite(last):
        last = None
    return {
        "first": first,
        "last": last,
        "iters": int(info.get("effective_iters", len(losses))),
    }


def _read_stage_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
