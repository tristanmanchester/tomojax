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
import math
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

from tomojax.bench.real_laminography_profiles import (
    REAL_LAMINO_PROFILE_CHOICES,
    REAL_LAMINO_STAGED_PATH,
    REFERENCE_REGRESSION_CONTRACT,
    REFERENCE_REGRESSION_STAGE_MAP,
    STAGED_LAMINO_CONTRACT,
)

STAGED_PATH = REAL_LAMINO_STAGED_PATH

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
                "binned_translation_bounds_scale": float(_binned_pixel_scale(args)),
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
            "method_constraints": _method_constraints(),
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
            bounds=_setup_det_u_bounds(args),
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
        ("02_pose_phi", ("phi",), tuple(ctx.args.levels_phi), _pose_phi_bounds(ctx.args)),
        ("03_pose_dx_dz", ("dx", "dz"), tuple(ctx.args.levels_dx_dz), _pose_dx_dz_bounds(ctx.args)),
        (
            "04_pose_polish",
            ("alpha", "beta", "phi", "dx", "dz"),
            tuple(ctx.args.levels_polish),
            _pose_polish_bounds(ctx.args),
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


def _validate_stage_output(
    stage_dir: Path,
    *,
    stage_name: str,
    volume: Any | None,
    params5: Any | None,
    stats: list[dict[str, Any]],
    require_data_loss: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    volume_fraction = _finite_fraction(volume)
    if volume_fraction != 1.0:
        failures.append(f"reconstruction volume finite fraction is {volume_fraction:.6g}")
    params_fraction = _finite_fraction(params5)
    if params_fraction != 1.0:
        failures.append(f"pose/setup params finite fraction is {params_fraction:.6g}")
    checkpoint_failures = _checkpoint_validation_failures(stage_dir)
    failures.extend(checkpoint_failures)
    failures.extend(_stat_validation_failures(stats, require_data_loss=require_data_loss))
    artifact_failures = _artifact_validation_failures(stage_dir)
    failures.extend(artifact_failures)
    return {
        "schema": "tomojax.real_lamino_stage_validation.v1",
        "stage": stage_name,
        "passed": not failures,
        "failures": failures,
        "volume_finite_fraction": volume_fraction,
        "params_finite_fraction": params_fraction,
        "checkpoint_failures": checkpoint_failures,
        "artifact_failures": artifact_failures,
        "require_data_loss": bool(require_data_loss),
    }


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


def _finite_fraction(value: Any | None) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).mean())


def _checkpoint_validation_failures(stage_dir: Path) -> list[str]:
    failures: list[str] = []
    checkpoint_dir = stage_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return failures
    for path in sorted(checkpoint_dir.glob("*.npz")):
        try:
            with np.load(path) as payload:
                if "x" not in payload:
                    failures.append(f"{path.name} missing x checkpoint array")
                    continue
                fraction = _finite_fraction(payload["x"])
        except Exception as exc:
            failures.append(f"{path.name} could not be read: {type(exc).__name__}: {exc}")
            continue
        if fraction != 1.0:
            failures.append(f"{path.name} x finite fraction is {fraction:.6g}")
    return failures


def _stat_validation_failures(
    stats: list[dict[str, Any]],
    *,
    require_data_loss: bool,
) -> list[str]:
    failures: list[str] = []
    for idx, stat in enumerate(stats):
        finite_reported_losses = [
            key
            for key in ("geometry_loss_before", "geometry_loss_after", "loss_before", "loss_after")
            if key in stat and _is_finite_scalar(stat.get(key))
        ]
        failures.extend(
            f"stat[{idx}] {key} is non-finite: {stat.get(key)!r}"
            for key in ("geometry_loss_before", "geometry_loss_after", "loss_before", "loss_after")
            if key in stat and not _is_finite_scalar(stat.get(key))
        )
        if (
            require_data_loss
            and stat.get("data_loss_computed") is False
            and not finite_reported_losses
        ):
            failures.append(
                f"stat[{idx}] data_loss_computed is false and no finite objective loss was reported"
            )
    return failures


def _artifact_validation_failures(stage_dir: Path) -> list[str]:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return ["stage artifacts payload is missing or not an object"]
    failures: list[str] = []
    for key, raw_path in artifacts.items():
        path = _resolve_artifact_path(stage_dir, raw_path)
        if not path.exists():
            failures.append(f"artifact {key} is missing: {path}")
        elif path.stat().st_size <= 0:
            failures.append(f"artifact {key} is empty: {path}")
    return failures


def _is_finite_scalar(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def build_real_lamino_staged_report(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    reference_report: Path | None = None,
) -> dict[str, Any]:
    """Write the staged real-laminography report."""
    root = Path(run_dir)
    out = out_dir or root / "real_lamino_report"
    out.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(root / "run_manifest.json")
    status = _read_json(root / "status.json") if (root / "status.json").exists() else {}
    records = [_stage_record(root, spec) for spec in STAGED_PATH]
    reconstruction = _reconstruction_comparison(root)
    success = _success_payload(reconstruction, records)
    publication = _copy_publication_images(
        root,
        out,
        full_completed=reconstruction["final"]["status"] == "completed"
        and reconstruction["final"]["loss"]["last"] is not None,
    )
    residual_trace = _write_residual_trace(out / "real_lamino_residual_trace.csv", records)
    geometry_trace = _write_geometry_trace(out / "real_lamino_geometry_trace.json", records)
    reference_regression = _write_reference_regression_audit(
        root=root,
        out_dir=out,
        reference_report=reference_report,
        run_manifest=run_manifest,
    )
    summary: dict[str, Any] = {
        "schema": "tomojax.real_lamino_staged_report.v2",
        "contract_compatible_with": "tomojax.real_lamino_staged_report.v2",
        "run_dir": str(root),
        "reference_target_report": str(reference_report) if reference_report else None,
        "reference_case": root.name,
        "success": success,
        "quality_basis": {
            "kind": success["quality_kind"],
            "primary_metric": success["primary_metric"],
            "full_staged_primary_metric": "final_fista_last_loss_lt_cor_only_fista_last_loss",
            "full_staged_success_deferred": success["full_staged_success_deferred"],
            "truth_metrics": "not_applicable_real_data",
            "synthetic_truth_metrics_allowed": False,
        },
        "staged_path": records,
        "reconstruction_comparison": reconstruction,
        "publication_artifacts": publication,
        "reference_regression": reference_regression["payload"],
        "provenance": {
            "input": run_manifest.get("input"),
            "binning": run_manifest.get("binning"),
            "backend": run_manifest.get("backend"),
            "devices": run_manifest.get("devices"),
            "final_volume_shape": run_manifest.get("final_volume_shape"),
            "final_setup_estimates": run_manifest.get("final_setup_estimates"),
            "final_pose_summary": run_manifest.get("final_pose_summary"),
            "status_completed_at": status.get("completed_at", run_manifest.get("completed_at")),
        },
        "method_constraints": _method_constraints(),
        "artifacts": {
            "summary_json": str((out / "real_lamino_summary.json").resolve()),
            "summary_md": str((out / "real_lamino_summary.md").resolve()),
            "residual_trace_csv": str(residual_trace.resolve()),
            "geometry_trace_json": str(geometry_trace.resolve()),
            "publication_dir": str((out / "publication").resolve()),
            **reference_regression["artifacts"],
        },
    }
    _write_json(out / "real_lamino_summary.json", summary)
    _write_partial_markdown(out / "real_lamino_summary.md", summary)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Run the v2 staged real-laminography workflow."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--expected-projection-shape", type=_parse_shape3, default=None)
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
        type=_parse_shape3,
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
        type=_parse_shape3,
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


def _apply_real_lamino_profile_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if str(args.profile) == "reference-regression":
        if bool(args.smoke):
            parser.error("--profile reference-regression cannot be combined with diagnostic mode")
        _apply_profile_contract_args(args, REFERENCE_REGRESSION_CONTRACT)
        args.reference_regression = True
    elif str(args.profile) == "staged-lamino":
        _apply_profile_contract_args(args, STAGED_LAMINO_CONTRACT)
        args.reference_regression = False
    elif str(args.profile) == "diagnostic-fast":
        args.full_staged = True
        args.smoke = True
        if str(args.final_candidate_policy) == "all":
            args.final_candidate_policy = "last_valid"
        args.reference_regression = False
    else:
        args.reference_regression = False


def _apply_profile_contract_args(args: argparse.Namespace, contract: Mapping[str, Any]) -> None:
    args.full_staged = True
    args.levels_setup = list(contract["levels_setup"])
    args.levels_phi = list(contract["levels_phi"])
    args.levels_dx_dz = list(contract["levels_dx_dz"])
    args.levels_polish = list(contract["levels_polish"])
    args.outer_iters = int(contract["outer_iters"])
    args.recon_iters = int(contract["recon_iters"])
    args.tv_prox_iters = int(contract["tv_prox_iters"])
    args.lambda_tv = float(contract["lambda_tv"])
    args.align_profile = str(contract["align_profile"])
    args.regulariser = str(contract["regulariser"])
    args.gn_damping = float(contract["gn_damping"])
    args.quality_tier = str(contract["quality_tier"])
    args.fallback_policy = str(contract["fallback_policy"])
    args.fold_rigid_detector_grid = bool(contract["fold_rigid_detector_grid"])
    args.pose_model = str(contract["pose_model"])
    args.knot_spacing = int(contract["knot_spacing"])
    args.pose_degree = int(contract["pose_degree"])
    args.pose_bounds_profile = str(contract["pose_bounds_profile"])
    args.canonical_det_grid = bool(contract["canonical_det_grid"])
    args.projection_background = str(contract["projection_background"])
    args.background_edge_px = int(contract["background_edge_px"])
    args.recon_positivity = bool(contract["recon_positivity"])
    args.views_per_batch = int(contract["views_per_batch"])
    args.gather_dtype = str(contract["gather_dtype"])
    args.final_candidate_policy = str(contract["final_candidate_policy"])


def _normalize_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve memory-sensitive runtime defaults after CLI parsing."""
    if int(args.views_per_batch) <= 0:
        args.views_per_batch = 1
    args.bin_factor = _validate_bin_factor(args.bin_factor)
    return args


def _reference_regression_contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    actual = {
        "projection_background": str(args.projection_background),
        "background_edge_px": int(args.background_edge_px),
        "canonical_det_grid": bool(args.canonical_det_grid),
        "levels_setup": list(args.levels_setup),
        "levels_phi": list(args.levels_phi),
        "levels_dx_dz": list(args.levels_dx_dz),
        "levels_polish": list(args.levels_polish),
        "outer_iters": int(args.outer_iters),
        "recon_iters": int(args.recon_iters),
        "tv_prox_iters": int(args.tv_prox_iters),
        "lambda_tv": float(args.lambda_tv),
        "align_profile": str(args.align_profile),
        "regulariser": str(args.regulariser),
        "loss_spec": "l2_otsu",
        "loss_normalization": "align_config_default_l2_otsu_per_level",
        "mask_vol": "cyl",
        "optimizer_kind": "gn",
        "gn_damping": float(args.gn_damping),
        "quality_tier": str(args.quality_tier),
        "fallback_policy": str(args.fallback_policy),
        "fold_rigid_detector_grid": bool(getattr(args, "fold_rigid_detector_grid", True)),
        "pose_model": str(args.pose_model),
        "knot_spacing": int(args.knot_spacing),
        "pose_degree": int(args.pose_degree),
        "pose_bounds_profile": str(args.pose_bounds_profile),
        "pose_gauge_policy": "mean_translation_for_dx_dz",
        "final_candidate_policy": str(args.final_candidate_policy),
        "views_per_batch": int(args.views_per_batch),
        "gather_dtype": str(args.gather_dtype),
        "recon_positivity": bool(args.recon_positivity),
        "setup_outer_count_replay": "reference_stage_summary_counts",
        "pose_phi_bounds": _pose_phi_bounds(args),
        "pose_dx_dz_bounds": _pose_dx_dz_bounds(args),
        "pose_polish_bounds": _pose_polish_bounds(args),
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in REFERENCE_REGRESSION_CONTRACT.items()
        if actual.get(key) != expected
    }
    return {
        "schema": "tomojax.real_lamino_reference_regression_contract.v2",
        "source_script": "scripts/real_laminography/run_real_lamino_reference_regression.py",
        "expected": REFERENCE_REGRESSION_CONTRACT,
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


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


def _validate_bin_factor(value: object) -> int:
    try:
        factor = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bin factor must be an integer >= 1, got {value!r}") from exc
    if factor < 1:
        raise ValueError(f"bin factor must be an integer >= 1, got {value!r}")
    return factor


def _resolve_fixture_bin_factor(
    *,
    projection_shape: tuple[int, int, int],
    slab_nz: int,
    requested_bin_factor: int,
    smoke_shape: tuple[int, int, int] | None,
) -> int:
    factor = _validate_bin_factor(requested_bin_factor)
    if smoke_shape is None:
        return factor
    _target_views, target_nv, target_nu = smoke_shape
    if target_nv < 1 or target_nu < 1:
        raise ValueError(f"diagnostic shape must be positive, got {smoke_shape!r}")
    _n_views, nv, nu = projection_shape
    factor = max(factor, math.ceil(float(nv) / float(target_nv)))
    factor = max(factor, math.ceil(float(nu) / float(target_nu)))
    factor = max(factor, math.ceil(float(slab_nz) / float(max(1, target_nv))))
    return _validate_bin_factor(factor)


def _view_indices_for_smoke_shape(
    n_views: int,
    smoke_shape: tuple[int, int, int] | None,
) -> np.ndarray:
    if smoke_shape is None or int(smoke_shape[0]) >= int(n_views):
        return np.arange(int(n_views), dtype=np.int64)
    target = max(1, int(smoke_shape[0]))
    return np.unique(np.rint(np.linspace(0, int(n_views) - 1, target)).astype(np.int64))


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
    view_indices = _view_indices_for_smoke_shape(original_shape[0], args.smoke_shape)
    raw_selected = np.asarray(raw_projections[view_indices], dtype=np.float32)
    thetas_selected = np.asarray(thetas[view_indices], dtype=np.float32)
    bin_factor = _resolve_fixture_bin_factor(
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
        "detector_shift_bound_scale": float(_binned_pixel_scale(args)),
        "pose_dx_dz_bound_scale": float(_binned_pixel_scale(args)),
    }
    geometry_inputs = {"grid": grid, "detector": detector, "full_nz": int(original_full_nz)}
    return working_raw, thetas_selected, geometry_inputs, provenance


def _binned_pixel_scale(args: argparse.Namespace) -> float:
    fallback = getattr(args, "bin_factor", 1)
    return 1.0 / float(max(1, int(getattr(args, "effective_bin_factor", fallback))))


def _scaled_symmetric_bound(name: str, value: float, args: argparse.Namespace) -> str:
    scaled = float(value) * _binned_pixel_scale(args)
    return f"{name}={-scaled:.8g}:{scaled:.8g}"


def _setup_det_u_bounds(args: argparse.Namespace) -> str:
    return _scaled_symmetric_bound("det_u_px", 24.0, args)


def _pose_phi_bounds(args: argparse.Namespace) -> str:
    if str(args.pose_bounds_profile) == "wide":
        return "phi=-0.0872665:0.0872665"
    return "phi=-0.00872665:0.00872665"


def _pose_dx_dz_bounds(args: argparse.Namespace) -> str:
    value = 16.0 if str(args.pose_bounds_profile) == "wide" else 10.0
    dx = _scaled_symmetric_bound("dx", value, args)
    dz = _scaled_symmetric_bound("dz", value, args)
    return f"{dx},{dz}"


def _pose_polish_bounds(args: argparse.Namespace) -> str:
    dx_dz = _pose_dx_dz_bounds(args)
    if str(args.pose_bounds_profile) == "wide":
        return (
            "alpha=-0.0349066:0.0349066,beta=-0.0349066:0.0349066,"
            f"phi=-0.0872665:0.0872665,{dx_dz}"
        )
    return (
        "alpha=-0.00872665:0.00872665,beta=-0.00872665:0.00872665,"
        f"phi=-0.00872665:0.00872665,{dx_dz}"
    )


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


def _stage_record(root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    stage_dir = root / str(spec["stage"])
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {
            "label": spec["label"],
            "stage": spec["stage"],
            "status": "missing",
            "active_dofs": spec.get("active_dofs", []),
        }
    manifest = _read_json(manifest_path)
    return {
        "label": spec["label"],
        "stage": spec["stage"],
        "status": manifest.get("status"),
        "active_dofs": manifest.get("active_dofs", spec.get("active_dofs", [])),
        "bounds": manifest.get("bounds"),
        "levels": manifest.get("levels", []),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "stats_count": manifest.get("stats_count"),
        "geometry_calibration_state": manifest.get(
            "geometry_calibration_state",
            manifest.get("setup_state"),
        ),
        "params_summary": manifest.get("params_summary"),
        "reconstruction_loss": _stage_reconstruction_loss(manifest),
        "artifacts": _stage_artifacts(stage_dir, manifest),
        "summary_rows": _read_stage_summary(stage_dir / "stage_summary.csv"),
        "planned_after": manifest.get("planned_after"),
        "skip_reason": manifest.get("skip_reason"),
        "failure_provenance": manifest.get("failure_provenance"),
    }


def _reconstruction_comparison(root: Path) -> dict[str, Any]:
    baseline_manifest = _read_json(root / "00_baseline" / "stage_manifest.json")
    cor_manifest = _read_json(root / "06_cor_only_fista" / "stage_manifest.json")
    cor_loss = _loss_summary(cor_manifest.get("fista_info", {}))
    final_path = root / "05_final" / "stage_manifest.json"
    final_manifest = _read_json(final_path) if final_path.exists() else {"status": "missing"}
    final_info = final_manifest.get("recon_info", {})
    if not isinstance(final_info, Mapping):
        final_info = {}
    final_loss = _loss_summary(final_info)
    final_shape = final_manifest.get(
        "volume_shape",
        _read_json(root / "run_manifest.json").get("final_volume_shape"),
    )
    final_completed = final_manifest.get("status") == "completed" and final_loss["last"] is not None
    improvement = None
    relative = None
    if final_completed and cor_loss["last"] is not None:
        improvement = float(cor_loss["last"]) - float(final_loss["last"])
        relative = improvement / max(abs(float(cor_loss["last"])), 1.0e-12)
    return {
        "baseline": {
            "stage": "00_baseline",
            "volume_shape": baseline_manifest.get("volume_shape"),
            "loss": {"first": None, "last": None, "iters": 0},
            "role": "raw FBP visual/reference baseline",
        },
        "cor_only": {
            "stage": "06_cor_only_fista",
            "volume_shape": cor_manifest.get("volume_shape"),
            "loss": cor_loss,
            "regulariser": cor_manifest.get("fista_info", {}).get("regulariser"),
        },
        "final": {
            "stage": "05_final",
            "status": final_manifest.get("status"),
            "loss": final_loss,
            "volume_shape": final_shape,
            "regulariser": final_info.get("regulariser"),
        },
        "same_volume_shape": (
            (final_shape if final_completed else baseline_manifest.get("volume_shape"))
            == cor_manifest.get("volume_shape")
        ),
        "loss_improvement_abs": improvement,
        "loss_improvement_rel": relative,
        "full_staged_vs_cor_only_deferred": not final_completed,
    }


def _success_payload(
    reconstruction: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = {
        str(record.get("stage"))
        for record in records
        if str(record.get("status")) == "completed"
    }
    planned = {
        str(record.get("stage"))
        for record in records
        if str(record.get("status")) == "planned"
    }
    failed_or_skipped = [
        record
        for record in records
        if str(record.get("status")) in {"failed", "skipped"}
    ]
    partial_required = {"00_baseline", "01_setup_geometry/01_cor", "06_cor_only_fista"}
    full_required = {
        "00_baseline",
        "01_setup_geometry/01_cor",
        "01_setup_geometry/02_detector_roll",
        "01_setup_geometry/03_axis_direction",
        "02_pose_phi",
        "03_pose_dx_dz",
        "04_pose_polish",
        "05_final",
        "06_cor_only_fista",
    }
    final_loss = reconstruction["final"]["loss"]["last"]
    cor_loss = reconstruction["cor_only"]["loss"]["last"]
    full_complete = full_required <= completed
    full_improved = (
        full_complete
        and final_loss is not None
        and cor_loss is not None
        and float(final_loss) < float(cor_loss)
        and bool(reconstruction.get("same_volume_shape"))
    )
    phase = (
        "v2_full_staged_failed_validation"
        if failed_or_skipped
        else "v2_full_staged"
        if full_complete
        else "v2_cor_only_partial"
    )
    partial_complete = partial_required <= completed and cor_loss is not None
    validation_failed = bool(failed_or_skipped)
    passed = bool((full_improved if full_complete else partial_complete) and not validation_failed)
    reason = (
        "v2 staged path failed validation; final report uses the last finite candidate"
        if validation_failed
        else
        "v2 full staged reconstruction improves COR-only FISTA loss"
        if full_improved
        else "v2 full staged reconstruction did not improve COR-only FISTA loss"
        if full_complete
        else "v2 partial path completed baseline, det_u setup, and COR-only FISTA"
        if partial_complete
        else "v2 path is missing required baseline/det_u/COR-only evidence"
    )
    return {
        "passed": passed,
        "reason": reason,
        "phase": phase,
        "quality_kind": (
            "real_reconstruction_quality"
            if full_complete
            else "real_reconstruction_quality_partial_cor_only"
        ),
        "primary_metric": (
            "final_fista_last_loss_lt_cor_only_fista_last_loss"
            if full_complete
            else "cor_only_fista_loss_recorded_after_v2_det_u_stage"
        ),
        "required_stages_completed": sorted(
            (full_required if full_complete else partial_required) & completed
        ),
        "planned_stages": sorted(planned),
        "final_loss": final_loss,
        "cor_only_loss": cor_loss,
        "loss_improvement_abs": reconstruction.get("loss_improvement_abs"),
        "loss_improvement_rel": reconstruction.get("loss_improvement_rel"),
        "same_volume_shape": bool(reconstruction.get("same_volume_shape")),
        "full_staged_success_deferred": not full_complete,
        "validation_failed": validation_failed,
        "failed_or_skipped_stages": [
            {
                "stage": record.get("stage"),
                "status": record.get("status"),
                "failure_provenance": record.get("failure_provenance"),
                "skip_reason": record.get("skip_reason"),
            }
            for record in failed_or_skipped
        ],
    }


def _copy_publication_images(root: Path, out_dir: Path, *, full_completed: bool) -> dict[str, str]:
    pub_dir = out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    images = [
        ("before", "00_baseline", "orthos.png"),
        ("before_xy", "00_baseline", "aligned_xy_global_z209.png"),
        ("cor_only", "06_cor_only_fista", "orthos.png"),
        ("cor_only_xy", "06_cor_only_fista", "aligned_xy_global_z209.png"),
    ]
    if full_completed:
        images.extend(
            [
                ("full", "05_final", "orthos.png"),
                ("full_xy", "05_final", "aligned_xy_global_z209.png"),
                ("full_delta_xy", "05_final", "delta_xy_global_z209.png"),
            ]
        )
    copied: dict[str, str] = {}
    for label, stage, filename in images:
        source = root / stage / filename
        if not source.exists():
            raise FileNotFoundError(f"missing publication artifact {source}")
        dest = pub_dir / f"{label}_{filename}"
        shutil.copy2(source, dest)
        copied[label] = str(dest.resolve())
    return copied


def _write_residual_trace(path: Path, records: list[Mapping[str, Any]]) -> Path:
    fields = (
        "label",
        "stage",
        "status",
        "level_factor",
        "iteration",
        "loss_before",
        "loss_after",
        "accepted",
        "active_dofs",
        "elapsed_seconds",
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        summary_rows = record.get("summary_rows", [])
        if not summary_rows:
            loss = record.get("reconstruction_loss") or {}
            rows.append(
                {
                    "label": record.get("label"),
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                    "level_factor": "",
                    "iteration": "",
                    "loss_before": loss.get("first"),
                    "loss_after": loss.get("last"),
                    "accepted": "",
                    "active_dofs": ",".join(str(v) for v in record.get("active_dofs", [])),
                    "elapsed_seconds": record.get("elapsed_seconds"),
                }
            )
            continue
        active_dofs = ",".join(str(v) for v in record.get("active_dofs", []))
        rows.extend(
            {
                "label": record.get("label"),
                "stage": record.get("stage"),
                "status": record.get("status"),
                "level_factor": row.get("level_factor", ""),
                "iteration": row.get("outer_iter", row.get("outer_idx", "")),
                "loss_before": row.get("geometry_loss_before", row.get("loss_before", "")),
                "loss_after": row.get("geometry_loss_after", row.get("loss_after", "")),
                "accepted": row.get("geometry_accepted", row.get("accepted", "")),
                "active_dofs": row.get("active_dofs", active_dofs),
                "elapsed_seconds": row.get("elapsed_seconds", row.get("cumulative_time", "")),
            }
            for row in summary_rows
        )
    _write_csv(path, rows, fields)
    return path


def _write_geometry_trace(path: Path, records: list[Mapping[str, Any]]) -> Path:
    _write_json(
        path,
        {
            "schema": "tomojax.real_lamino_geometry_trace.v1",
            "stages": [
                {
                    "label": record.get("label"),
                    "stage": record.get("stage"),
                    "status": record.get("status"),
                    "active_dofs": record.get("active_dofs", []),
                    "bounds": record.get("bounds"),
                    "geometry_calibration_state": record.get("geometry_calibration_state"),
                    "params_summary": record.get("params_summary"),
                    "planned_after": record.get("planned_after"),
                }
                for record in records
            ],
        },
    )
    return path


def _write_reference_regression_audit(
    *,
    root: Path,
    out_dir: Path,
    reference_report: Path | None,
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    workflow = run_manifest.get("workflow", {})
    enabled = bool(isinstance(workflow, Mapping) and workflow.get("reference_regression"))
    if reference_report is None or not enabled:
        return {
            "payload": {
                "enabled": enabled,
                "status": "skipped",
                "reason": "reference-regression mode and reference report are required",
            },
            "artifacts": {},
        }
    reference_root = Path(reference_report).resolve().parents[1]
    rows = _build_reference_regression_rows(reference_root=reference_root, v2_root=root)
    csv_path = out_dir / "real_lamino_reference_regression_table.csv"
    fields = (
        "stage",
        "level_factor",
        "iteration",
        "reference_loss_before",
        "reference_loss_after",
        "current_loss_before",
        "current_loss_after",
        "loss_scale_ratio_after",
        "status",
        "notes",
    )
    _write_csv(csv_path, rows, fields)
    pose_scale_failures = [
        row
        for row in rows
        if str(row["stage"]).startswith(("02_pose", "03_pose", "04_pose"))
        and row["status"] == "loss_scale_mismatch"
    ]
    row_shape_failures = [
        row
        for row in rows
        if row["status"] in {"missing_reference_row", "missing_current_row"}
    ]
    contract = (
        workflow.get("reference_regression_contract", {})
        if isinstance(workflow, Mapping)
        else {}
    )
    payload = {
        "schema": "tomojax.real_lamino_reference_regression.v2",
        "enabled": True,
        "status": "failed" if pose_scale_failures or row_shape_failures else "recorded",
        "source_reference_run": str(reference_root),
        "source_script": "scripts/real_laminography/run_real_lamino_reference_regression.py",
        "contract": contract,
        "pose_loss_scale_failures": pose_scale_failures,
        "row_shape_failures": row_shape_failures,
        "stage_summaries": _reference_regression_stage_summaries(reference_root, root),
        "table_csv": str(csv_path.resolve()),
    }
    json_path = out_dir / "real_lamino_reference_regression.json"
    _write_json(json_path, payload)
    return {
        "payload": payload,
        "artifacts": {
            "reference_regression_table_csv": str(csv_path.resolve()),
            "reference_regression_json": str(json_path.resolve()),
        },
    }


def _build_reference_regression_rows(
    *,
    reference_root: Path,
    v2_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference_stage, current_stage in REFERENCE_REGRESSION_STAGE_MAP:
        if reference_stage in {"05_final", "06_cor_only_fista"}:
            reference_rows = _reconstruction_loss_rows_for_stage(reference_root / reference_stage)
            current_rows = _reconstruction_loss_rows_for_stage(v2_root / current_stage)
        else:
            reference_rows = _loss_rows_for_stage(reference_root / reference_stage)
            current_rows = _loss_rows_for_stage(v2_root / current_stage)
        keys = sorted(set(reference_rows) | set(current_rows), key=_reference_row_sort_key)
        for key in keys:
            reference = reference_rows.get(key, {})
            current = current_rows.get(key, {})
            ratio = _loss_scale_ratio(reference.get("loss_after"), current.get("loss_after"))
            status = "matched"
            notes = ""
            if not reference:
                status = "missing_reference_row"
            elif not current:
                status = "missing_current_row"
            elif (
                reference_stage.startswith(("02_pose", "03_pose", "04_pose"))
                and ratio is not None
                and (ratio > 10.0 or ratio < 0.1)
            ):
                status = "loss_scale_mismatch"
                notes = "pose loss scale differs by more than 10x"
            rows.append(
                {
                    "stage": current_stage,
                    "level_factor": key[0],
                    "iteration": key[1],
                    "reference_loss_before": reference.get("loss_before", ""),
                    "reference_loss_after": reference.get("loss_after", ""),
                    "current_loss_before": current.get("loss_before", ""),
                    "current_loss_after": current.get("loss_after", ""),
                    "loss_scale_ratio_after": "" if ratio is None else ratio,
                    "status": status,
                    "notes": notes,
                }
            )
    return rows


def _loss_rows_for_stage(stage_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_rows = _read_stage_summary(stage_dir / "stage_summary.csv")
    if summary_rows:
        return {
            (
                str(row.get("level_factor", "")),
                str(row.get("outer_iter", row.get("outer_idx", ""))),
            ): {
                "loss_before": row.get("geometry_loss_before", row.get("loss_before", "")),
                "loss_after": row.get("geometry_loss_after", row.get("loss_after", "")),
            }
            for row in summary_rows
        }
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {}
    loss = _stage_reconstruction_loss(_read_json(manifest_path))
    if not loss:
        return {}
    return {
        ("final", ""): {
            "loss_before": loss.get("first", ""),
            "loss_after": loss.get("last", ""),
        }
    }


def _reconstruction_loss_rows_for_stage(stage_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_path = stage_dir / "stage_manifest.json"
    if not manifest_path.exists():
        return {}
    loss = _stage_reconstruction_loss(_read_json(manifest_path))
    if not loss:
        return {}
    return {
        ("final", ""): {
            "loss_before": loss.get("first", ""),
            "loss_after": loss.get("last", ""),
        }
    }


def _reference_row_sort_key(key: tuple[str, str]) -> tuple[int, int, str, str]:
    level, iteration = key
    try:
        level_i = int(level)
    except ValueError:
        level_i = 10**9
    try:
        iter_i = int(iteration)
    except ValueError:
        iter_i = 10**9
    return (level_i, iter_i, level, iteration)


def _loss_scale_ratio(reference_loss: Any, current_loss: Any) -> float | None:
    try:
        reference = float(reference_loss)
        current = float(current_loss)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(reference) and np.isfinite(current)) or abs(reference) <= 1e-12:
        return None
    return float(current / reference)


def _reference_regression_stage_summaries(
    reference_root: Path,
    v2_root: Path,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for reference_stage, current_stage in REFERENCE_REGRESSION_STAGE_MAP:
        reference_manifest = _read_json(reference_root / reference_stage / "stage_manifest.json")
        current_manifest_path = v2_root / current_stage / "stage_manifest.json"
        current_manifest = (
            _read_json(current_manifest_path) if current_manifest_path.exists() else {}
        )
        summaries.append(
            {
                "stage": current_stage,
                "reference": _reference_manifest_summary(reference_manifest),
                "current": _reference_manifest_summary(current_manifest),
            }
        )
    return summaries


def _reference_manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": manifest.get("status"),
        "active_dofs": manifest.get("active_dofs"),
        "bounds": manifest.get("bounds"),
        "levels": manifest.get("levels"),
        "geometry_calibration_state": manifest.get("geometry_calibration_state"),
        "params_summary": manifest.get("params_summary"),
        "reconstruction_loss": _stage_reconstruction_loss(manifest),
    }


def _write_partial_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    success = summary["success"]
    reconstruction = summary["reconstruction_comparison"]
    lines = [
        "# Real Laminography Staged Report",
        "",
        f"- Reference target report: `{summary['reference_target_report']}`",
        f"- Phase complete: `{success['passed']}`",
        f"- Criterion: {success['reason']}",
        f"- Final loss: `{success['final_loss']}`",
        f"- COR-only loss: `{success['cor_only_loss']}`",
        f"- Full staged success deferred: `{success['full_staged_success_deferred']}`",
        "",
        "| Stage | Active DOFs | Status |",
        "|---|---|---|",
    ]
    for record in summary["staged_path"]:
        dofs = ",".join(str(v) for v in record.get("active_dofs", []))
        lines.append(f"| `{record['stage']}` | `{dofs}` | `{record.get('status')}` |")
    lines.extend(
        [
            "",
            "## Reconstruction Comparison",
            "",
            "| Comparator | First FISTA loss | Last FISTA loss | Effective iterations |",
            "|---|---:|---:|---:|",
            "| Baseline FBP |  |  | 0 |",
            "| COR-only | {first} | {last} | {iters} |".format(
                **reconstruction["cor_only"]["loss"]
            ),
            "| Full staged final | {first} | {last} | {iters} |".format(
                **reconstruction["final"]["loss"]
            ),
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Residual trace CSV: `{summary['artifacts']['residual_trace_csv']}`",
            f"- Geometry trace JSON: `{summary['artifacts']['geometry_trace_json']}`",
            f"- Publication image directory: `{summary['artifacts']['publication_dir']}`",
            "- Reference-regression table CSV: "
            f"`{summary['artifacts'].get('reference_regression_table_csv', '')}`",
            "",
            "Truth metrics are intentionally not used for this real-data gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_constraints() -> dict[str, Any]:
    return {
        "cor_grid_search_added": False,
        "sinogram_or_correlation_method_added": False,
        "sharpness_or_autofocus_method_added": False,
        "benchmark_only_knobs_promoted": False,
        "cor_only_role": "first v2 final reconstruction comparator",
    }


def _stage_artifacts(stage_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("artifacts", {})
    if not isinstance(raw, Mapping):
        return {}
    artifacts: dict[str, str] = {}
    for key, value in raw.items():
        artifacts[str(key)] = str(_resolve_artifact_path(stage_dir, value))
    return artifacts


def _resolve_artifact_path(stage_dir: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return stage_dir / candidate


def _stage_reconstruction_loss(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    info = manifest.get("recon_info", manifest.get("fista_info"))
    if not isinstance(info, Mapping):
        return None
    return _loss_summary(info)


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


def _parse_shape3(text: str) -> tuple[int, int, int]:
    parts = str(text).lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected projection shape as N,NV,NU or NxNVxNU, got {text!r}"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer projection shape {text!r}") from exc
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError(
            f"projection shape dimensions must be positive, got {text!r}"
        )
    return (shape[0], shape[1], shape[2])


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


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
