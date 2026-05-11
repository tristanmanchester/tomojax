#!/usr/bin/env python3
# pyright: reportAny=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Run the first v2 real-laminography COR MVP vertical slice."""

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

PARTIAL_STAGED_PATH: tuple[dict[str, Any], ...] = (
    {"label": "baseline", "stage": "00_baseline", "active_dofs": [], "status": "required"},
    {
        "label": "cor_detu",
        "stage": "01_setup_geometry/01_cor",
        "active_dofs": ["det_u_px"],
        "status": "required",
    },
    {
        "label": "detector_roll",
        "stage": "01_setup_geometry/02_detector_roll",
        "active_dofs": ["detector_roll_deg"],
        "status": "planned",
    },
    {
        "label": "axis_direction",
        "stage": "01_setup_geometry/03_axis_direction",
        "active_dofs": ["axis_rot_x_deg", "axis_rot_y_deg"],
        "status": "planned",
    },
    {"label": "pose_phi", "stage": "02_pose_phi", "active_dofs": ["phi"], "status": "planned"},
    {
        "label": "pose_dx_dz",
        "stage": "03_pose_dx_dz",
        "active_dofs": ["dx", "dz"],
        "status": "planned",
    },
    {
        "label": "pose_5dof_polish",
        "stage": "04_pose_polish",
        "active_dofs": ["alpha", "beta", "phi", "dx", "dz"],
        "status": "planned",
    },
    {
        "label": "final_reconstruction",
        "stage": "05_final",
        "active_dofs": ["detector_roll", "axis_direction", "pose_5dof"],
        "status": "planned",
    },
    {
        "label": "cor_only_comparator",
        "stage": "06_cor_only_fista",
        "active_dofs": ["det_u_px"],
        "status": "required",
    },
)


def main(argv: list[str] | None = None) -> int:
    """Run the v2 COR-MVP workflow."""
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
        summary = run_v2_cor_mvp(args, native=native, started_at=started)
        print(f"v2_cor_mvp_report: {summary['artifacts']['summary_json']}")
        print(f"phase_complete: {summary['success']['passed']}")
        return 0
    finally:
        monitor.close()


def run_v2_cor_mvp(
    args: argparse.Namespace,
    *,
    native: Any | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Execute baseline, det_u setup, COR-only FISTA, and write the partial report."""
    if native is None:
        native = _load_native_runner()
    run_root = Path(args.out)
    run_root.mkdir(parents=True, exist_ok=True)
    started = started_at or datetime.now().isoformat(timespec="seconds")
    ctx = native.RunContext(args)
    native._status(ctx.status_path, state="starting", started_at=started)
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
        projections, background_offsets = native._apply_projection_background(
            raw_projections,
            mode=str(args.projection_background),
            edge_px=int(args.background_edge_px),
        )
        np.save(
            run_root / "projection_background_offsets.npy",
            background_offsets.astype(np.float32),
        )
        n_views, nv, nu = projections.shape
        full_nz = int(nv)
        center_phys_z = native._global_z_to_phys(int(args.slab_center_z), full_nz=full_nz)
        grid = native.Grid(
            nx=int(nu),
            ny=int(nu),
            nz=int(args.slab_nz),
            vx=1.0,
            vy=1.0,
            vz=1.0,
            vol_center=(0.0, 0.0, center_phys_z),
        )
        detector = native.Detector(nu=int(nu), nv=int(nv), du=1.0, dv=1.0, det_center=(0.0, 0.0))
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
        run_manifest = {
            "schema": "tomojax.real_lamino_v2_cor_mvp_run.v1",
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
                "implemented_stages": [
                    "00_baseline",
                    "01_setup_geometry/01_cor",
                    "06_cor_only_fista",
                ],
                "planned_stages": [
                    "01_setup_geometry/02_detector_roll",
                    "01_setup_geometry/03_axis_direction",
                    "02_pose_phi",
                    "03_pose_dx_dz",
                    "04_pose_polish",
                    "05_final",
                ],
                "full_mvp_success_deferred": True,
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
            bounds="det_u_px=-24:24",
        )
        _write_planned_stage_manifests(run_root, native=native)
        cor_only = run_cor_only_fista(
            ctx,
            native=native,
            geometry=geometry,
            grid=grid,
            detector=detector,
            projections=projections,
            full_nz=full_nz,
            setup_state=setup_state,
        )
        completed = datetime.now().isoformat(timespec="seconds")
        final_payload = {
            "status": "completed",
            "completed_at": completed,
            "stage_records": [
                {"stage": "00_baseline", "volume_shape": list(baseline.shape)},
                {
                    "stage": "01_setup_geometry/01_cor",
                    "stats_count": len(stats),
                    "geometry_calibration_state": setup_state.to_calibration_state().to_dict(),
                },
                {"stage": "06_cor_only_fista", "volume_shape": list(cor_only.shape)},
            ],
            "final_setup_estimates": setup_state.to_calibration_state().to_dict(),
            "final_volume_shape": list(cor_only.shape),
        }
        native._write_json(run_root / "run_manifest.json", {**run_manifest, **final_payload})
        native._status(ctx.status_path, state="completed", stage="complete", **final_payload)
        return build_v2_cor_mvp_report(
            run_root,
            out_dir=run_root / "v2_cor_mvp_report",
            reference_report=Path(args.reference_report) if args.reference_report else None,
        )
    except Exception as exc:
        native._status(ctx.status_path, state="failed", stage="error", error=repr(exc))
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
            views_per_batch=None
            if int(ctx.args.views_per_batch) == 0
            else max(1, int(ctx.args.views_per_batch)),
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


def build_v2_cor_mvp_report(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    reference_report: Path | None = None,
) -> dict[str, Any]:
    """Write a report matching the real-MVP shape for the partial COR v2 path."""
    root = Path(run_dir)
    out = out_dir or root / "v2_cor_mvp_report"
    out.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(root / "run_manifest.json")
    status = _read_json(root / "status.json") if (root / "status.json").exists() else {}
    records = [_stage_record(root, spec) for spec in PARTIAL_STAGED_PATH]
    reconstruction = _partial_reconstruction_comparison(root)
    success = _partial_success_payload(reconstruction, records)
    publication = _copy_partial_publication_images(root, out)
    residual_trace = _write_residual_trace(out / "real_mvp_residual_trace.csv", records)
    geometry_trace = _write_geometry_trace(out / "real_mvp_geometry_trace.json", records)
    summary: dict[str, Any] = {
        "schema": "tomojax.real_lamino_v2_cor_mvp_report.v1",
        "contract_compatible_with": "tomojax.real_lamino_mvp_report.v1",
        "run_dir": str(root),
        "reference_target_report": str(reference_report) if reference_report else None,
        "reference_case": root.name,
        "success": success,
        "quality_basis": {
            "kind": "real_reconstruction_quality_partial_cor_mvp",
            "primary_metric": "cor_only_fista_loss_recorded_after_v2_det_u_stage",
            "full_mvp_primary_metric": "final_fista_last_loss_lt_cor_only_fista_last_loss",
            "full_mvp_success_deferred": True,
            "truth_metrics": "not_applicable_real_data",
            "synthetic_truth_metrics_allowed": False,
        },
        "staged_path": records,
        "reconstruction_comparison": reconstruction,
        "publication_artifacts": publication,
        "provenance": {
            "input": run_manifest.get("input"),
            "backend": run_manifest.get("backend"),
            "devices": run_manifest.get("devices"),
            "final_volume_shape": run_manifest.get("final_volume_shape"),
            "final_setup_estimates": run_manifest.get("final_setup_estimates"),
            "status_completed_at": status.get("completed_at", run_manifest.get("completed_at")),
        },
        "method_constraints": _method_constraints(),
        "artifacts": {
            "summary_json": str((out / "real_mvp_summary.json").resolve()),
            "summary_md": str((out / "real_mvp_summary.md").resolve()),
            "residual_trace_csv": str(residual_trace.resolve()),
            "geometry_trace_json": str(geometry_trace.resolve()),
            "publication_dir": str((out / "publication").resolve()),
        },
    }
    _write_json(out / "real_mvp_summary.json", summary)
    _write_partial_markdown(out / "real_mvp_summary.md", summary)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v2 baseline + COR/det_u + COR-only FISTA for real laminography."
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
    parser.add_argument("--levels-setup", nargs="+", type=int, default=[8, 4, 2])
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
    parser.add_argument("--gn-damping", type=float, default=1e-3)
    parser.add_argument("--filter", dest="filter_name", default="ramp")
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-rel", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--snapshot-max-cols", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.smoke):
        args.slab_nz = min(int(args.slab_nz), 48)
        args.levels_setup = [8]
        args.outer_iters = min(int(args.outer_iters), 1)
        args.recon_iters = min(int(args.recon_iters), 3)
        args.tv_prox_iters = min(int(args.tv_prox_iters), 2)
        args.snapshot_max_cols = min(int(args.snapshot_max_cols), 4)
        if int(args.views_per_batch) == 0:
            args.views_per_batch = 16
    return args


def _load_native_runner() -> Any:
    path = Path(__file__).with_name("run_real_lamino_native_setup_pose_256.py")
    spec = importlib.util.spec_from_file_location("run_real_lamino_native_setup_pose_256", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load native runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_planned_stage_manifests(root: Path, *, native: Any) -> None:
    for spec in PARTIAL_STAGED_PATH:
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
    }


def _partial_reconstruction_comparison(root: Path) -> dict[str, Any]:
    baseline_manifest = _read_json(root / "00_baseline" / "stage_manifest.json")
    cor_manifest = _read_json(root / "06_cor_only_fista" / "stage_manifest.json")
    cor_loss = _loss_summary(cor_manifest.get("fista_info", {}))
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
            "status": "planned",
            "loss": {"first": None, "last": None, "iters": 0},
            "volume_shape": None,
        },
        "same_volume_shape": (
            baseline_manifest.get("volume_shape") == cor_manifest.get("volume_shape")
        ),
        "full_staged_vs_cor_only_deferred": True,
    }


def _partial_success_payload(
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
    required = {"00_baseline", "01_setup_geometry/01_cor", "06_cor_only_fista"}
    required_complete = required <= completed
    cor_loss = reconstruction["cor_only"]["loss"]["last"]
    passed = bool(required_complete and cor_loss is not None)
    return {
        "passed": passed,
        "reason": (
            "v2 COR-MVP partial path completed baseline, det_u setup, and COR-only FISTA"
            if passed
            else "v2 COR-MVP partial path is missing required baseline/det_u/COR-only evidence"
        ),
        "phase": "v2_cor_mvp_partial",
        "required_stages_completed": sorted(required & completed),
        "planned_stages": sorted(planned),
        "cor_only_loss": cor_loss,
        "same_volume_shape": bool(reconstruction.get("same_volume_shape")),
        "full_mvp_success_deferred": True,
    }


def _copy_partial_publication_images(root: Path, out_dir: Path) -> dict[str, str]:
    pub_dir = out_dir / "publication"
    pub_dir.mkdir(parents=True, exist_ok=True)
    images = (
        ("before", "00_baseline", "orthos.png"),
        ("before_xy", "00_baseline", "aligned_xy_global_z209.png"),
        ("cor_only", "06_cor_only_fista", "orthos.png"),
        ("cor_only_xy", "06_cor_only_fista", "aligned_xy_global_z209.png"),
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


def _write_partial_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    success = summary["success"]
    reconstruction = summary["reconstruction_comparison"]
    lines = [
        "# Real Laminography v2 COR-MVP Report",
        "",
        f"- Reference target report: `{summary['reference_target_report']}`",
        f"- Phase complete: `{success['passed']}`",
        f"- Criterion: {success['reason']}",
        f"- COR-only loss: `{success['cor_only_loss']}`",
        f"- Full staged success deferred: `{success['full_mvp_success_deferred']}`",
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
            "| Full staged final |  |  | 0 |",
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Residual trace CSV: `{summary['artifacts']['residual_trace_csv']}`",
            f"- Geometry trace JSON: `{summary['artifacts']['geometry_trace_json']}`",
            f"- Publication image directory: `{summary['artifacts']['publication_dir']}`",
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
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = stage_dir / candidate
        artifacts[str(key)] = str(candidate)
    return artifacts


def _stage_reconstruction_loss(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    info = manifest.get("recon_info", manifest.get("fista_info"))
    if not isinstance(info, Mapping):
        return None
    return _loss_summary(info)


def _loss_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    losses = info.get("loss", [])
    if not isinstance(losses, list) or not losses:
        return {"first": None, "last": None, "iters": 0}
    return {
        "first": float(losses[0]),
        "last": float(losses[-1]),
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
