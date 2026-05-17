#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
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

from tomojax.align.api import (
    AlignConfig,
    AlignmentState,
    BaseGeometryArrays,
    GeometryCalibrationState,
    L2LossSpec,
    PoseState,
    SetupGeometryState,
    align,
    apply_alignment_state,
    geometry_with_axis_state,
)
from tomojax.bench import (
    apply_real_lamino_projection_background,
    parse_real_lamino_z_range,
    real_lamino_global_z_to_phys,
    real_lamino_pose_params_summary,
    real_lamino_projection_stats,
    save_real_lamino_z_stack,
    save_uint8_png,
    validate_real_lamino_loaded_input,
)
from tomojax.bench.real_laminography_runtime import (
    RealLaminoGpuMonitor as GpuMonitor,
    append_real_lamino_csv as _append_csv,
    real_lamino_commit_info as _commit_info,
    relative_l2 as _relative_l2,
    select_real_lamino_views as _select_views,
    timed_repeats as _timed_repeats,
    update_real_lamino_status as _status,
    write_real_lamino_json as _write_json,
)
from tomojax.core.projector import get_detector_grid_device
from tomojax.geometry import Detector, Grid, LaminographyGeometry
from tomojax.io import load_real_laminography_input
from tomojax.recon.fista_tv_core import (
    FistaCoreConfig,
    fista_tv_core_arrays,
    projection_loss_arrays,
)
from tomojax.recon.multires import bin_projections, scale_detector, scale_grid

_apply_projection_background = apply_real_lamino_projection_background
_projection_stats = real_lamino_projection_stats
_save_png = save_uint8_png
_save_z_stack = save_real_lamino_z_stack
_validate_loaded_input = validate_real_lamino_loaded_input


def _setup_to_alignment_state(setup: GeometryCalibrationState, n_views: int) -> AlignmentState:
    return AlignmentState(
        setup=SetupGeometryState.from_degrees(
            det_u_px=setup.det_u_px,
            det_v_px=setup.det_v_px,
            detector_roll_deg=setup.detector_roll_deg,
            axis_rot_x_deg=setup.axis_rot_x_deg,
            axis_rot_y_deg=setup.axis_rot_y_deg,
            nominal_axis_unit=setup.nominal_axis_unit,
        ),
        pose=PoseState(jnp.zeros((int(n_views), 5), dtype=jnp.float32)),
        volume=None,
    )


def _make_core_cfg(args: argparse.Namespace, *, backend: str) -> FistaCoreConfig:
    return FistaCoreConfig(
        iters=int(args.recon_iters),
        lambda_tv=float(args.lambda_tv),
        regulariser="huber_tv",
        huber_delta=float(args.huber_delta),
        L=float(args.lipschitz),
        checkpoint_projector=True,
        projector_unroll=1,
        gather_dtype=str(args.gather_dtype),
        views_per_batch=max(1, int(args.views_per_batch)),
        forward_projector=backend,
        backprojector=backend,
        pallas_tile_shape=(int(args.tile_u), int(args.tile_v)),
        pallas_num_warps=int(args.num_warps),
        compute_iteration_loss=False,
        compute_final_data_loss=False,
        compute_final_regulariser_value=False,
    )


def _run_residual_probe(
    *,
    run_root: Path,
    status_path: Path,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projections: jnp.ndarray,
    volume: jnp.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _status(status_path, state="running", stage="01_residual_probe")
    stage_dir = run_root / "01_residual_probe"
    stage_dir.mkdir(parents=True, exist_ok=True)
    jax_cfg = _make_core_cfg(args, backend="jax")
    pallas_cfg = _make_core_cfg(args, backend="pallas")

    def make_fn(cfg: FistaCoreConfig):
        return jax.jit(
            lambda vol: projection_loss_arrays(
                T_all=T_all,
                grid=grid,
                detector=detector,
                volume=vol,
                det_grid=det_grid,
                projections=projections,
                cfg=cfg,
            )
        )

    jax_fn = make_fn(jax_cfg)
    pallas_fn = make_fn(pallas_cfg)
    out_jax, timing_jax = _timed_repeats(
        name="projection_loss_jax",
        fn=lambda: jax_fn(volume),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
    )
    out_pallas, timing_pallas = _timed_repeats(
        name="projection_loss_pallas_forced",
        fn=lambda: pallas_fn(volume),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
    )
    payload = {
        "stage": "01_residual_probe",
        "requested_backend": "pallas",
        "actual_pallas_path": "fista_tv_core.projection_loss_arrays/forward_projector=pallas",
        "jax_loss": float(np.asarray(out_jax)),
        "pallas_loss": float(np.asarray(out_pallas)),
        "relative_loss_delta": abs(float(np.asarray(out_pallas)) - float(np.asarray(out_jax)))
        / max(abs(float(np.asarray(out_jax))), 1e-8),
        "timings": [timing_jax, timing_pallas],
        "speedup_warm_median_pallas_vs_jax": timing_jax["median_seconds"]
        / max(timing_pallas["median_seconds"], 1e-12),
    }
    _write_json(stage_dir / "results.json", payload)
    return payload


def _run_fista_probe(
    *,
    run_root: Path,
    status_path: Path,
    T_all: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projections: jnp.ndarray,
    x0: jnp.ndarray,
    args: argparse.Namespace,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    _status(status_path, state="running", stage="02_fista_core_probe")
    stage_dir = run_root / "02_fista_core_probe"
    stage_dir.mkdir(parents=True, exist_ok=True)
    jax_cfg = _make_core_cfg(args, backend="jax")
    pallas_cfg = _make_core_cfg(args, backend="pallas")

    def make_fn(cfg: FistaCoreConfig):
        return jax.jit(
            lambda x: fista_tv_core_arrays(
                x0=x,
                T_all=T_all,
                det_grid=det_grid,
                projections=projections,
                grid=grid,
                detector=detector,
                cfg=cfg,
            ).x
        )

    jax_fn = make_fn(jax_cfg)
    pallas_fn = make_fn(pallas_cfg)
    out_jax, timing_jax = _timed_repeats(
        name="fista_core_jax",
        fn=lambda: jax_fn(x0),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
    )
    out_pallas, timing_pallas = _timed_repeats(
        name="fista_core_pallas_forced",
        fn=lambda: pallas_fn(x0),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
    )
    out_pallas_np = np.asarray(out_pallas, dtype=np.float32)
    np.save(stage_dir / "pallas_fista_volume.npy", out_pallas_np)
    _save_png(stage_dir / "pallas_fista_xy_mid.png", out_pallas_np[:, :, out_pallas_np.shape[2] // 2].T)
    payload = {
        "stage": "02_fista_core_probe",
        "requested_backend": "pallas",
        "actual_pallas_path": (
            "fista_tv_core_arrays/forward_projector=pallas/backprojector=pallas/"
            "forward_project_loss_and_grad_T_pallas"
        ),
        "relative_l2_pallas_vs_jax": _relative_l2(out_pallas, out_jax),
        "timings": [timing_jax, timing_pallas],
        "speedup_warm_median_pallas_vs_jax": timing_jax["median_seconds"]
        / max(timing_pallas["median_seconds"], 1e-12),
    }
    _write_json(stage_dir / "results.json", payload)
    return jnp.asarray(out_pallas, dtype=jnp.float32), payload


def _run_alignment_probe(
    *,
    run_root: Path,
    status_path: Path,
    geometry: LaminographyGeometry,
    grid: Grid,
    detector: Detector,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    projections: jnp.ndarray,
    x0: jnp.ndarray,
    params5: jnp.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _status(status_path, state="running", stage="03_alignment_probe")
    stage_dir = run_root / "03_alignment_probe"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cfg = AlignConfig(
        outer_iters=int(args.align_outer_iters),
        recon_iters=int(args.align_recon_iters),
        lambda_tv=float(args.lambda_tv),
        regulariser="huber_tv",
        huber_delta=float(args.huber_delta),
        recon_algo="fista",
        views_per_batch=max(1, int(args.views_per_batch)),
        checkpoint_projector=True,
        gather_dtype=str(args.gather_dtype),
        projector_backend="pallas",
        opt_method="gn",
        gn_damping=float(args.gn_damping),
        optimise_dofs=tuple(str(name) for name in args.align_dofs),
        bounds="phi=-0.0872665:0.0872665,dx=-16:16,dz=-16:16",
        pose_model="per_view",
        gauge_fix="mean_translation",
        loss=L2LossSpec(),
        mask_vol="cyl",
        recon_positivity=False,
        seed_translations=False,
        early_stop=False,
        log_summary=True,
        log_compact=True,
    )
    t0 = time.perf_counter()
    try:
        x_aligned, params_out, info = align(
            geometry,
            grid,
            detector,
            projections,
            cfg=cfg,
            init_x=x0,
            init_params5=params5,
            det_grid_override=det_grid,
        )
        jax.block_until_ready(x_aligned)
        elapsed = time.perf_counter() - t0
        x_np = np.asarray(x_aligned, dtype=np.float32)
        params_np = np.asarray(params_out, dtype=np.float32)
        np.save(stage_dir / "aligned_volume.npy", x_np)
        np.save(stage_dir / "params5.npy", params_np)
        _save_png(stage_dir / "aligned_xy_mid.png", x_np[:, :, x_np.shape[2] // 2].T)
        payload = {
            "stage": "03_alignment_probe",
            "status": "completed",
            "requested_backend": "pallas",
            "expected_contract": (
                "reconstruction should use Pallas huber FISTA core; pose objective may "
                "fall back to JAX because it requires differentiable projector semantics"
            ),
            "elapsed_seconds": float(elapsed),
            "info": info,
            "params_summary": real_lamino_pose_params_summary(params_np),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        payload = {
            "stage": "03_alignment_probe",
            "status": "unsupported",
            "requested_backend": "pallas",
            "elapsed_seconds": float(elapsed),
            "error": repr(exc),
            "capability_boundary": (
                "The current production align() path cannot yet force the Pallas "
                "FISTA loss/gradient helper because the jitted alignment wrapper "
                "passes detector-grid arrays dynamically; the Pallas helper requires "
                "None or the canonical eager detector grid."
            ),
        }
    _write_json(stage_dir / "align_info.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused real-laminography probe that forces the new Pallas-backed paths."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--setup-state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tilt-deg", type=float, default=34.4)
    parser.add_argument("--tilt-about", choices=["x", "z"], default="x")
    parser.add_argument("--flip-u", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-v", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transpose-detector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-projection-shape", default=None)
    parser.add_argument("--preview-z", type=int, default=209)
    parser.add_argument("--slab-center-z", type=int, default=209)
    parser.add_argument("--slab-nz", type=int, default=96)
    parser.add_argument("--stack-z-range", default="198:220")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--max-views", type=int, default=180)
    parser.add_argument("--projection-background", choices=["none", "view_median", "edge_median"], default="edge_median")
    parser.add_argument("--background-edge-px", type=int, default=16)
    parser.add_argument("--views-per-batch", type=int, default=8)
    parser.add_argument("--gather-dtype", default="bf16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--recon-iters", type=int, default=4)
    parser.add_argument("--align-recon-iters", type=int, default=4)
    parser.add_argument("--align-outer-iters", type=int, default=1)
    parser.add_argument("--align-dofs", nargs="+", default=["phi", "dx", "dz"])
    parser.add_argument("--lambda-tv", type=float, default=0.008)
    parser.add_argument("--huber-delta", type=float, default=1e-2)
    parser.add_argument("--lipschitz", type=float, default=100.0)
    parser.add_argument("--gn-damping", type=float, default=1e-3)
    parser.add_argument("--tile-u", type=int, default=16)
    parser.add_argument("--tile-v", type=int, default=4)
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--skip-alignment", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.out)
    if run_root.exists() and any(run_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists and is not empty: {run_root}")
    if run_root.exists() and bool(args.overwrite):
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    (run_root / "runner.pid").write_text(str(os.getpid()) + "\n")
    monitor = GpuMonitor(run_root / "gpu_memory.csv")
    monitor.start()
    started = datetime.now().isoformat(timespec="seconds")
    _status(status_path, state="starting", started_at=started)
    try:
        loaded = load_real_laminography_input(
            Path(args.input),
            flip_u=bool(args.flip_u),
            flip_v=bool(args.flip_v),
            transpose_detector=bool(args.transpose_detector),
        )
        raw = loaded.projections
        thetas = loaded.thetas_deg
        expected = None
        if args.expected_projection_shape:
            parts = [int(part) for part in str(args.expected_projection_shape).replace("x", ",").split(",")]
            expected = tuple(parts)
        raw, thetas = _validate_loaded_input(raw, thetas, expected_projection_shape=expected)
        projections, background_offsets = _apply_projection_background(
            raw,
            mode=str(args.projection_background),
            edge_px=int(args.background_edge_px),
        )
        projections, thetas, view_indices = _select_views(
            projections,
            thetas,
            max_views=int(args.max_views),
        )
        n_views, nv, nu = projections.shape
        full_nz = int(nv)
        center_phys_z = real_lamino_global_z_to_phys(int(args.slab_center_z), full_nz=full_nz)
        grid = Grid(
            nx=int(nu),
            ny=int(nu),
            nz=int(args.slab_nz),
            vx=1.0,
            vy=1.0,
            vz=1.0,
            vol_center=(0.0, 0.0, center_phys_z),
        )
        detector = Detector(nu=int(nu), nv=int(nv), du=1.0, dv=1.0, det_center=(0.0, 0.0))
        geometry = LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(args.tilt_deg),
            tilt_about=str(args.tilt_about),
        )
        setup_payload = json.loads(Path(args.setup_state).read_text())
        setup_state = GeometryCalibrationState.from_checkpoint(setup_payload, geometry)

        factor = max(1, int(args.factor))
        g = scale_grid(grid, factor)
        d = scale_detector(detector, factor)
        y = bin_projections(jnp.asarray(projections, dtype=jnp.float32), factor)
        geometry_level = LaminographyGeometry(
            grid=g,
            detector=d,
            thetas_deg=thetas,
            tilt_deg=float(args.tilt_deg),
            tilt_about=str(args.tilt_about),
        )
        geom_eff = geometry_with_axis_state(geometry_level, g, d, setup_state)
        pallas_det_grid = get_detector_grid_device(d)
        base = BaseGeometryArrays.from_geometry(geometry_level, d, level_factor=factor)
        effective = apply_alignment_state(base, _setup_to_alignment_state(setup_state, n_views))
        T_all = effective.pose_stack
        params5 = jnp.zeros((int(n_views), 5), dtype=jnp.float32)
        x0 = jnp.zeros((int(g.nx), int(g.ny), int(g.nz)), dtype=jnp.float32)
        z_range = parse_real_lamino_z_range(str(args.stack_z_range))

        _write_json(
            run_root / "run_manifest.json",
            {
                "status": "running",
                "started_at": started,
                "input": str(args.input),
                "setup_state": str(args.setup_state),
                "purpose": "force Pallas-backed residual/FISTA/alignment probes on real laminography data",
                "input_shape_after_view_selection": list(projections.shape),
                "raw_projection_stats": _projection_stats(raw),
                "working_projection_stats": _projection_stats(projections),
                "view_indices": [int(v) for v in view_indices.tolist()],
                "factor": int(factor),
                "level_shapes": {
                    "grid": g.to_dict(),
                    "detector": d.to_dict(),
                    "projections": list(np.asarray(y).shape),
                },
                "setup_calibration_state": setup_state.to_calibration_state().to_dict(),
            "pallas_forcing": {
                "residual_probe": "projection_loss_arrays with forward_projector=pallas",
                "fista_probe": "fista_tv_core_arrays with forward_projector=pallas and backprojector=pallas",
                "alignment_probe": (
                    "AlignConfig(projector_backend='pallas', regulariser='huber_tv'); "
                    "pose objective may fall back to JAX for differentiability"
                ),
                "detector_grid_policy": (
                    "Pallas currently requires the canonical detector grid, so these probes "
                    "force get_detector_grid_device(detector) instead of the calibrated "
                    "setup-state detector grid. Axis/setup pose state is still applied."
                ),
            },
                "worktree": _commit_info(Path.cwd()),
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "z_stack_range": list(z_range),
            },
        )

        residual = _run_residual_probe(
            run_root=run_root,
            status_path=status_path,
            T_all=T_all,
            grid=g,
            detector=d,
            det_grid=pallas_det_grid,
            projections=y,
            volume=x0,
            args=args,
        )
        fista_volume, fista_result = _run_fista_probe(
            run_root=run_root,
            status_path=status_path,
            T_all=T_all,
            grid=g,
            detector=d,
            det_grid=pallas_det_grid,
            projections=y,
            x0=x0,
            args=args,
        )
        alignment_result: dict[str, Any] | None = None
        if not bool(args.skip_alignment):
            alignment_result = _run_alignment_probe(
                run_root=run_root,
                status_path=status_path,
                geometry=geom_eff,
                grid=g,
                detector=d,
                det_grid=pallas_det_grid,
                projections=y,
                x0=fista_volume,
                params5=params5,
                args=args,
            )
        fista_np = np.asarray(fista_volume, dtype=np.float32)
        _save_z_stack(
            run_root / "pallas_fista_z_stack.png",
            fista_np,
            grid=g,
            full_nz=full_nz,
            z_range=z_range,
            max_cols=6,
        )
        summary_rows = []
        for stage in (residual, fista_result):
            for timing in stage["timings"]:
                summary_rows.append(
                    {
                        "stage": stage["stage"],
                        "name": timing["name"],
                        "median_seconds": timing["median_seconds"],
                        "mean_seconds": timing["mean_seconds"],
                        "cold_seconds": timing["cold_seconds"],
                        "requested_backend": stage["requested_backend"],
                    }
                )
        for row in summary_rows:
            _append_csv(
                run_root / "summary.csv",
                row,
                ["stage", "name", "median_seconds", "mean_seconds", "cold_seconds", "requested_backend"],
            )
        completed = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "residual_probe": residual,
            "fista_probe": fista_result,
            "alignment_probe": alignment_result,
            "summary_csv": str(run_root / "summary.csv"),
            "gpu_memory_csv": str(run_root / "gpu_memory.csv"),
        }
        _write_json(run_root / "results.json", completed)
        _write_json(run_root / "run_manifest.json", {**json.loads((run_root / "run_manifest.json").read_text()), **completed})
        _status(status_path, state="completed", stage="complete", **completed)
        return 0
    except Exception as exc:
        _status(status_path, state="failed", stage="error", error=repr(exc))
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    raise SystemExit(main())
