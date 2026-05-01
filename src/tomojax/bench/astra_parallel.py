from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
from tomojax.data.simulate import SimConfig, make_phantom
from tomojax.recon.fbp import (
    _default_fbp_scale,
    _run_parallel_fbp_direct_pallas,
    _supports_parallel_fbp_z_integer,
    fbp,
)

try:
    import astra
except Exception:  # pragma: no cover - optional benchmark dependency
    astra = None

try:
    from tomojax.core.pallas_projector import (
        forward_project_views_T_pallas,
    )
except Exception:  # pragma: no cover - optional experimental backend
    forward_project_views_T_pallas = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional runtime dependency
    pynvml = None


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


class GpuMemorySampler:
    def __init__(self, interval_sec: float = 0.005):
        self.interval_sec = float(interval_sec)
        self.pid = os.getpid()
        self.samples_mb: list[float] = []
        self.device_samples_mb: list[float] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None

    def __enter__(self):
        if pynvml is None:
            self.error = "pynvml_unavailable"
            return self
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._sample_once()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self.error = f"{type(exc).__name__}: {exc}"
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample_once()
        try:
            if pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def _sample_once(self) -> None:
        if pynvml is None or self._handle is None:
            return
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.device_samples_mb.append(float(mem.used) / (1024.0 * 1024.0))

            total = 0.0
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if int(proc.pid) == self.pid:
                    used = getattr(proc, "usedGpuMemory", None)
                    if used is not None:
                        total += float(used) / (1024.0 * 1024.0)
            self.samples_mb.append(total)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"

    def summary(self) -> dict[str, Any]:
        process = self.samples_mb
        device = self.device_samples_mb

        def stats(samples: list[float]) -> dict[str, float | None]:
            if not samples:
                return {"before_mb": None, "peak_mb": None, "after_mb": None}
            return {
                "before_mb": float(samples[0]),
                "peak_mb": float(max(samples)),
                "after_mb": float(samples[-1]),
            }

        return {
            "process": stats(process),
            "device": stats(device),
            "sample_count": len(process),
            "sampling_interval_sec": self.interval_sec,
            "error": self.error,
        }


def _time_call(
    fn,
    *,
    warmup: int = 1,
    repeat: int = 3,
) -> tuple[Any, list[float], list[dict[str, Any]]]:
    value = None
    for _ in range(max(0, int(warmup))):
        value = fn()
        _block(value)

    times: list[float] = []
    memory: list[dict[str, Any]] = []
    for _ in range(int(repeat)):
        gc.collect()
        with GpuMemorySampler() as sampler:
            start = time.perf_counter()
            value = fn()
            _block(value)
            times.append(time.perf_counter() - start)
        memory.append(sampler.summary())
    return value, times, memory


def _time_call_with_cold(
    fn,
    *,
    warmup: int = 1,
    repeat: int = 3,
) -> tuple[Any, float, list[float], list[dict[str, Any]]]:
    gc.collect()
    start = time.perf_counter()
    value = fn()
    _block(value)
    cold_sec = time.perf_counter() - start
    for _ in range(max(0, int(warmup))):
        value = fn()
        _block(value)

    times: list[float] = []
    memory: list[dict[str, Any]] = []
    for _ in range(int(repeat)):
        gc.collect()
        with GpuMemorySampler() as sampler:
            start = time.perf_counter()
            value = fn()
            _block(value)
            times.append(time.perf_counter() - start)
        memory.append(sampler.summary())
    return value, cold_sec, times, memory


def _time_summary(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {
            "runs": 0,
            "median_sec": None,
            "mean_sec": None,
            "min_sec": None,
            "max_sec": None,
        }
    return {
        "runs": len(times),
        "median_sec": float(statistics.median(times)),
        "mean_sec": float(statistics.fmean(times)),
        "min_sec": float(min(times)),
        "max_sec": float(max(times)),
    }


def _memory_summary(samples: list[dict[str, Any]]) -> dict[str, float | int | None]:
    peaks: list[float] = []
    deltas: list[float] = []
    for sample in samples:
        proc = sample.get("process", {})
        peak = proc.get("peak_mb")
        before = proc.get("before_mb")
        if peak is not None:
            peaks.append(float(peak))
        if peak is not None and before is not None:
            deltas.append(max(0.0, float(peak) - float(before)))
    return {
        "runs": len(samples),
        "peak_process_mb": float(max(peaks)) if peaks else None,
        "median_peak_process_mb": float(statistics.median(peaks)) if peaks else None,
        "peak_delta_process_mb": float(max(deltas)) if deltas else None,
        "median_peak_delta_process_mb": float(statistics.median(deltas)) if deltas else None,
    }


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _metrics(recon: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    recon32 = np.asarray(recon, dtype=np.float32)
    truth32 = np.asarray(truth, dtype=np.float32)
    diff = recon32 - truth32
    mse = float(np.mean(diff * diff))
    rmse = float(math.sqrt(mse))
    peak = float(max(np.max(truth32) - np.min(truth32), 1e-12))
    psnr = float(20.0 * math.log10(peak / max(rmse, 1e-12)))
    return {
        "mse": mse,
        "rmse": rmse,
        "psnr_db": psnr,
        "min": float(np.min(recon32)),
        "max": float(np.max(recon32)),
        "mean": float(np.mean(recon32)),
    }


def _projection_metrics(proj: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    proj32 = np.asarray(proj, dtype=np.float32)
    ref32 = np.asarray(ref, dtype=np.float32)
    diff = proj32 - ref32
    denom = float(np.linalg.norm(ref32.ravel())) or 1.0
    return {
        "mse_vs_tomojax": float(np.mean(diff * diff)),
        "rmse_vs_tomojax": float(math.sqrt(float(np.mean(diff * diff)))),
        "max_abs_vs_tomojax": float(np.max(np.abs(diff))),
        "relative_l2_vs_tomojax": float(np.linalg.norm(diff.ravel()) / denom),
        "min": float(np.min(proj32)),
        "max": float(np.max(proj32)),
        "mean": float(np.mean(proj32)),
    }


def _make_volume(args: argparse.Namespace) -> np.ndarray:
    if args.input is not None:
        with np.load(args.input) as data:
            return np.asarray(data["volume"], dtype=np.float32)

    cfg = SimConfig(
        nx=args.size,
        ny=args.size,
        nz=args.size,
        nu=args.detector,
        nv=args.detector,
        n_views=args.views,
        rotation_deg=180.0,
        geometry="parallel",
        phantom=args.phantom,
        n_cubes=args.n_cubes,
        n_spheres=args.n_spheres,
        min_size=args.min_size,
        max_size=args.max_size,
        seed=args.seed,
    )
    return np.asarray(make_phantom(cfg), dtype=np.float32)


def _make_geometry(size: int, detector: int, views: int) -> tuple[Grid, Detector, ParallelGeometry, np.ndarray]:
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    det = Detector(detector, detector, 1.0, 1.0, det_center=(0.0, 0.0))
    angles = np.linspace(0.0, 180.0, int(views), endpoint=False).astype(np.float32)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=angles)
    return grid, det, geom, angles


def _tomojax_forward(
    volume: Any,
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry,
    views: int,
) -> jnp.ndarray:
    vol = jnp.asarray(volume, dtype=jnp.float32)
    poses = stack_view_poses(geom, views)
    det_grid = get_detector_grid_device(det)

    @jax.jit
    def project_all(vol_in):
        return jax.vmap(
            lambda T: forward_project_view_T(
                T,
                grid,
                det,
                vol_in,
                use_checkpoint=True,
                gather_dtype="fp32",
                det_grid=det_grid,
            )
        )(poses)

    return project_all(vol)


@functools.lru_cache(maxsize=8)
def _cached_tomojax_pallas_forward_callable(
    *,
    nx: int,
    ny: int,
    nz: int,
    vx: float,
    vy: float,
    vz: float,
    nu: int,
    nv: int,
    du: float,
    dv: float,
    det_center: tuple[float, float],
    thetas_deg: tuple[float, ...],
):
    grid = Grid(nx, ny, nz, vx, vy, vz)
    det = Detector(nu, nv, du, dv, det_center=det_center)
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas_deg)
    poses = stack_view_poses(geom, len(thetas_deg))
    det_grid = get_detector_grid_device(det)
    tile_shape = (16, 4) if max(int(nu), int(nv)) <= 64 else (8, 4)
    num_warps = 2 if tile_shape == (16, 4) else 1

    @jax.jit
    def project(vol_in: jnp.ndarray) -> jnp.ndarray:
        return forward_project_views_T_pallas(
            poses,
            grid,
            det,
            vol_in,
            gather_dtype="fp32",
            det_grid=det_grid,
            tile_shape=tile_shape,
            num_warps=num_warps,
            kernel_variant="auto",
            layout_variant="detector_vu",
            state_mode="inline",
        )

    return project


def _tomojax_pallas_forward(
    volume: Any,
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry,
    views: int,
) -> jnp.ndarray:
    if forward_project_views_T_pallas is None:
        raise RuntimeError("Pallas projector is not importable in this TomoJAX checkout")
    vol = jnp.asarray(volume, dtype=jnp.float32)
    project = _cached_tomojax_pallas_forward_callable(
        nx=int(grid.nx),
        ny=int(grid.ny),
        nz=int(grid.nz),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        nu=int(det.nu),
        nv=int(det.nv),
        du=float(det.du),
        dv=float(det.dv),
        det_center=(float(det.det_center[0]), float(det.det_center[1])),
        thetas_deg=tuple(float(theta) for theta in np.asarray(geom.thetas_deg[:views])),
    )
    return project(vol)


def _make_tomojax_pallas_forward_runner(
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry,
    views: int,
):
    if forward_project_views_T_pallas is None:
        raise RuntimeError("Pallas projector is not importable in this TomoJAX checkout")
    project = _cached_tomojax_pallas_forward_callable(
        nx=int(grid.nx),
        ny=int(grid.ny),
        nz=int(grid.nz),
        vx=float(grid.vx),
        vy=float(grid.vy),
        vz=float(grid.vz),
        nu=int(det.nu),
        nv=int(det.nv),
        du=float(det.du),
        dv=float(det.dv),
        det_center=(float(det.det_center[0]), float(det.det_center[1])),
        thetas_deg=tuple(float(theta) for theta in np.asarray(geom.thetas_deg[:views])),
    )

    def run(volume: Any) -> jnp.ndarray:
        if getattr(volume, "dtype", None) == jnp.dtype(jnp.float32):
            return project(volume)
        return project(jnp.asarray(volume, dtype=jnp.float32))

    return run


def _tomojax_fbp(
    projections: Any,
    grid: Grid,
    det: Detector,
    geom: ParallelGeometry,
    views_per_batch: int,
) -> jnp.ndarray:
    if jax.default_backend() == "gpu" and _supports_parallel_fbp_z_integer(grid, det):
        poses = stack_view_poses(geom, int(projections.shape[0]))
        return _run_parallel_fbp_direct_pallas(
            poses,
            jnp.asarray(projections, dtype=jnp.float32),
            grid=grid,
            detector=det,
            filter_name="ramp",
        ) * jnp.float32(_default_fbp_scale(int(projections.shape[0])))
    recon = fbp(
        geom,
        grid,
        det,
        jnp.asarray(projections, dtype=jnp.float32),
        filter_name="ramp",
        views_per_batch=views_per_batch,
        gather_dtype="fp32",
    )
    return recon


def _rotz_jax(theta: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.asarray(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )


def _differentiability_guard(size: int = 24) -> dict[str, Any]:
    """Exercise the differentiable ray-marched projector on a tiny alignment-like loss."""
    grid = Grid(size, size, size, 1.0, 1.0, 1.0)
    detector = Detector(size, size, 1.0, 1.0, det_center=(0.0, 0.0))
    det_grid = get_detector_grid_device(detector)
    volume = jnp.zeros((size, size, size), dtype=jnp.float32)
    volume = volume.at[size // 5 : size // 5 + 5, size // 3 : size // 3 + 4, size // 2 : size // 2 + 3].set(1.0)
    volume = volume.at[size // 2 + 2 : size // 2 + 6, size // 5 : size // 5 + 3, size // 4 : size // 4 + 4].set(0.65)
    target_theta = jnp.float32(0.08)

    def project(theta: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            _rotz_jax(theta),
            grid,
            detector,
            volume,
            use_checkpoint=True,
            gather_dtype="fp32",
            det_grid=det_grid,
        )

    target = jax.lax.stop_gradient(project(target_theta))

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        residual = project(theta) - target
        return jnp.mean(residual * residual)

    grad_fn = jax.jit(jax.value_and_grad(loss))
    # Compile/warm once outside the measured section.
    warm_loss, warm_grad = grad_fn(jnp.float32(0.0))
    warm_loss.block_until_ready()
    warm_grad.block_until_ready()
    start = time.perf_counter()
    initial_loss, grad = grad_fn(jnp.float32(0.0))
    initial_loss.block_until_ready()
    grad.block_until_ready()
    grad_runtime_sec = time.perf_counter() - start
    initial_loss_f = float(initial_loss)
    grad_f = float(grad)
    candidate_steps = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    best_step = None
    best_theta = None
    best_loss = None
    for step in candidate_steps:
        theta_next = jnp.float32(0.0 - step * grad_f)
        stepped_loss = loss(theta_next)
        stepped_loss.block_until_ready()
        stepped_loss_f = float(stepped_loss)
        if best_loss is None or stepped_loss_f < best_loss:
            best_step = float(step)
            best_theta = float(theta_next)
            best_loss = stepped_loss_f

    finite = math.isfinite(initial_loss_f) and math.isfinite(grad_f) and (
        best_loss is not None and math.isfinite(best_loss)
    )
    grad_abs = abs(grad_f)
    reduced = best_loss is not None and best_loss < initial_loss_f
    passed = bool(finite and grad_abs > 1e-8 and reduced)
    return {
        "status": "pass" if passed else "fail",
        "size": int(size),
        "target_theta_rad": float(target_theta),
        "initial_theta_rad": 0.0,
        "initial_loss": initial_loss_f,
        "gradient": grad_f,
        "gradient_abs": grad_abs,
        "gradient_finite": bool(math.isfinite(grad_f)),
        "loss_finite": bool(math.isfinite(initial_loss_f)),
        "best_step": best_step,
        "stepped_theta_rad": best_theta,
        "stepped_loss": best_loss,
        "loss_reduced": bool(reduced),
        "grad_runtime_sec": float(grad_runtime_sec),
    }


def _astra_forward_3d(volume: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    nz = int(volume.shape[2])
    ny = int(volume.shape[1])
    nx = int(volume.shape[0])
    vol_geom = astra.create_vol_geom(ny, nx, nz)
    # TomoJAX's positive object rotation maps to the opposite ASTRA projection angle.
    proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, nz, nx, -angles_rad)

    # ASTRA 3D data is indexed as (z, y, x). TomoJAX volume is (x, y, z).
    astra_volume = np.transpose(volume, (2, 1, 0)).astype(np.float32, copy=False)
    proj_id, proj = astra.create_sino3d_gpu(astra_volume, proj_geom, vol_geom)
    try:
        # ASTRA projection data is (det_row, angle, det_col) -> TomoJAX (angle, v, u).
        return np.transpose(np.asarray(proj, dtype=np.float32), (1, 0, 2))
    finally:
        astra.data3d.delete(proj_id)


def _astra_fbp_slice(volume: np.ndarray, angles_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = map(int, volume.shape)
    rec = np.empty_like(volume, dtype=np.float32)
    sino = np.empty((len(angles_rad), nz, nx), dtype=np.float32)
    vol_geom = astra.create_vol_geom(ny, nx)
    # Keep the slice-wise FBP convention aligned with the 3D forward comparison.
    proj_geom = astra.create_proj_geom("parallel", 1.0, nx, -angles_rad)
    projector_id = astra.create_projector("cuda", proj_geom, vol_geom)
    try:
        for iz in range(nz):
            # ASTRA 2D data is (rows=y, cols=x). TomoJAX slice is (x, y).
            astra_slice = np.transpose(volume[:, :, iz]).astype(np.float32, copy=False)
            sino_id, sino_2d = astra.create_sino(astra_slice, projector_id)
            rec_id = astra.data2d.create("-vol", vol_geom)
            alg_cfg = astra.astra_dict("FBP_CUDA")
            alg_cfg["ProjectionDataId"] = sino_id
            alg_cfg["ReconstructionDataId"] = rec_id
            alg_cfg["FilterType"] = "ram-lak"
            alg_id = astra.algorithm.create(alg_cfg)
            try:
                astra.algorithm.run(alg_id)
                rec_slice = astra.data2d.get(rec_id)
                rec[:, :, iz] = np.transpose(np.asarray(rec_slice, dtype=np.float32))
                sino[:, iz, :] = np.asarray(sino_2d, dtype=np.float32)
            finally:
                astra.algorithm.delete(alg_id)
                astra.data2d.delete([sino_id, rec_id])
    finally:
        astra.projector.delete(projector_id)
    return sino, rec


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _operation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "tomojax_forward": ("Forward projection", "TomoJAX", "3D parallel JAX"),
        "tomojax_pallas_forward": (
            "Forward projection",
            "TomoJAX",
            "batched Pallas sinogram",
        ),
        "astra_parallel3d_forward": ("Forward projection", "ASTRA", "parallel3d CUDA"),
        "tomojax_fbp": ("FBP reconstruction", "TomoJAX", "3D adjoint FBP"),
        "astra_slice_fbp": ("FBP reconstruction", "ASTRA", "slice-wise FBP_CUDA"),
    }
    rows: list[dict[str, Any]] = []
    for key, (operation, library, method) in labels.items():
        timing = report["timing_summary"][key]
        memory = report["gpu_memory_summary_mb"][key]
        rows.append(
            {
                "operation": operation,
                "library": library,
                "method": method,
                "runs": timing["runs"],
                "cold_sec": report.get("cold_timing_summary", {}).get(key, {}).get("seconds"),
                "median_sec": timing["median_sec"],
                "mean_sec": timing["mean_sec"],
                "min_sec": timing["min_sec"],
                "max_sec": timing["max_sec"],
                "peak_process_gpu_mb": memory["peak_process_mb"],
                "peak_delta_process_gpu_mb": memory["peak_delta_process_mb"],
            }
        )
    return rows


def _quality_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    forward = report["forward_projection"]["astra_parallel3d_vs_tomojax"]
    pallas = report["forward_projection"].get("tomojax_pallas_vs_tomojax")
    recon = report["reconstruction"]
    rows = [
        {
            "comparison": "TomoJAX Pallas forward vs TomoJAX JAX forward",
            "mse": pallas["mse_vs_tomojax"] if pallas else None,
            "rmse": pallas["rmse_vs_tomojax"] if pallas else None,
            "psnr_db": None,
            "relative_l2": pallas["relative_l2_vs_tomojax"] if pallas else None,
            "max_abs": pallas["max_abs_vs_tomojax"] if pallas else None,
        },
        {
            "comparison": "ASTRA parallel3d forward vs TomoJAX forward",
            "mse": forward["mse_vs_tomojax"],
            "rmse": forward["rmse_vs_tomojax"],
            "psnr_db": None,
            "relative_l2": forward["relative_l2_vs_tomojax"],
            "max_abs": forward["max_abs_vs_tomojax"],
        },
        {
            "comparison": "TomoJAX FBP vs phantom",
            "mse": recon["tomojax_fbp_vs_truth"]["mse"],
            "rmse": recon["tomojax_fbp_vs_truth"]["rmse"],
            "psnr_db": recon["tomojax_fbp_vs_truth"]["psnr_db"],
            "relative_l2": None,
            "max_abs": None,
        },
        {
            "comparison": "ASTRA slice FBP vs phantom",
            "mse": recon["astra_slice_fbp_vs_truth"]["mse"],
            "rmse": recon["astra_slice_fbp_vs_truth"]["rmse"],
            "psnr_db": recon["astra_slice_fbp_vs_truth"]["psnr_db"],
            "relative_l2": None,
            "max_abs": None,
        },
        {
            "comparison": "ASTRA slice FBP vs TomoJAX FBP",
            "mse": recon["astra_slice_fbp_vs_tomojax_fbp"]["mse"],
            "rmse": recon["astra_slice_fbp_vs_tomojax_fbp"]["rmse"],
            "psnr_db": recon["astra_slice_fbp_vs_tomojax_fbp"]["psnr_db"],
            "relative_l2": None,
            "max_abs": None,
        },
        {
            "comparison": "TomoJAX direct FBP vs TomoJAX generic FBP",
            "mse": recon["tomojax_direct_fbp_vs_generic_fbp"]["mse_vs_tomojax"],
            "rmse": recon["tomojax_direct_fbp_vs_generic_fbp"]["rmse_vs_tomojax"],
            "psnr_db": None,
            "relative_l2": recon["tomojax_direct_fbp_vs_generic_fbp"][
                "relative_l2_vs_tomojax"
            ],
            "max_abs": recon["tomojax_direct_fbp_vs_generic_fbp"]["max_abs_vs_tomojax"],
        },
    ]
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    op_rows = _operation_rows(report)
    quality_rows = _quality_rows(report)
    speedups = report["speedups"]
    config = report["config"]

    lines = [
        "# TomoJAX vs ASTRA Parallel Benchmark",
        "",
        f"- Size: `{config['size']}^3`",
        f"- Detector: `{config['detector']} x {config['detector']}`",
        f"- Views: `{config['views']}` over `180 deg`",
        f"- Warmup runs: `{config['warmup']}`",
        f"- Measured repeats: `{config['repeat']}`",
        f"- ASTRA forward: `parallel3d` CUDA",
        f"- ASTRA FBP: slice-wise `FBP_CUDA`",
        f"- FDK: intentionally unused",
        "",
        "## Speedups",
        "",
        "| Comparison | Speedup |",
        "|---|---:|",
        f"| ASTRA forward vs TomoJAX forward | {_fmt(speedups['astra_forward_vs_tomojax_forward_median'])}x |",
        f"| Pallas forward vs TomoJAX JAX forward | {_fmt(speedups['pallas_forward_vs_tomojax_forward_median'])}x |",
        f"| ASTRA forward vs Pallas forward | {_fmt(speedups['astra_forward_vs_pallas_forward_median'])}x |",
        f"| ASTRA slice FBP vs TomoJAX FBP | {_fmt(speedups['astra_slice_fbp_vs_tomojax_fbp_median'])}x |",
        "",
        "## Timing And Memory",
        "",
        "| Operation | Library | Method | Cold sec | Warm median sec | Warm mean sec | Peak process GPU MB | Peak delta GPU MB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in op_rows:
        lines.append(
            "| {operation} | {library} | {method} | {cold_sec} | {median_sec} | {mean_sec} | "
            "{peak_process_gpu_mb} | {peak_delta_process_gpu_mb} |".format(
                operation=row["operation"],
                library=row["library"],
                method=row["method"],
                cold_sec=_fmt(row.get("cold_sec")),
                median_sec=_fmt(row["median_sec"]),
                mean_sec=_fmt(row["mean_sec"]),
                peak_process_gpu_mb=_fmt(row["peak_process_gpu_mb"]),
                peak_delta_process_gpu_mb=_fmt(row["peak_delta_process_gpu_mb"]),
            )
        )

    lines.extend(
        [
            "",
            "## Quality",
            "",
            "| Comparison | MSE | RMSE | PSNR dB | Relative L2 | Max abs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            "| {comparison} | {mse} | {rmse} | {psnr_db} | {relative_l2} | {max_abs} |".format(
                comparison=row["comparison"],
                mse=_fmt(row["mse"]),
                rmse=_fmt(row["rmse"]),
                psnr_db=_fmt(row["psnr_db"]),
                relative_l2=_fmt(row["relative_l2"]),
                max_abs=_fmt(row["max_abs"]),
            )
        )
    guard = report.get("differentiability_guard")
    if guard is not None:
        lines.extend(
            [
                "",
                "## Differentiability Guard",
                "",
                "| Check | Value |",
                "|---|---:|",
                f"| Status | {guard['status']} |",
                f"| Guard size | {_fmt(guard['size'])}^3 |",
                f"| Initial loss | {_fmt(guard['initial_loss'])} |",
                f"| Gradient | {_fmt(guard['gradient'])} |",
                f"| Gradient abs | {_fmt(guard['gradient_abs'])} |",
                f"| One-step loss | {_fmt(guard['stepped_loss'])} |",
                f"| Loss reduced | {guard['loss_reduced']} |",
                f"| Best step | {_fmt(guard['best_step'])} |",
                f"| Grad runtime | {_fmt(guard['grad_runtime_sec'])} sec |",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TomoJAX parallel forward/FBP with ASTRA CUDA baselines."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arrays-out", type=Path, default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--detector", type=int, default=128)
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--phantom", default="random_shapes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cubes", type=int, default=8)
    parser.add_argument("--n-spheres", type=int, default=7)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=32)
    parser.add_argument("--tomojax-views-per-batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-pallas", action="store_true")
    parser.add_argument("--no-diff-guard", action="store_true")
    parser.add_argument("--diff-guard-size", type=int, default=24)
    parser.add_argument("--note", default="")
    parser.add_argument("--git-branch", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--quality-csv", type=Path, default=None)
    args = parser.parse_args()
    if astra is None:
        raise RuntimeError("ASTRA is required for the ASTRA comparison benchmark")

    volume = _make_volume(args)
    size = int(volume.shape[0])
    if volume.shape != (size, size, size):
        raise ValueError(f"expected cubic volume, got {volume.shape}")

    grid, det, geom, angles_deg = _make_geometry(size, args.detector, args.views)
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    astra_cuda = bool(astra.use_cuda()) if hasattr(astra, "use_cuda") else None
    device = {
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "astra_version": getattr(astra, "__version__", "unknown"),
        "astra_cuda": astra_cuda,
        "astra_gpu_info": astra.get_gpu_info() if astra_cuda and hasattr(astra, "get_gpu_info") else None,
    }

    differentiability_guard = (
        None if args.no_diff_guard else _differentiability_guard(args.diff_guard_size)
    )
    if differentiability_guard is not None and differentiability_guard["status"] != "pass":
        raise RuntimeError(
            "Differentiability guard failed: "
            + json.dumps(differentiability_guard, sort_keys=True)
        )

    tomojax_volume = jnp.asarray(volume, dtype=jnp.float32)
    tomojax_proj, tomojax_forward_cold, tomojax_forward_times, tomojax_forward_memory = (
        _time_call_with_cold(
        lambda: _tomojax_forward(tomojax_volume, grid, det, geom, args.views),
        warmup=args.warmup,
        repeat=args.repeat,
        )
    )
    if args.no_pallas:
        pallas_proj = None
        pallas_forward_cold = None
        pallas_forward_times: list[float] = []
        pallas_forward_memory: list[dict[str, Any]] = []
    else:
        tomojax_pallas_forward = _make_tomojax_pallas_forward_runner(
            grid, det, geom, args.views
        )
        pallas_proj, pallas_forward_cold, pallas_forward_times, pallas_forward_memory = (
            _time_call_with_cold(
            lambda: tomojax_pallas_forward(tomojax_volume),
            warmup=args.warmup,
            repeat=args.repeat,
            )
        )
    astra_proj, astra_forward_cold, astra_forward_times, astra_forward_memory = _time_call_with_cold(
        lambda: _astra_forward_3d(volume, angles_rad),
        warmup=args.warmup,
        repeat=args.repeat,
    )

    tomojax_recon, tomojax_fbp_cold, tomojax_fbp_times, tomojax_fbp_memory = _time_call_with_cold(
        lambda: _tomojax_fbp(
            tomojax_proj,
            grid,
            det,
            geom,
            args.tomojax_views_per_batch,
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    generic_recon = np.asarray(
        fbp(
            geom,
            grid,
            det,
            jnp.asarray(tomojax_proj, dtype=jnp.float32),
            filter_name="ramp",
            views_per_batch=args.tomojax_views_per_batch,
            gather_dtype="fp32",
            det_grid=get_detector_grid_device(det),
        ),
        dtype=np.float32,
    )
    (astra_sino, astra_recon), astra_fbp_cold, astra_fbp_times, astra_fbp_memory = (
        _time_call_with_cold(
        lambda: _astra_fbp_slice(volume, angles_rad),
        warmup=args.warmup,
        repeat=args.repeat,
        )
    )

    cold_timing_summary = {
        "tomojax_forward": {"seconds": tomojax_forward_cold},
        "tomojax_pallas_forward": {"seconds": pallas_forward_cold},
        "astra_parallel3d_forward": {"seconds": astra_forward_cold},
        "tomojax_fbp": {"seconds": tomojax_fbp_cold},
        "astra_slice_fbp": {"seconds": astra_fbp_cold},
    }
    timing_seconds = {
        "tomojax_forward": tomojax_forward_times,
        "tomojax_pallas_forward": pallas_forward_times,
        "astra_parallel3d_forward": astra_forward_times,
        "tomojax_fbp": tomojax_fbp_times,
        "astra_slice_fbp": astra_fbp_times,
    }
    gpu_memory_mb = {
        "tomojax_forward": tomojax_forward_memory,
        "tomojax_pallas_forward": pallas_forward_memory,
        "astra_parallel3d_forward": astra_forward_memory,
        "tomojax_fbp": tomojax_fbp_memory,
        "astra_slice_fbp": astra_fbp_memory,
    }
    timing_summary = {key: _time_summary(value) for key, value in timing_seconds.items()}
    gpu_memory_summary = {key: _memory_summary(value) for key, value in gpu_memory_mb.items()}
    speedups = {
        "astra_forward_vs_tomojax_forward_median": _ratio(
            timing_summary["tomojax_forward"]["median_sec"],
            timing_summary["astra_parallel3d_forward"]["median_sec"],
        ),
        "pallas_forward_vs_tomojax_forward_median": _ratio(
            timing_summary["tomojax_forward"]["median_sec"],
            timing_summary["tomojax_pallas_forward"]["median_sec"],
        ),
        "astra_forward_vs_pallas_forward_median": _ratio(
            timing_summary["tomojax_pallas_forward"]["median_sec"],
            timing_summary["astra_parallel3d_forward"]["median_sec"],
        ),
        "astra_slice_fbp_vs_tomojax_fbp_median": _ratio(
            timing_summary["tomojax_fbp"]["median_sec"],
            timing_summary["astra_slice_fbp"]["median_sec"],
        ),
    }

    report = {
        "benchmark": "tomojax_vs_astra_parallel",
        "notes": [
            "ASTRA forward uses parallel3d CUDA.",
            "ASTRA FBP uses 2D FBP_CUDA independently for each z slice.",
            "FDK_CUDA is intentionally not used because the benchmark is parallel beam.",
        ],
        "config": {
            "size": size,
            "detector": args.detector,
            "views": args.views,
            "rotation_deg": 180.0,
            "phantom": args.phantom,
            "seed": args.seed,
            "tomojax_views_per_batch": args.tomojax_views_per_batch,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "include_pallas": not args.no_pallas,
            "include_differentiability_guard": not args.no_diff_guard,
            "differentiability_guard_size": args.diff_guard_size,
        },
        "experiment": {
            "note": args.note,
            "git_branch": args.git_branch,
            "git_commit": args.git_commit,
        },
        "device": device,
        "timing_seconds": timing_seconds,
        "timing_summary": timing_summary,
        "cold_timing_summary": cold_timing_summary,
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_memory_summary_mb": gpu_memory_summary,
        "speedups": speedups,
        "differentiability_guard": differentiability_guard,
        "forward_projection": {
            "tomojax": _projection_metrics(tomojax_proj, tomojax_proj),
            "tomojax_pallas_vs_tomojax": (
                _projection_metrics(pallas_proj, tomojax_proj)
                if pallas_proj is not None
                else None
            ),
            "astra_parallel3d_vs_tomojax": _projection_metrics(astra_proj, tomojax_proj),
        },
        "reconstruction": {
            "tomojax_fbp_vs_truth": _metrics(tomojax_recon, volume),
            "tomojax_direct_fbp_vs_generic_fbp": _projection_metrics(
                tomojax_recon,
                generic_recon,
            ),
            "astra_slice_fbp_vs_truth": _metrics(astra_recon, volume),
            "astra_slice_fbp_vs_tomojax_fbp": _metrics(astra_recon, tomojax_recon),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.summary_csv is not None:
        _write_csv(args.summary_csv, _operation_rows(report))
    if args.quality_csv is not None:
        _write_csv(args.quality_csv, _quality_rows(report))
    if args.summary_md is not None:
        _write_markdown(args.summary_md, report)
    if args.arrays_out is not None:
        _write_npz(
            args.arrays_out,
            volume=volume,
            tomojax_proj=tomojax_proj,
            tomojax_pallas_proj=(
                pallas_proj if pallas_proj is not None else np.empty((0,), dtype=np.float32)
            ),
            astra_proj=astra_proj,
            astra_slice_sino=astra_sino,
            tomojax_recon=tomojax_recon,
            astra_recon=astra_recon,
            angles_deg=angles_deg,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
