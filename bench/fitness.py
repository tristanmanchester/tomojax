#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = BENCH_ROOT / "profiles"
FIXTURES_DIR = BENCH_ROOT / "fixtures"
DATA_DIR = BENCH_ROOT / "data"
OUT_DIR = BENCH_ROOT / "out"
MB = 1024.0 * 1024.0


def _repo_pythonpath() -> None:
    src = REPO_ROOT / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed TomoJAX benchmark profiles.")
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile name (e.g. speed_recon_small) or path to a YAML profile.",
    )
    parser.add_argument("--out", required=True, help="Path to the metrics JSON to write.")
    parser.add_argument(
        "--profile-root",
        default=str(PROFILES_DIR),
        help="Directory used to resolve named profiles (default: bench/profiles).",
    )
    return parser.parse_args()


def _resolve_profile_path(profile_arg: str, profile_root: str) -> Path:
    candidate = Path(profile_arg)
    if candidate.exists():
        return candidate.resolve()
    root = Path(profile_root)
    if candidate.suffix in {".yaml", ".yml"}:
        resolved = root / candidate.name
    else:
        resolved = root / f"{profile_arg}.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark profile not found: {profile_arg}")
    return resolved.resolve()


def _load_profile(profile_path: Path) -> dict[str, Any]:
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict):
        raise TypeError(f"Profile must be a YAML mapping: {profile_path}")
    name = profile.get("name") or profile_path.stem
    profile["name"] = str(name)
    return profile


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class FixtureBundle:
    name: str
    grid: dict[str, Any]
    detector: dict[str, Any]
    geometry_type: str
    geometry_meta: dict[str, Any] | None
    thetas_deg: np.ndarray
    volume: np.ndarray
    projections: np.ndarray
    align_params: np.ndarray | None = None

    @property
    def shape_summary(self) -> dict[str, Any]:
        return {
            "volume_shape": [int(v) for v in self.volume.shape],
            "projection_shape": [int(v) for v in self.projections.shape],
            "n_views": int(self.projections.shape[0]),
        }


def _save_fixture(bundle: FixtureBundle, path: Path) -> None:
    payload: dict[str, Any] = {
        "meta_json": json.dumps(
            {
                "name": bundle.name,
                "grid": bundle.grid,
                "detector": bundle.detector,
                "geometry_type": bundle.geometry_type,
                "geometry_meta": bundle.geometry_meta,
            },
            sort_keys=True,
        ),
        "thetas_deg": np.asarray(bundle.thetas_deg, dtype=np.float32),
        "volume": np.asarray(bundle.volume, dtype=np.float32),
        "projections": np.asarray(bundle.projections, dtype=np.float32),
    }
    if bundle.align_params is not None:
        payload["align_params"] = np.asarray(bundle.align_params, dtype=np.float32)
    np.savez_compressed(path, **payload)



def _load_fixture(path: Path) -> FixtureBundle:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"].item()))
    align_params = None
    if "align_params" in data:
        align_params = np.asarray(data["align_params"], dtype=np.float32)
    return FixtureBundle(
        name=str(meta.get("name") or path.stem),
        grid=dict(meta["grid"]),
        detector=dict(meta["detector"]),
        geometry_type=str(meta.get("geometry_type", "parallel")),
        geometry_meta=meta.get("geometry_meta"),
        thetas_deg=np.asarray(data["thetas_deg"], dtype=np.float32),
        volume=np.asarray(data["volume"], dtype=np.float32),
        projections=np.asarray(data["projections"], dtype=np.float32),
        align_params=align_params,
    )


def _configure_environment(profile: dict[str, Any]) -> dict[str, Any]:
    env_updates = dict(profile.get("env") or {})
    jax_cache_dir = profile.get("jax_cache_dir") or env_updates.get("JAX_COMPILATION_CACHE_DIR")
    if jax_cache_dir:
        env_updates["JAX_COMPILATION_CACHE_DIR"] = str(jax_cache_dir)
    env_updates.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for key, value in env_updates.items():
        if value is None:
            continue
        os.environ[str(key)] = str(value)
    if jax_cache_dir:
        _ensure_dir(Path(str(jax_cache_dir)))
    return env_updates


def _configure_jax_cache(profile: dict[str, Any], jax_module: Any) -> None:
    cache_dir = profile.get("jax_cache_dir") or os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return
    jax_module.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax_module.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax_module.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax_module.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )


@dataclass
class ImportedModules:
    jax: Any
    jnp: Any
    Grid: Any
    Detector: Any
    ParallelGeometry: Any
    LaminographyGeometry: Any
    SimConfig: Any
    simulate: Any
    fbp: Any
    fista_tv: Any
    AlignConfig: Any
    align: Any
    align_multires: Any
    se3_from_5d: Any
    forward_project_view_T: Any
    get_detector_grid_device: Any
    loss_metrics_abs: Any
    loss_metrics_relative: Any
    loss_metrics_gf: Any
    gt_projection_helper: Any



def _import_modules(profile: dict[str, Any]) -> ImportedModules:
    _repo_pythonpath()
    import jax
    import jax.numpy as jnp

    _configure_jax_cache(profile, jax)

    from tomojax.core.geometry import (
        Grid,
        Detector,
        ParallelGeometry,
        LaminographyGeometry,
    )
    from tomojax.data.simulate import SimConfig, simulate
    from tomojax.recon.fbp import fbp
    from tomojax.recon.fista_tv import fista_tv
    from tomojax.align.pipeline import AlignConfig, align, align_multires
    from tomojax.align.parametrizations import se3_from_5d
    from tomojax.core.projector import forward_project_view_T, get_detector_grid_device
    from tomojax.cli.loss_bench import (
        _metrics_abs,
        _metrics_relative,
        _metrics_gauge_fixed,
        _gt_projection_mse,
    )

    return ImportedModules(
        jax=jax,
        jnp=jnp,
        Grid=Grid,
        Detector=Detector,
        ParallelGeometry=ParallelGeometry,
        LaminographyGeometry=LaminographyGeometry,
        SimConfig=SimConfig,
        simulate=simulate,
        fbp=fbp,
        fista_tv=fista_tv,
        AlignConfig=AlignConfig,
        align=align,
        align_multires=align_multires,
        se3_from_5d=se3_from_5d,
        forward_project_view_T=forward_project_view_T,
        get_detector_grid_device=get_detector_grid_device,
        loss_metrics_abs=_metrics_abs,
        loss_metrics_relative=_metrics_relative,
        loss_metrics_gf=_metrics_gauge_fixed,
        gt_projection_helper=_gt_projection_mse,
    )



def _bundle_geometry(bundle: FixtureBundle, mods: ImportedModules) -> tuple[Any, Any, Any]:
    grid = mods.Grid(**bundle.grid)
    detector = mods.Detector(
        **{k: bundle.detector[k] for k in ("nu", "nv", "du", "dv")},
        det_center=tuple(bundle.detector.get("det_center", (0.0, 0.0))),
    )
    if bundle.geometry_type == "parallel":
        geometry = mods.ParallelGeometry(grid=grid, detector=detector, thetas_deg=bundle.thetas_deg)
    elif bundle.geometry_type == "lamino":
        meta = dict(bundle.geometry_meta or {})
        geometry = mods.LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=bundle.thetas_deg,
            tilt_deg=float(meta.get("tilt_deg", 30.0)),
            tilt_about=str(meta.get("tilt_about", "x")),
        )
    else:
        raise ValueError(f"Unsupported geometry_type in fixture: {bundle.geometry_type}")
    return grid, detector, geometry



def _build_recon_fixture(dataset_cfg: dict[str, Any], mods: ImportedModules, name: str) -> FixtureBundle:
    sim_cfg = mods.SimConfig(
        nx=int(dataset_cfg["nx"]),
        ny=int(dataset_cfg.get("ny", dataset_cfg["nx"])),
        nz=int(dataset_cfg["nz"]),
        nu=int(dataset_cfg["nu"]),
        nv=int(dataset_cfg["nv"]),
        n_views=int(dataset_cfg["n_views"]),
        du=float(dataset_cfg.get("du", 1.0)),
        dv=float(dataset_cfg.get("dv", 1.0)),
        vx=float(dataset_cfg.get("vx", 1.0)),
        vy=float(dataset_cfg.get("vy", 1.0)),
        vz=float(dataset_cfg.get("vz", 1.0)),
        rotation_deg=(None if dataset_cfg.get("rotation_deg") is None else float(dataset_cfg["rotation_deg"])),
        geometry=str(dataset_cfg.get("geometry", "parallel")),
        tilt_deg=float(dataset_cfg.get("tilt_deg", 30.0)),
        tilt_about=str(dataset_cfg.get("tilt_about", "x")),
        phantom=str(dataset_cfg.get("phantom", "shepp")),
        single_size=float(dataset_cfg.get("single_size", 0.5)),
        single_value=float(dataset_cfg.get("single_value", 1.0)),
        single_rotate=bool(dataset_cfg.get("single_rotate", True)),
        n_cubes=int(dataset_cfg.get("n_cubes", 8)),
        n_spheres=int(dataset_cfg.get("n_spheres", 7)),
        min_size=int(dataset_cfg.get("min_size", 4)),
        max_size=int(dataset_cfg.get("max_size", 32)),
        min_value=float(dataset_cfg.get("min_value", 0.1)),
        max_value=float(dataset_cfg.get("max_value", 1.0)),
        max_rot_deg=float(dataset_cfg.get("max_rot_deg", 180.0)),
        noise=str(dataset_cfg.get("noise", "none")),
        noise_level=float(dataset_cfg.get("noise_level", 0.0)),
        seed=int(dataset_cfg.get("seed", 0)),
        lamino_thickness_ratio=float(dataset_cfg.get("lamino_thickness_ratio", 0.2)),
    )
    payload = mods.simulate(sim_cfg)
    return FixtureBundle(
        name=name,
        grid=dict(payload["grid"]),
        detector=dict(payload["detector"]),
        geometry_type=str(payload["geometry_type"]),
        geometry_meta=(dict(payload.get("geometry_meta")) if payload.get("geometry_meta") else None),
        thetas_deg=np.asarray(payload["thetas_deg"], dtype=np.float32),
        volume=np.asarray(payload["volume"], dtype=np.float32),
        projections=np.asarray(payload["projections"], dtype=np.float32),
        align_params=None,
    )



def _build_align_fixture(dataset_cfg: dict[str, Any], mods: ImportedModules, name: str) -> FixtureBundle:
    gt_cfg = dict(dataset_cfg)
    gt_cfg.setdefault("kind", "recon")
    gt_cfg["noise"] = "none"
    gt_cfg["noise_level"] = 0.0
    gt_bundle = _build_recon_fixture(gt_cfg, mods, name)
    grid, detector, geometry = _bundle_geometry(gt_bundle, mods)
    jax = mods.jax
    jnp = mods.jnp

    mis_cfg = dict(dataset_cfg.get("misalignment") or {})
    seed = int(mis_cfg.get("seed", int(dataset_cfg.get("seed", 0)) + 1))
    rot_deg = float(mis_cfg.get("rot_deg", 1.0))
    trans_px = float(mis_cfg.get("trans_px", 5.0))
    include_phi = bool(mis_cfg.get("include_phi", True))

    rng = np.random.default_rng(seed)
    n_views = int(gt_bundle.thetas_deg.shape[0])
    params5 = np.zeros((n_views, 5), dtype=np.float32)
    rot_scale = np.deg2rad(rot_deg)
    params5[:, 0] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    params5[:, 1] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    if include_phi:
        params5[:, 2] = rng.uniform(-rot_scale, rot_scale, n_views).astype(np.float32)
    params5[:, 3] = rng.uniform(-trans_px, trans_px, n_views).astype(np.float32) * float(detector.du)
    params5[:, 4] = rng.uniform(-trans_px, trans_px, n_views).astype(np.float32) * float(detector.dv)

    vol = jnp.asarray(gt_bundle.volume, dtype=jnp.float32)
    T_nom = jnp.stack(
        [jnp.asarray(geometry.pose_for_view(i), dtype=jnp.float32) for i in range(n_views)],
        axis=0,
    )
    T_aug = T_nom @ jax.vmap(mods.se3_from_5d)(jnp.asarray(params5, dtype=jnp.float32))
    det_grid = mods.get_detector_grid_device(detector)
    vm_project = jax.vmap(
        lambda T: mods.forward_project_view_T(
            T,
            grid,
            detector,
            vol,
            use_checkpoint=True,
            det_grid=det_grid,
        ),
        in_axes=0,
    )
    projections = vm_project(T_aug)
    projections = np.asarray(jax.device_get(projections), dtype=np.float32)

    return FixtureBundle(
        name=name,
        grid=gt_bundle.grid,
        detector=gt_bundle.detector,
        geometry_type=gt_bundle.geometry_type,
        geometry_meta=gt_bundle.geometry_meta,
        thetas_deg=gt_bundle.thetas_deg,
        volume=gt_bundle.volume,
        projections=projections,
        align_params=params5,
    )



def _ensure_fixture(profile: dict[str, Any], mods: ImportedModules) -> tuple[FixtureBundle, bool, Path]:
    fixture_name = profile.get("fixture")
    if fixture_name:
        fixture_path = FIXTURES_DIR / str(fixture_name)
    else:
        fixture_path = DATA_DIR / f"{profile['name']}.npz"
    generated = False
    if fixture_path.exists():
        return _load_fixture(fixture_path), generated, fixture_path

    dataset_cfg = dict(profile.get("data") or {})
    if not dataset_cfg:
        raise FileNotFoundError(
            f"Fixture not found and profile has no 'data' section to generate it: {fixture_path}"
        )
    _ensure_dir(fixture_path.parent)
    task = str(profile.get("task", "recon"))
    if task == "align":
        bundle = _build_align_fixture(dataset_cfg, mods, profile["name"])
    else:
        bundle = _build_recon_fixture(dataset_cfg, mods, profile["name"])
    _save_fixture(bundle, fixture_path)
    generated = True
    return bundle, generated, fixture_path



def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


class PeakMemoryMonitor:
    def __init__(
        self,
        *,
        sample_host_rss: bool,
        sample_gpu_memory: bool,
        host_interval: float,
        gpu_interval: float,
    ) -> None:
        self.sample_host_rss = bool(sample_host_rss)
        self.sample_gpu_memory = bool(sample_gpu_memory)
        self.host_interval = max(float(host_interval), 0.01)
        self.gpu_interval = max(float(gpu_interval), 0.01)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid())
        self.peak_host_rss_mb: float | None = None
        self.peak_gpu_memory_mb: float | None = None
        self.gpu_sampler_error: str | None = None

    def _sample_gpu_memory(self) -> None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if not values:
                return
            peak = max(values)
            self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb or 0.0, peak)
        except FileNotFoundError:
            self.gpu_sampler_error = "nvidia-smi not found"
            self.sample_gpu_memory = False
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            self.gpu_sampler_error = str(exc)
            self.sample_gpu_memory = False

    def _sample_host_rss(self) -> None:
        try:
            rss = self._process.memory_info().rss / MB
            self.peak_host_rss_mb = max(self.peak_host_rss_mb or 0.0, rss)
        except Exception:
            pass

    def _run(self) -> None:
        last_host = 0.0
        last_gpu = 0.0
        while not self._stop.is_set():
            now = time.perf_counter()
            if self.sample_host_rss and (now - last_host >= self.host_interval):
                self._sample_host_rss()
                last_host = now
            if self.sample_gpu_memory and (now - last_gpu >= self.gpu_interval):
                self._sample_gpu_memory()
                last_gpu = now
            time.sleep(0.01)
        if self.sample_host_rss:
            self._sample_host_rss()
        if self.sample_gpu_memory:
            self._sample_gpu_memory()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="peak-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)



def _block_tree_ready(jax_module: Any, value: Any) -> Any:
    try:
        return jax_module.block_until_ready(value)
    except Exception:
        pass
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, dict):
        return {k: _block_tree_ready(jax_module, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        t = [_block_tree_ready(jax_module, v) for v in value]
        return type(value)(t) if not isinstance(value, tuple) else tuple(t)
    return value


@dataclass
class RunResult:
    output: Any
    seconds: float
    peak_host_rss_mb: float | None
    peak_gpu_memory_mb: float | None
    gpu_sampler_error: str | None



def _timed_call(fn: Any, mods: ImportedModules, measurement_cfg: dict[str, Any]) -> RunResult:
    monitor = PeakMemoryMonitor(
        sample_host_rss=bool(measurement_cfg.get("host_rss", True)),
        sample_gpu_memory=bool(measurement_cfg.get("gpu_memory", True)),
        host_interval=float(measurement_cfg.get("host_poll_interval_seconds", 0.05)),
        gpu_interval=float(measurement_cfg.get("gpu_poll_interval_seconds", 0.05)),
    )
    monitor.start()
    start = time.perf_counter()
    try:
        output = fn()
        _block_tree_ready(mods.jax, output)
        seconds = time.perf_counter() - start
    finally:
        monitor.stop()
    return RunResult(
        output=output,
        seconds=seconds,
        peak_host_rss_mb=_float_or_none(monitor.peak_host_rss_mb),
        peak_gpu_memory_mb=_float_or_none(monitor.peak_gpu_memory_mb),
        gpu_sampler_error=monitor.gpu_sampler_error,
    )



def _device_info(mods: ImportedModules) -> dict[str, Any]:
    devices = []
    gpu_name = None
    for dev in mods.jax.devices():
        kind = getattr(dev, "device_kind", None) or getattr(dev, "platform", None) or str(dev)
        devices.append(str(kind))
        if gpu_name is None and getattr(dev, "platform", None) == "gpu":
            gpu_name = str(kind)
    return {
        "jax_backend": str(mods.jax.default_backend()),
        "jax_devices": devices,
        "gpu_name": gpu_name,
    }



def _run_recon_profile(bundle: FixtureBundle, profile: dict[str, Any], mods: ImportedModules) -> dict[str, Any]:
    grid, detector, geometry = _bundle_geometry(bundle, mods)
    jnp = mods.jnp
    projections = jnp.asarray(bundle.projections, dtype=jnp.float32)
    volume_gt = jnp.asarray(bundle.volume, dtype=jnp.float32)

    recon_cfg = dict(profile.get("recon") or {})
    algorithm = str(recon_cfg.get("algorithm", "fbp"))

    if algorithm == "fbp":
        def task() -> dict[str, Any]:
            recon = mods.fbp(
                geometry,
                grid,
                detector,
                projections,
                filter_name=str(recon_cfg.get("filter_name", "ramp")),
                scale=(_float_or_none(recon_cfg.get("scale")) if recon_cfg.get("scale") is not None else None),
                views_per_batch=int(recon_cfg.get("views_per_batch", 1)),
                projector_unroll=int(recon_cfg.get("projector_unroll", 1)),
                checkpoint_projector=bool(recon_cfg.get("checkpoint_projector", True)),
                gather_dtype=str(recon_cfg.get("gather_dtype", "fp32")),
            )
            return {"volume": recon}

    elif algorithm == "fista_tv":
        def task() -> dict[str, Any]:
            recon, info = mods.fista_tv(
                geometry,
                grid,
                detector,
                projections,
                iters=int(recon_cfg.get("iters", 6)),
                lambda_tv=float(recon_cfg.get("lambda_tv", 0.003)),
                L=(_float_or_none(recon_cfg.get("L")) if recon_cfg.get("L") is not None else None),
                views_per_batch=int(recon_cfg.get("views_per_batch", 1)),
                projector_unroll=int(recon_cfg.get("projector_unroll", 1)),
                checkpoint_projector=bool(recon_cfg.get("checkpoint_projector", True)),
                gather_dtype=str(recon_cfg.get("gather_dtype", "fp32")),
                grad_mode=str(recon_cfg.get("grad_mode", "auto")),
                tv_prox_iters=int(recon_cfg.get("tv_prox_iters", 10)),
                recon_rel_tol=(_float_or_none(recon_cfg.get("recon_rel_tol"))),
                recon_patience=int(recon_cfg.get("recon_patience", 0)),
            )
            return {"volume": recon, "info": info}

    else:
        raise ValueError(f"Unsupported recon algorithm in profile: {algorithm}")

    measurement_cfg = dict(profile.get("measurement") or {})
    warm_runs = max(1, int(profile.get("warm_runs", 3)))

    first = _timed_call(task, mods, measurement_cfg)
    warms: list[RunResult] = [_timed_call(task, mods, measurement_cfg) for _ in range(warm_runs)]
    warm_seconds = [run.seconds for run in warms]
    warm_volume = warms[-1].output["volume"]
    recon_mse = float(jnp.mean((warm_volume - volume_gt) ** 2).item())

    warm_peak_gpu = max((v for v in [run.peak_gpu_memory_mb for run in warms] if v is not None), default=None)
    warm_peak_host = max((v for v in [run.peak_host_rss_mb for run in warms] if v is not None), default=None)
    first_peak_gpu = first.peak_gpu_memory_mb
    first_peak_host = first.peak_host_rss_mb

    peak_gpu = warm_peak_gpu if warm_peak_gpu is not None else first_peak_gpu
    peak_host = warm_peak_host if warm_peak_host is not None else first_peak_host

    metrics = {
        "profile": profile["name"],
        "task": "recon",
        "algorithm": algorithm,
        "first_run_seconds": first.seconds,
        "warm_run_seconds_mean": float(statistics.mean(warm_seconds)),
        "warm_run_seconds_std": float(statistics.pstdev(warm_seconds) if len(warm_seconds) > 1 else 0.0),
        "first_run_peak_gpu_memory_mb": first_peak_gpu,
        "warm_run_peak_gpu_memory_mb_max": warm_peak_gpu,
        "peak_gpu_memory_mb": peak_gpu,
        "first_run_peak_host_rss_mb": first_peak_host,
        "warm_run_peak_host_rss_mb_max": warm_peak_host,
        "peak_host_rss_mb": peak_host,
        "quality": {
            "recon_mse": recon_mse,
        },
        "gpu_sampler_error": first.gpu_sampler_error or next((w.gpu_sampler_error for w in warms if w.gpu_sampler_error), None),
    }
    return metrics



def _run_align_profile(bundle: FixtureBundle, profile: dict[str, Any], mods: ImportedModules) -> dict[str, Any]:
    if bundle.align_params is None:
        raise ValueError("Alignment profile requires fixture align_params")
    grid, detector, geometry = _bundle_geometry(bundle, mods)
    jnp = mods.jnp
    projections = jnp.asarray(bundle.projections, dtype=jnp.float32)
    align_cfg = dict(profile.get("align") or {})
    levels = align_cfg.get("levels")

    cfg_kwargs = {
        "outer_iters": int(align_cfg.get("outer_iters", 4)),
        "recon_iters": int(align_cfg.get("recon_iters", 10)),
        "lambda_tv": float(align_cfg.get("lambda_tv", 0.005)),
        "tv_prox_iters": int(align_cfg.get("tv_prox_iters", 10)),
        "recon_rel_tol": _float_or_none(align_cfg.get("recon_rel_tol")),
        "recon_patience": int(align_cfg.get("recon_patience", 2)),
        "lr_rot": float(align_cfg.get("lr_rot", 5e-4)),
        "lr_trans": float(align_cfg.get("lr_trans", 5e-2)),
        "views_per_batch": int(align_cfg.get("views_per_batch", 1)),
        "projector_unroll": int(align_cfg.get("projector_unroll", 1)),
        "checkpoint_projector": bool(align_cfg.get("checkpoint_projector", True)),
        "gather_dtype": str(align_cfg.get("gather_dtype", "auto")),
        "opt_method": str(align_cfg.get("opt_method", "gn")),
        "gn_damping": float(align_cfg.get("gn_damping", 1e-3)),
        "w_rot": float(align_cfg.get("w_rot", 1e-3)),
        "w_trans": float(align_cfg.get("w_trans", 1e-3)),
        "seed_translations": bool(align_cfg.get("seed_translations", False)),
        "log_summary": False,
        "log_compact": True,
        "recon_L": (_float_or_none(align_cfg.get("recon_L")) if align_cfg.get("recon_L") is not None else None),
        "early_stop": bool(align_cfg.get("early_stop", True)),
        "early_stop_rel_impr": float(align_cfg.get("early_stop_rel_impr", 1e-3)),
        "early_stop_patience": int(align_cfg.get("early_stop_patience", 2)),
        "loss_kind": str(align_cfg.get("loss_kind", "l2_otsu")),
        "loss_params": align_cfg.get("loss_params"),
    }
    cfg = mods.AlignConfig(**cfg_kwargs)

    if levels:
        level_tuple = tuple(int(v) for v in levels)

        def task() -> dict[str, Any]:
            volume, params, info = mods.align_multires(
                geometry,
                grid,
                detector,
                projections,
                factors=level_tuple,
                cfg=cfg,
            )
            return {"volume": volume, "params": params, "info": info}
    else:
        def task() -> dict[str, Any]:
            volume, params, info = mods.align(
                geometry,
                grid,
                detector,
                projections,
                cfg=cfg,
            )
            return {"volume": volume, "params": params, "info": info}

    measurement_cfg = dict(profile.get("measurement") or {})
    warm_runs = max(1, int(profile.get("warm_runs", 1)))

    first = _timed_call(task, mods, measurement_cfg)
    warms: list[RunResult] = [_timed_call(task, mods, measurement_cfg) for _ in range(warm_runs)]
    warm_seconds = [run.seconds for run in warms]
    warm_params = np.asarray(mods.jax.device_get(warms[-1].output["params"]), dtype=np.float32)

    gt_params = np.asarray(bundle.align_params, dtype=np.float32)
    abs_metrics = mods.loss_metrics_abs(gt_params, warm_params, du=float(detector.du), dv=float(detector.dv))
    rel_metrics = mods.loss_metrics_relative(
        gt_params,
        warm_params,
        du=float(detector.du),
        dv=float(detector.dv),
        k_step=int(align_cfg.get("k_step", 1)),
    )
    gf_metrics = mods.loss_metrics_gf(gt_params, warm_params, du=float(detector.du), dv=float(detector.dv))
    y_hat = mods.gt_projection_helper(
        jnp.asarray(bundle.volume, dtype=jnp.float32),
        grid,
        detector,
        geometry,
        warm_params,
    )
    gt_mse = float(jnp.mean((y_hat - projections) ** 2).item())

    warm_peak_gpu = max((v for v in [run.peak_gpu_memory_mb for run in warms] if v is not None), default=None)
    warm_peak_host = max((v for v in [run.peak_host_rss_mb for run in warms] if v is not None), default=None)
    first_peak_gpu = first.peak_gpu_memory_mb
    first_peak_host = first.peak_host_rss_mb
    peak_gpu = warm_peak_gpu if warm_peak_gpu is not None else first_peak_gpu
    peak_host = warm_peak_host if warm_peak_host is not None else first_peak_host

    metrics = {
        "profile": profile["name"],
        "task": "align",
        "loss_kind": cfg.loss_kind,
        "first_run_seconds": first.seconds,
        "warm_run_seconds_mean": float(statistics.mean(warm_seconds)),
        "warm_run_seconds_std": float(statistics.pstdev(warm_seconds) if len(warm_seconds) > 1 else 0.0),
        "first_run_peak_gpu_memory_mb": first_peak_gpu,
        "warm_run_peak_gpu_memory_mb_max": warm_peak_gpu,
        "peak_gpu_memory_mb": peak_gpu,
        "first_run_peak_host_rss_mb": first_peak_host,
        "warm_run_peak_host_rss_mb_max": warm_peak_host,
        "peak_host_rss_mb": peak_host,
        "quality": {
            **abs_metrics,
            **rel_metrics,
            **gf_metrics,
            "gt_mse": gt_mse,
        },
        "gpu_sampler_error": first.gpu_sampler_error or next((w.gpu_sampler_error for w in warms if w.gpu_sampler_error), None),
    }
    return metrics



def _resolve_objective(metrics: dict[str, Any], profile: dict[str, Any]) -> tuple[str, str, Any]:
    objective_name = str(profile.get("objective_name", "warm_run_seconds_mean"))
    objective_direction = str(profile.get("objective_direction", "minimise"))

    if objective_name in metrics:
        objective_value = metrics.get(objective_name)
    else:
        quality = metrics.get("quality") or {}
        objective_value = quality.get(objective_name)
    return objective_name, objective_direction, objective_value



def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value



def main() -> int:
    args = _parse_args()
    out_path = Path(args.out).resolve()
    _ensure_dir(out_path.parent)
    _ensure_dir(DATA_DIR)
    _ensure_dir(OUT_DIR)

    profile_path = _resolve_profile_path(args.profile, args.profile_root)
    profile = _load_profile(profile_path)
    env_updates = _configure_environment(profile)

    metrics: dict[str, Any] = {
        "profile": profile.get("name", profile_path.stem),
        "success": False,
        "objective_name": str(profile.get("objective_name", "warm_run_seconds_mean")),
        "objective_direction": str(profile.get("objective_direction", "minimise")),
        "objective_value": None,
        "first_run_seconds": None,
        "warm_run_seconds_mean": None,
        "warm_run_seconds_std": None,
        "peak_gpu_memory_mb": None,
        "peak_host_rss_mb": None,
        "quality": {},
        "device": {},
        "oom": False,
        "error": None,
        "profile_path": str(profile_path),
        "env": env_updates,
    }

    try:
        mods = _import_modules(profile)
        fixture, fixture_generated, fixture_path = _ensure_fixture(profile, mods)
        if str(profile.get("task", "recon")) == "align":
            run_metrics = _run_align_profile(fixture, profile, mods)
        else:
            run_metrics = _run_recon_profile(fixture, profile, mods)

        metrics.update(run_metrics)
        objective_name, objective_direction, objective_value = _resolve_objective(metrics, profile)
        metrics["objective_name"] = objective_name
        metrics["objective_direction"] = objective_direction
        metrics["objective_value"] = _float_or_none(objective_value) if isinstance(objective_value, (float, int, np.floating, np.integer)) else objective_value
        metrics["device"] = _device_info(mods)
        metrics["fixture"] = {
            "path": str(fixture_path),
            "generated_in_process": fixture_generated,
            **fixture.shape_summary,
        }
        metrics["success"] = metrics.get("objective_value") is not None or objective_name == "warm_run_seconds_mean"
    except Exception as exc:  # pragma: no cover - exercised in error conditions
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        metrics["error"] = message
        msg_lower = message.lower()
        metrics["oom"] = ("resource exhausted" in msg_lower) or ("out of memory" in msg_lower)
        metrics["success"] = False
        if not metrics.get("device"):
            try:
                import jax
                metrics["device"] = {
                    "jax_backend": str(jax.default_backend()),
                    "jax_devices": [str(getattr(d, "device_kind", d)) for d in jax.devices()],
                    "gpu_name": None,
                }
            except Exception:
                pass

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0 if metrics.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
