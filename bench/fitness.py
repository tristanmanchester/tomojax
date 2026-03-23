#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil
import yaml
from memory import GpuMemoryMonitor, GpuMemorySnapshot
from visualize import save_alignment_summary

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = BENCH_ROOT / "profiles"
FIXTURES_DIR = BENCH_ROOT / "fixtures"
DATA_DIR = BENCH_ROOT / "data"
OUT_DIR = BENCH_ROOT / "out"
MB = 1024.0 * 1024.0
ProgressCallback = Callable[[dict[str, Any]], None]


def _repo_pythonpath() -> None:
    src = REPO_ROOT / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed TomoJAX benchmark profiles.")
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile name (e.g. screen_speed_parallel_fbp_128) or path to a YAML profile.",
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


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)


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

@dataclass(frozen=True)
class ConvergenceConfig:
    enabled: bool = False
    metric: str = "gt_mse"
    threshold: float | None = None
    stop_on_threshold: bool = True
    stop_on_plateau: bool = True
    min_finest_level_checks: int = 2
    plateau_patience: int = 2
    rel_improvement_tol: float = 0.02
    required_warm_successes: int = 1

@dataclass
class ConvergenceRunSummary:
    metric: str
    threshold: float | None
    threshold_met: bool
    stopped_on_threshold: bool
    stopped_on_plateau: bool
    stopped_on_budget: bool
    seconds_to_threshold: float | None
    outer_iters_to_threshold: int | None
    best_quality_value: float | None
    best_quality_elapsed_seconds: float | None
    total_outer_iters_executed: int
    final_stop_reason: str | None
    final_stop_level_factor: int | None
    first_threshold_crossing_level_factor: int | None
    trace: list[dict[str, Any]]


def _median_or_none(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(statistics.median(finite))


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(statistics.mean(finite))


def _int_median_or_none(values: list[int | None]) -> int | None:
    nums = [int(v) for v in values if v is not None]
    if not nums:
        return None
    return int(round(statistics.median(nums)))


def _is_meaningful_relative_improvement(
    previous_best: float | None,
    current_value: float | None,
    rel_tol: float,
) -> bool:
    if previous_best is None or current_value is None:
        return True
    if not math.isfinite(previous_best) or not math.isfinite(current_value):
        return False
    if current_value < previous_best:
        delta = previous_best - current_value
        return delta >= (max(abs(previous_best), 1e-12) * float(rel_tol))
    return False


def _aggregate_warm_convergence_runs(
    runs: list[ConvergenceRunSummary],
    *,
    required_successes: int,
) -> dict[str, Any]:
    if not runs:
        return {
            "quality_threshold_met": False,
            "warm_threshold_hit_count": 0,
            "warm_threshold_total_runs": 0,
            "warm_threshold_success_rate": None,
            "warm_seconds_to_quality_threshold": None,
            "warm_outer_iters_to_quality_threshold": None,
            "warm_best_quality_value": None,
            "best_quality_value": None,
            "best_quality_elapsed_seconds": None,
            "warm_total_outer_iters_executed": None,
            "total_outer_iters_executed": None,
            "warm_stopped_on_threshold_count": 0,
            "warm_stopped_on_plateau_count": 0,
            "warm_stopped_on_budget_count": 0,
            "stopped_on_threshold": False,
            "stopped_on_plateau": False,
            "stopped_on_budget": False,
            "warm_convergence_trace": [],
            "warm_convergence_traces": [],
            "final_stop_reason": None,
            "final_stop_level_factor": None,
            "first_threshold_crossing_level_factor": None,
        }

    hit_count = sum(1 for run in runs if run.threshold_met)
    total_runs = len(runs)
    required = max(1, min(int(required_successes), total_runs))
    representative = runs[-1]
    successful_runs = [run for run in runs if run.threshold_met]
    best_quality_values = [run.best_quality_value for run in runs]
    best_quality_times = [run.best_quality_elapsed_seconds for run in runs]
    total_outers = [run.total_outer_iters_executed for run in runs]

    return {
        "quality_threshold_met": hit_count >= required,
        "warm_threshold_hit_count": hit_count,
        "warm_threshold_total_runs": total_runs,
        "warm_threshold_success_rate": float(hit_count / total_runs),
        "warm_seconds_to_quality_threshold": _median_or_none(
            [run.seconds_to_threshold for run in successful_runs]
        ),
        "warm_outer_iters_to_quality_threshold": _int_median_or_none(
            [run.outer_iters_to_threshold for run in successful_runs]
        ),
        "warm_best_quality_value": _median_or_none(best_quality_values),
        "best_quality_value": _median_or_none(best_quality_values),
        "best_quality_elapsed_seconds": _median_or_none(best_quality_times),
        "warm_total_outer_iters_executed": _int_median_or_none(total_outers),
        "total_outer_iters_executed": _int_median_or_none(total_outers),
        "warm_stopped_on_threshold_count": sum(1 for run in runs if run.stopped_on_threshold),
        "warm_stopped_on_plateau_count": sum(1 for run in runs if run.stopped_on_plateau),
        "warm_stopped_on_budget_count": sum(1 for run in runs if run.stopped_on_budget),
        "stopped_on_threshold": representative.stopped_on_threshold,
        "stopped_on_plateau": representative.stopped_on_plateau,
        "stopped_on_budget": representative.stopped_on_budget,
        "final_stop_reason": representative.final_stop_reason,
        "final_stop_level_factor": representative.final_stop_level_factor,
        "first_threshold_crossing_level_factor": representative.first_threshold_crossing_level_factor,
        "warm_convergence_trace": representative.trace,
        "warm_convergence_traces": [run.trace for run in runs],
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
    projections = _apply_projection_noise(
        projections,
        noise=str(dataset_cfg.get("noise", "none")),
        noise_level=float(dataset_cfg.get("noise_level", 0.0)),
        seed=int(dataset_cfg.get("noise_seed", seed + 17)),
    )

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



def _ensure_fixture(
    profile: dict[str, Any],
    mods: ImportedModules,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[FixtureBundle, bool, Path]:
    fixture_name = profile.get("fixture")
    if fixture_name:
        fixture_path = FIXTURES_DIR / str(fixture_name)
    else:
        fixture_path = DATA_DIR / f"{profile['name']}.npz"
    generated = False
    if fixture_path.exists():
        _emit_progress(
            progress_callback,
            stage_kind="fixture_prepare",
            message="Loading cached benchmark fixture.",
            detail=f"fixture {fixture_path.name}",
            fixture_path=str(fixture_path),
            fixture_generated=False,
        )
        return _load_fixture(fixture_path), generated, fixture_path

    dataset_cfg = dict(profile.get("data") or {})
    if not dataset_cfg:
        raise FileNotFoundError(
            f"Fixture not found and profile has no 'data' section to generate it: {fixture_path}"
        )
    _ensure_dir(fixture_path.parent)
    _emit_progress(
        progress_callback,
        stage_kind="fixture_prepare",
        message="Generating benchmark fixture.",
        detail=f"fixture {fixture_path.name}",
        fixture_path=str(fixture_path),
        fixture_generated=True,
    )
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

def _convergence_config(profile: dict[str, Any]) -> ConvergenceConfig:
    raw = dict(profile.get("convergence") or {})
    enabled = bool(raw.get("enabled", False))
    metric = str(raw.get("metric", "gt_mse"))
    threshold = _float_or_none(raw.get("threshold"))
    stop_on_threshold = bool(raw.get("stop_on_threshold", True))
    stop_on_plateau = bool(raw.get("stop_on_plateau", True))
    return ConvergenceConfig(
        enabled=enabled,
        metric=metric,
        threshold=threshold,
        stop_on_threshold=stop_on_threshold,
        stop_on_plateau=stop_on_plateau,
        min_finest_level_checks=max(1, int(raw.get("min_finest_level_checks", 2))),
        plateau_patience=max(1, int(raw.get("plateau_patience", 2))),
        rel_improvement_tol=max(0.0, float(raw.get("rel_improvement_tol", 0.02))),
        required_warm_successes=max(1, int(raw.get("required_warm_successes", 1))),
    )

def _quality_threshold_met(metric: str, threshold: float | None, value: float | None) -> bool:
    if threshold is None or value is None:
        return False
    if metric == "gt_mse":
        return value <= threshold
    raise ValueError(f"Unsupported convergence metric: {metric}")


def _convergence_action_for_level(
    *,
    level_factor: int,
    finest_factor: int,
    budget_hit: bool,
    threshold_hit: bool,
    plateau_hit: bool,
) -> str:
    if budget_hit:
        return "stop_run"
    if threshold_hit:
        return "stop_run" if level_factor == finest_factor else "advance_level"
    if plateau_hit:
        return "stop_run" if level_factor == finest_factor else "advance_level"
    return "continue"

def _convergence_summary_from_trace(
    *,
    metric: str,
    threshold: float | None,
    trace: list[dict[str, Any]],
    stopped_on_threshold: bool = False,
    stopped_on_plateau: bool = False,
    stopped_on_budget: bool = False,
) -> ConvergenceRunSummary:
    best_quality_value: float | None = None
    best_quality_elapsed_seconds: float | None = None
    seconds_to_threshold: float | None = None
    outer_iters_to_threshold: int | None = None
    first_threshold_crossing_level_factor: int | None = None

    for point in trace:
        quality_value = _float_or_none(point.get("quality_value"))
        elapsed_seconds = _float_or_none(point.get("elapsed_seconds"))
        outer_idx = point.get("outer_idx")

        if quality_value is not None and (
            best_quality_value is None or quality_value < best_quality_value
        ):
            best_quality_value = quality_value
            best_quality_elapsed_seconds = elapsed_seconds

        if (
            seconds_to_threshold is None
            and _quality_threshold_met(metric, threshold, quality_value)
        ):
            seconds_to_threshold = elapsed_seconds
            try:
                level_factor = point.get("level_factor")
                first_threshold_crossing_level_factor = (
                    int(level_factor) if level_factor is not None else None
                )
            except Exception:
                first_threshold_crossing_level_factor = None
            try:
                outer_iters_to_threshold = int(outer_idx) if outer_idx is not None else None
            except Exception:
                outer_iters_to_threshold = None

    final_stop_level_factor: int | None = None
    if trace:
        try:
            last_level_factor = trace[-1].get("level_factor")
            final_stop_level_factor = int(last_level_factor) if last_level_factor is not None else None
        except Exception:
            final_stop_level_factor = None
    final_stop_reason = (
        "threshold"
        if stopped_on_threshold
        else ("plateau" if stopped_on_plateau else ("budget" if stopped_on_budget else None))
    )

    return ConvergenceRunSummary(
        metric=metric,
        threshold=threshold,
        threshold_met=seconds_to_threshold is not None,
        stopped_on_threshold=bool(stopped_on_threshold),
        stopped_on_plateau=bool(stopped_on_plateau),
        stopped_on_budget=bool(stopped_on_budget),
        seconds_to_threshold=seconds_to_threshold,
        outer_iters_to_threshold=outer_iters_to_threshold,
        best_quality_value=best_quality_value,
        best_quality_elapsed_seconds=best_quality_elapsed_seconds,
        total_outer_iters_executed=len(trace),
        final_stop_reason=final_stop_reason,
        final_stop_level_factor=final_stop_level_factor,
        first_threshold_crossing_level_factor=first_threshold_crossing_level_factor,
        trace=trace,
    )


def _apply_projection_noise(
    projections: np.ndarray,
    *,
    noise: str,
    noise_level: float,
    seed: int,
) -> np.ndarray:
    noise_kind = str(noise or "none").strip().lower()
    sigma_or_scale = float(noise_level)
    if noise_kind == "none" or sigma_or_scale <= 0:
        return np.asarray(projections, dtype=np.float32)

    rng = np.random.default_rng(seed)
    proj = np.asarray(projections, dtype=np.float32)
    if noise_kind == "gaussian":
        noisy = proj + rng.normal(scale=sigma_or_scale, size=proj.shape).astype(np.float32)
        return np.asarray(noisy, dtype=np.float32)
    if noise_kind == "poisson":
        lam = np.clip(proj, 0.0, None) * sigma_or_scale
        noisy = rng.poisson(lam=lam).astype(np.float32) / max(sigma_or_scale, 1e-6)
        return np.asarray(noisy, dtype=np.float32)
    raise ValueError(f"Unsupported noise kind in benchmark profile: {noise}")


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
        self.gpu_memory = GpuMemoryMonitor(
            enabled=self.sample_gpu_memory,
            interval_seconds=self.gpu_interval,
            root_pid=os.getpid(),
        )

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
                self.gpu_memory.sample_once()
                last_gpu = now
            time.sleep(0.01)
        if self.sample_host_rss:
            self._sample_host_rss()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="peak-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> GpuMemorySnapshot:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.gpu_memory.stop()



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
    peak_gpu_memory_process_mb: float | None
    peak_gpu_memory_device_mb: float | None
    gpu_memory_backend: str
    gpu_memory_scope: str
    gpu_memory_process_source: str | None
    gpu_memory_process_supported: bool
    gpu_memory_sample_interval_seconds: float
    gpu_memory_sample_count: int
    gpu_memory_observed_gpu_count: int
    gpu_sampler_error: str | None



def _timed_call(
    fn: Any,
    mods: ImportedModules,
    measurement_cfg: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    progress_payload: dict[str, Any] | None = None,
) -> RunResult:
    monitor = PeakMemoryMonitor(
        sample_host_rss=bool(measurement_cfg.get("host_rss", True)),
        sample_gpu_memory=bool(measurement_cfg.get("gpu_memory", True)),
        host_interval=float(measurement_cfg.get("host_poll_interval_seconds", 0.05)),
        gpu_interval=float(measurement_cfg.get("gpu_poll_interval_seconds", 0.05)),
    )
    monitor.start()
    start = time.perf_counter()
    _emit_progress(progress_callback, **(progress_payload or {}))
    try:
        output = fn()
        _block_tree_ready(mods.jax, output)
        seconds = time.perf_counter() - start
    finally:
        gpu_snapshot = monitor.stop()
    return RunResult(
        output=output,
        seconds=seconds,
        peak_host_rss_mb=_float_or_none(monitor.peak_host_rss_mb),
        peak_gpu_memory_mb=_float_or_none(
            gpu_snapshot.process_peak_mb
            if gpu_snapshot.process_peak_mb is not None
            else gpu_snapshot.device_peak_mb
        ),
        peak_gpu_memory_process_mb=_float_or_none(gpu_snapshot.process_peak_mb),
        peak_gpu_memory_device_mb=_float_or_none(gpu_snapshot.device_peak_mb),
        gpu_memory_backend=gpu_snapshot.backend,
        gpu_memory_scope=gpu_snapshot.scope,
        gpu_memory_process_source=gpu_snapshot.process_source,
        gpu_memory_process_supported=bool(gpu_snapshot.process_supported),
        gpu_memory_sample_interval_seconds=gpu_snapshot.sample_interval_seconds,
        gpu_memory_sample_count=gpu_snapshot.sample_count,
        gpu_memory_observed_gpu_count=gpu_snapshot.observed_gpu_count,
        gpu_sampler_error=gpu_snapshot.sampler_error,
    )


def _maybe_save_jax_device_memory_profile(
    mods: ImportedModules, measurement_cfg: dict[str, Any], out_path: Path
) -> tuple[str | None, str | None]:
    if not bool(measurement_cfg.get("save_jax_device_memory_profile", False)):
        return None, None
    try:
        artifact_path = out_path.with_suffix(out_path.suffix + ".device-memory.prof")
        mods.jax.profiler.save_device_memory_profile(str(artifact_path))
        return str(artifact_path), None
    except Exception as exc:
        return None, str(exc)


def _should_render_alignment_summary(profile: dict[str, Any]) -> bool:
    if str(profile.get("task", "recon")) != "align":
        return False
    visualization_cfg = dict(profile.get("visualization") or {})
    return bool(visualization_cfg.get("enabled", True))


def _alignment_summary_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".summary.png")


def _alignment_baseline_volume(
    bundle: FixtureBundle,
    align_cfg: dict[str, Any],
    mods: ImportedModules,
) -> np.ndarray:
    grid, detector, geometry = _bundle_geometry(bundle, mods)
    projections = mods.jnp.asarray(bundle.projections, dtype=mods.jnp.float32)
    baseline = mods.fbp(
        geometry,
        grid,
        detector,
        projections,
        filter_name="ramp",
        scale=None,
        views_per_batch=int(align_cfg.get("views_per_batch", 1)),
        projector_unroll=int(align_cfg.get("projector_unroll", 1)),
        checkpoint_projector=bool(align_cfg.get("checkpoint_projector", True)),
        gather_dtype=str(align_cfg.get("gather_dtype", "fp32")),
    )
    return np.asarray(mods.jax.device_get(baseline), dtype=np.float32)

def _make_align_task(
    *,
    bundle: FixtureBundle,
    grid: Any,
    detector: Any,
    geometry: Any,
    projections: Any,
    cfg: Any,
    levels: tuple[int, ...] | None,
    convergence: ConvergenceConfig,
    mods: ImportedModules,
    progress_callback: ProgressCallback | None = None,
    run_kind: str = "warm",
    run_index: int = 1,
    total_runs: int = 1,
    total_outer_iters: int | None = None,
    time_budget_seconds: float | None = None,
    observe_trace: bool = False,
) -> Callable[[], dict[str, Any]]:
    gt_volume = mods.jnp.asarray(bundle.volume, dtype=mods.jnp.float32)
    gt_params = np.asarray(bundle.align_params, dtype=np.float32) if bundle.align_params is not None else None
    trace: list[dict[str, Any]] = []
    finest_factor = min(levels) if levels else 1
    level_checks = 0
    plateau_streak = 0
    best_level_quality: float | None = None
    stop_reason: str | None = None
    current_level_factor: int | None = None

    def _quality_value_for_params(params: Any) -> float:
        if convergence.metric != "gt_mse":
            raise ValueError(f"Unsupported convergence metric: {convergence.metric}")
        y_hat = mods.gt_projection_helper(gt_volume, grid, detector, geometry, params)
        return float(mods.jnp.mean((y_hat - projections) ** 2).item())

    def _trans_rmse_for_params(params: Any) -> float | None:
        if gt_params is None:
            return None
        params_np = np.asarray(mods.jax.device_get(params), dtype=np.float32)
        abs_metrics = mods.loss_metrics_abs(
            gt_params,
            params_np,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        return _float_or_none(abs_metrics.get("trans_rmse_px"))

    def _observer(_: Any, params: Any, stat: dict[str, Any]) -> str:
        nonlocal level_checks, plateau_streak, best_level_quality, stop_reason, current_level_factor
        quality_value = _quality_value_for_params(params)
        trans_rmse_px = _trans_rmse_for_params(params)
        level_factor = (
            int(stat["level_factor"]) if stat.get("level_factor") is not None else finest_factor
        )
        if current_level_factor != level_factor:
            current_level_factor = level_factor
            level_checks = 0
            plateau_streak = 0
            best_level_quality = None
        global_elapsed = _float_or_none(
            stat.get("global_elapsed_seconds", stat.get("cumulative_time"))
        )
        level_elapsed = _float_or_none(stat.get("level_elapsed_seconds", stat.get("cumulative_time")))
        budget_hit = (
            time_budget_seconds is not None
            and global_elapsed is not None
            and global_elapsed >= float(time_budget_seconds)
        )
        threshold_hit = _quality_threshold_met(
            convergence.metric, convergence.threshold, quality_value
        )
        is_finest_level = level_factor == finest_factor
        level_checks += 1
        if _is_meaningful_relative_improvement(
            best_level_quality, quality_value, convergence.rel_improvement_tol
        ):
            best_level_quality = quality_value
            plateau_streak = 0
        elif level_checks >= convergence.min_finest_level_checks:
            plateau_streak += 1
        plateau_hit = (
            level_checks >= convergence.min_finest_level_checks
            and plateau_streak >= convergence.plateau_patience
        )
        action = _convergence_action_for_level(
            level_factor=level_factor,
            finest_factor=finest_factor,
            budget_hit=budget_hit,
            threshold_hit=(convergence.stop_on_threshold and threshold_hit),
            plateau_hit=(convergence.stop_on_plateau and plateau_hit),
        )
        stop_reason = None
        if action == "stop_run":
            if budget_hit:
                stop_reason = "budget"
            else:
                stop_reason = "threshold" if threshold_hit else "plateau"
        trace.append(
            {
                "outer_idx": int(
                    stat.get("global_outer_idx", stat.get("outer_idx", len(trace) + 1))
                ),
                "level_index": (
                    int(stat["level_index"]) if stat.get("level_index") is not None else None
                ),
                "level_factor": (
                    int(stat["level_factor"]) if stat.get("level_factor") is not None else None
                ),
                "elapsed_seconds": global_elapsed,
                "level_elapsed_seconds": level_elapsed,
                "quality_value": quality_value,
                "trans_rmse_px": trans_rmse_px,
                "loss_after": _float_or_none(stat.get("loss_after")),
                "threshold_met": threshold_hit,
                "is_finest_level": is_finest_level,
                "plateau_streak": plateau_streak,
                "action": action,
                "stop_reason": stop_reason,
            }
        )
        _emit_progress(
            progress_callback,
            stage_kind="profile_running",
            task="align",
            run_kind=run_kind,
            run_index=run_index,
            total_runs=total_runs,
            level_factor=level_factor,
            level_index=(int(stat["level_index"]) if stat.get("level_index") is not None else None),
            outer_idx=int(stat.get("global_outer_idx", stat.get("outer_idx", len(trace)))),
            total_outer_iters=total_outer_iters,
            quality_metric=convergence.metric,
            quality_value=quality_value,
            trans_rmse_px=trans_rmse_px,
            quality_threshold=convergence.threshold,
            threshold_met=threshold_hit,
            stop_reason=stop_reason,
            is_finest_level=is_finest_level,
            run_seconds=global_elapsed,
            message=(
                f"{run_kind} run {run_index}/{total_runs}: level {level_factor}, "
                f"outer {int(stat.get('global_outer_idx', stat.get('outer_idx', len(trace))))}/{total_outer_iters or '?'}"
            ),
            detail=(
                f"{convergence.metric}={quality_value:.4f}"
                + (f" trans_rmse_px={trans_rmse_px:.4f}" if trans_rmse_px is not None else "")
                + (
                    f" threshold={convergence.threshold:.4f}"
                    if convergence.threshold is not None
                    else ""
                )
                + (
                    f" budget={time_budget_seconds:.0f}s"
                    if time_budget_seconds is not None
                    else ""
                )
            ),
        )
        return action

    def task() -> dict[str, Any]:
        if levels:
            volume, params, info = mods.align_multires(
                geometry,
                grid,
                detector,
                projections,
                factors=levels,
                cfg=cfg,
                observer=_observer if observe_trace else None,
            )
        else:
            volume, params, info = mods.align(
                geometry,
                grid,
                detector,
                projections,
                cfg=cfg,
                observer=_observer if observe_trace else None,
            )
        return {
            "volume": volume,
            "params": params,
            "info": info,
            "convergence": (
                _convergence_summary_from_trace(
                    metric=convergence.metric,
                    threshold=convergence.threshold,
                    trace=list(trace),
                    stopped_on_threshold=(stop_reason == "threshold"),
                    stopped_on_plateau=(stop_reason == "plateau"),
                    stopped_on_budget=(stop_reason == "budget"),
                )
                if trace
                else None
            ),
        }

    return task



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



def _run_recon_profile(
    bundle: FixtureBundle,
    profile: dict[str, Any],
    mods: ImportedModules,
    out_path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
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

    first = _timed_call(
        task,
        mods,
        measurement_cfg,
        progress_callback=progress_callback,
        progress_payload={
            "stage_kind": "profile_running",
            "task": "recon",
            "run_kind": "cold",
            "run_index": 1,
            "total_runs": 1,
            "message": f"Cold run for {algorithm}.",
            "detail": f"algorithm={algorithm}",
        },
    )
    warms: list[RunResult] = [
        _timed_call(
            task,
            mods,
            measurement_cfg,
            progress_callback=progress_callback,
            progress_payload={
                "stage_kind": "profile_running",
                "task": "recon",
                "run_kind": "warm",
                "run_index": index + 1,
                "total_runs": warm_runs,
                "message": f"Warm run {index + 1}/{warm_runs} for {algorithm}.",
                "detail": f"algorithm={algorithm}",
            },
        )
        for index in range(warm_runs)
    ]
    warm_seconds = [run.seconds for run in warms]
    warm_volume = warms[-1].output["volume"]
    recon_mse = float(jnp.mean((warm_volume - volume_gt) ** 2).item())

    warm_peak_gpu = max((v for v in [run.peak_gpu_memory_mb for run in warms] if v is not None), default=None)
    warm_peak_gpu_process = max(
        (v for v in [run.peak_gpu_memory_process_mb for run in warms] if v is not None),
        default=None,
    )
    warm_peak_gpu_device = max(
        (v for v in [run.peak_gpu_memory_device_mb for run in warms] if v is not None),
        default=None,
    )
    warm_peak_host = max((v for v in [run.peak_host_rss_mb for run in warms] if v is not None), default=None)
    first_peak_gpu = first.peak_gpu_memory_mb
    first_peak_gpu_process = first.peak_gpu_memory_process_mb
    first_peak_gpu_device = first.peak_gpu_memory_device_mb
    first_peak_host = first.peak_host_rss_mb
    jax_profile_path, jax_profile_error = _maybe_save_jax_device_memory_profile(
        mods, measurement_cfg, out_path
    )

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
        "first_run_peak_gpu_memory_process_mb": first_peak_gpu_process,
        "warm_run_peak_gpu_memory_process_mb_max": warm_peak_gpu_process,
        "peak_gpu_memory_process_mb": (
            warm_peak_gpu_process if warm_peak_gpu_process is not None else first_peak_gpu_process
        ),
        "first_run_peak_gpu_memory_device_mb": first_peak_gpu_device,
        "warm_run_peak_gpu_memory_device_mb_max": warm_peak_gpu_device,
        "peak_gpu_memory_device_mb": (
            warm_peak_gpu_device if warm_peak_gpu_device is not None else first_peak_gpu_device
        ),
        "first_run_peak_host_rss_mb": first_peak_host,
        "warm_run_peak_host_rss_mb_max": warm_peak_host,
        "peak_host_rss_mb": peak_host,
        "gpu_memory_backend": first.gpu_memory_backend,
        "gpu_memory_scope": (
            "process"
            if (warm_peak_gpu_process is not None or first_peak_gpu_process is not None)
            else first.gpu_memory_scope
        ),
        "gpu_memory_sample_interval_seconds": first.gpu_memory_sample_interval_seconds,
        "gpu_memory_sample_count": int(
            first.gpu_memory_sample_count + sum(w.gpu_memory_sample_count for w in warms)
        ),
        "gpu_memory_observed_gpu_count": max(
            [first.gpu_memory_observed_gpu_count, *[w.gpu_memory_observed_gpu_count for w in warms]]
        ),
        "gpu_memory_process_source": first.gpu_memory_process_source,
        "gpu_memory_process_supported": bool(
            first.gpu_memory_process_supported or any(w.gpu_memory_process_supported for w in warms)
        ),
        "jax_device_memory_profile_path": jax_profile_path,
        "jax_device_memory_profile_error": jax_profile_error,
        "quality": {
            "recon_mse": recon_mse,
        },
        "gpu_sampler_error": first.gpu_sampler_error or next((w.gpu_sampler_error for w in warms if w.gpu_sampler_error), None),
    }
    return metrics



def _run_align_profile(
    bundle: FixtureBundle,
    profile: dict[str, Any],
    mods: ImportedModules,
    out_path: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if bundle.align_params is None:
        raise ValueError("Alignment profile requires fixture align_params")
    grid, detector, geometry = _bundle_geometry(bundle, mods)
    jnp = mods.jnp
    projections = jnp.asarray(bundle.projections, dtype=jnp.float32)
    align_cfg = dict(profile.get("align") or {})
    levels = align_cfg.get("levels")
    level_tuple = tuple(int(v) for v in levels) if levels else None
    convergence = _convergence_config(profile)
    time_budget_seconds = _float_or_none(align_cfg.get("time_budget_seconds"))

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

    measurement_cfg = dict(profile.get("measurement") or {})
    warm_runs = max(1, int(profile.get("warm_runs", 1)))
    observe_trace = bool(
        convergence.enabled
        or time_budget_seconds is not None
        or progress_callback is not None
        or _should_render_alignment_summary(profile)
    )

    total_outer_iters = int(align_cfg.get("outer_iters", 4)) * (len(level_tuple) if level_tuple else 1)

    first = _timed_call(
        _make_align_task(
            bundle=bundle,
            grid=grid,
            detector=detector,
            geometry=geometry,
            projections=projections,
            cfg=cfg,
            levels=level_tuple,
            convergence=convergence,
            mods=mods,
            progress_callback=progress_callback,
            run_kind="cold",
            run_index=1,
            total_runs=1,
            total_outer_iters=total_outer_iters,
            time_budget_seconds=time_budget_seconds,
            observe_trace=observe_trace,
        ),
        mods,
        measurement_cfg,
        progress_callback=progress_callback,
        progress_payload={
            "stage_kind": "profile_running",
            "task": "align",
            "run_kind": "cold",
            "run_index": 1,
            "total_runs": 1,
            "total_outer_iters": total_outer_iters,
            "message": "Cold alignment run starting.",
            "detail": (
                f"levels={list(level_tuple) if level_tuple else [1]}"
                + (f" budget={time_budget_seconds:.0f}s" if time_budget_seconds is not None else "")
            ),
        },
    )
    warms: list[RunResult] = [
        _timed_call(
            _make_align_task(
                bundle=bundle,
                grid=grid,
                detector=detector,
                geometry=geometry,
                projections=projections,
                cfg=cfg,
                levels=level_tuple,
                convergence=convergence,
                mods=mods,
                progress_callback=progress_callback,
                run_kind="warm",
                run_index=index + 1,
                total_runs=warm_runs,
                total_outer_iters=total_outer_iters,
                time_budget_seconds=time_budget_seconds,
                observe_trace=observe_trace,
            ),
            mods,
            measurement_cfg,
            progress_callback=progress_callback,
            progress_payload={
                "stage_kind": "profile_running",
                "task": "align",
                "run_kind": "warm",
                "run_index": index + 1,
                "total_runs": warm_runs,
                "total_outer_iters": total_outer_iters,
                "message": f"Warm alignment run {index + 1}/{warm_runs} starting.",
                "detail": (
                    f"levels={list(level_tuple) if level_tuple else [1]}"
                    + (f" budget={time_budget_seconds:.0f}s" if time_budget_seconds is not None else "")
                ),
            },
        )
        for index in range(warm_runs)
    ]
    warm_seconds = [run.seconds for run in warms]
    warm_params = np.asarray(mods.jax.device_get(warms[-1].output["params"]), dtype=np.float32)
    final_volume = np.asarray(mods.jax.device_get(warms[-1].output["volume"]), dtype=np.float32)
    final_info = warms[-1].output.get("info") or {}
    first_convergence = first.output.get("convergence")
    warm_convergences = [
        run.output.get("convergence") for run in warms if run.output.get("convergence") is not None
    ]
    warm_convergence = warms[-1].output.get("convergence")

    gt_params = np.asarray(bundle.align_params, dtype=np.float32)
    warm_gt_mse_values: list[float] = []
    warm_trans_rmse_values: list[float | None] = []
    gt_volume = jnp.asarray(bundle.volume, dtype=jnp.float32)
    for run in warms:
        run_params = np.asarray(mods.jax.device_get(run.output["params"]), dtype=np.float32)
        run_abs_metrics = mods.loss_metrics_abs(
            gt_params,
            run_params,
            du=float(detector.du),
            dv=float(detector.dv),
        )
        run_y_hat = mods.gt_projection_helper(
            gt_volume,
            grid,
            detector,
            geometry,
            run_params,
        )
        warm_gt_mse_values.append(float(jnp.mean((run_y_hat - projections) ** 2).item()))
        warm_trans_rmse_values.append(_float_or_none(run_abs_metrics.get("trans_rmse_px")))
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
        gt_volume,
        grid,
        detector,
        geometry,
        warm_params,
    )
    gt_mse = float(jnp.mean((y_hat - projections) ** 2).item())

    warm_peak_gpu = max((v for v in [run.peak_gpu_memory_mb for run in warms] if v is not None), default=None)
    warm_peak_gpu_process = max(
        (v for v in [run.peak_gpu_memory_process_mb for run in warms] if v is not None),
        default=None,
    )
    warm_peak_gpu_device = max(
        (v for v in [run.peak_gpu_memory_device_mb for run in warms] if v is not None),
        default=None,
    )
    warm_peak_host = max((v for v in [run.peak_host_rss_mb for run in warms] if v is not None), default=None)
    first_peak_gpu = first.peak_gpu_memory_mb
    first_peak_gpu_process = first.peak_gpu_memory_process_mb
    first_peak_gpu_device = first.peak_gpu_memory_device_mb
    first_peak_host = first.peak_host_rss_mb
    jax_profile_path, jax_profile_error = _maybe_save_jax_device_memory_profile(
        mods, measurement_cfg, out_path
    )
    peak_gpu = warm_peak_gpu if warm_peak_gpu is not None else first_peak_gpu
    peak_host = warm_peak_host if warm_peak_host is not None else first_peak_host

    metrics = {
        "profile": profile["name"],
        "task": "align",
        "loss_kind": cfg.loss_kind,
        "success": True,
        "first_run_seconds": first.seconds,
        "warm_run_seconds_mean": float(statistics.mean(warm_seconds)),
        "warm_run_seconds_std": float(statistics.pstdev(warm_seconds) if len(warm_seconds) > 1 else 0.0),
        "first_run_peak_gpu_memory_mb": first_peak_gpu,
        "warm_run_peak_gpu_memory_mb_max": warm_peak_gpu,
        "peak_gpu_memory_mb": peak_gpu,
        "first_run_peak_gpu_memory_process_mb": first_peak_gpu_process,
        "warm_run_peak_gpu_memory_process_mb_max": warm_peak_gpu_process,
        "peak_gpu_memory_process_mb": (
            warm_peak_gpu_process if warm_peak_gpu_process is not None else first_peak_gpu_process
        ),
        "first_run_peak_gpu_memory_device_mb": first_peak_gpu_device,
        "warm_run_peak_gpu_memory_device_mb_max": warm_peak_gpu_device,
        "peak_gpu_memory_device_mb": (
            warm_peak_gpu_device if warm_peak_gpu_device is not None else first_peak_gpu_device
        ),
        "first_run_peak_host_rss_mb": first_peak_host,
        "warm_run_peak_host_rss_mb_max": warm_peak_host,
        "peak_host_rss_mb": peak_host,
        "gpu_memory_backend": first.gpu_memory_backend,
        "gpu_memory_scope": (
            "process"
            if (warm_peak_gpu_process is not None or first_peak_gpu_process is not None)
            else first.gpu_memory_scope
        ),
        "gpu_memory_sample_interval_seconds": first.gpu_memory_sample_interval_seconds,
        "gpu_memory_sample_count": int(
            first.gpu_memory_sample_count + sum(w.gpu_memory_sample_count for w in warms)
        ),
        "gpu_memory_observed_gpu_count": max(
            [first.gpu_memory_observed_gpu_count, *[w.gpu_memory_observed_gpu_count for w in warms]]
        ),
        "gpu_memory_process_source": first.gpu_memory_process_source,
        "gpu_memory_process_supported": bool(
            first.gpu_memory_process_supported or any(w.gpu_memory_process_supported for w in warms)
        ),
        "jax_device_memory_profile_path": jax_profile_path,
        "jax_device_memory_profile_error": jax_profile_error,
        "quality": {
            **abs_metrics,
            **rel_metrics,
            **gf_metrics,
            "gt_mse": gt_mse,
        },
        "time_budget_seconds": time_budget_seconds,
        "warm_gt_mse_mean": _mean_or_none(warm_gt_mse_values),
        "warm_gt_mse_median": _median_or_none(warm_gt_mse_values),
        "warm_gt_mse_std": (
            float(statistics.pstdev(warm_gt_mse_values)) if len(warm_gt_mse_values) > 1 else 0.0
        ),
        "warm_trans_rmse_px_mean": _mean_or_none(warm_trans_rmse_values),
        "warm_trans_rmse_px_median": _median_or_none(warm_trans_rmse_values),
        "gpu_sampler_error": first.gpu_sampler_error or next((w.gpu_sampler_error for w in warms if w.gpu_sampler_error), None),
        "summary_image_path": None,
        "summary_image_error": None,
    }
    if first_convergence is not None or warm_convergences:
        warm_aggregate = _aggregate_warm_convergence_runs(
            warm_convergences,
            required_successes=convergence.required_warm_successes,
        )
        metrics.update(
            {
                "quality_threshold_metric": convergence.metric,
                "quality_threshold_value": convergence.threshold,
                "quality_threshold_met": (
                    bool(warm_aggregate["quality_threshold_met"])
                    if convergence.threshold is not None
                    else None
                ),
                "required_warm_successes": int(convergence.required_warm_successes),
                "warm_threshold_hit_count": (
                    warm_aggregate["warm_threshold_hit_count"]
                    if convergence.threshold is not None
                    else None
                ),
                "warm_threshold_total_runs": (
                    warm_aggregate["warm_threshold_total_runs"]
                    if convergence.threshold is not None
                    else None
                ),
                "warm_threshold_success_rate": (
                    warm_aggregate["warm_threshold_success_rate"]
                    if convergence.threshold is not None
                    else None
                ),
                "stopped_on_threshold": warm_aggregate["stopped_on_threshold"],
                "stopped_on_plateau": warm_aggregate["stopped_on_plateau"],
                "stopped_on_budget": warm_aggregate["stopped_on_budget"],
                "warm_stopped_on_threshold_count": warm_aggregate["warm_stopped_on_threshold_count"],
                "warm_stopped_on_plateau_count": warm_aggregate["warm_stopped_on_plateau_count"],
                "warm_stopped_on_budget_count": warm_aggregate["warm_stopped_on_budget_count"],
                "final_stop_reason": warm_aggregate["final_stop_reason"],
                "final_stop_level_factor": warm_aggregate["final_stop_level_factor"],
                "first_threshold_crossing_level_factor": (
                    warm_aggregate["first_threshold_crossing_level_factor"]
                    if convergence.threshold is not None
                    else None
                ),
                "cold_seconds_to_quality_threshold": (
                    first_convergence.seconds_to_threshold
                    if first_convergence is not None and convergence.threshold is not None
                    else None
                ),
                "warm_seconds_to_quality_threshold": (
                    warm_aggregate["warm_seconds_to_quality_threshold"]
                    if convergence.threshold is not None
                    else None
                ),
                "cold_outer_iters_to_quality_threshold": (
                    first_convergence.outer_iters_to_threshold
                    if first_convergence is not None and convergence.threshold is not None
                    else None
                ),
                "warm_outer_iters_to_quality_threshold": (
                    warm_aggregate["warm_outer_iters_to_quality_threshold"]
                    if convergence.threshold is not None
                    else None
                ),
                "cold_best_quality_value": (
                    first_convergence.best_quality_value if first_convergence is not None else None
                ),
                "warm_best_quality_value": warm_aggregate["warm_best_quality_value"],
                "best_quality_value": (
                    warm_aggregate["best_quality_value"]
                    if warm_aggregate["best_quality_value"] is not None
                    else (first_convergence.best_quality_value if first_convergence is not None else None)
                ),
                "best_quality_elapsed_seconds": (
                    warm_aggregate["best_quality_elapsed_seconds"]
                    if warm_aggregate["best_quality_elapsed_seconds"] is not None
                    else (
                        first_convergence.best_quality_elapsed_seconds
                        if first_convergence is not None
                        else None
                    )
                ),
                "cold_total_outer_iters_executed": (
                    first_convergence.total_outer_iters_executed if first_convergence is not None else None
                ),
                "warm_total_outer_iters_executed": warm_aggregate["warm_total_outer_iters_executed"],
                "total_outer_iters_executed": (
                    warm_aggregate["total_outer_iters_executed"]
                    if warm_aggregate["total_outer_iters_executed"] is not None
                    else (first_convergence.total_outer_iters_executed if first_convergence is not None else None)
                ),
                "cold_convergence_trace": (
                    first_convergence.trace if first_convergence is not None else []
                ),
                "warm_convergence_trace": warm_aggregate["warm_convergence_trace"],
                "warm_convergence_traces": warm_aggregate["warm_convergence_traces"],
            }
        )
    if _should_render_alignment_summary(profile):
        summary_path = _alignment_summary_path(out_path)
        try:
            baseline_volume = _alignment_baseline_volume(bundle, align_cfg, mods)
            save_alignment_summary(
                out_path=summary_path,
                profile_name=str(profile["name"]),
                gt_volume=np.asarray(bundle.volume, dtype=np.float32),
                baseline_volume=baseline_volume,
                final_volume=final_volume,
                loss_history=[float(v) for v in list(final_info.get("loss") or [])],
                convergence_trace=metrics.get("warm_convergence_trace"),
                convergence_metric_name=metrics.get("quality_threshold_metric"),
                quality_threshold_value=metrics.get("quality_threshold_value"),
                metrics=metrics,
                quality=dict(metrics["quality"]),
                fixture=bundle.shape_summary,
            )
            metrics["summary_image_path"] = str(summary_path)
        except Exception as exc:
            metrics["summary_image_error"] = str(exc)
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



def execute_profile(
    *,
    profile_arg: str,
    out_path: Path,
    profile_root: str = str(PROFILES_DIR),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    out_path = out_path.resolve()
    _ensure_dir(out_path.parent)
    _ensure_dir(DATA_DIR)
    _ensure_dir(OUT_DIR)

    profile_path = _resolve_profile_path(profile_arg, profile_root)
    profile = _load_profile(profile_path)
    env_updates = _configure_environment(profile)
    _emit_progress(
        progress_callback,
        stage_kind="profile_loading",
        phase="profile_loading",
        task=str(profile.get("task", "recon")),
        message=f"Loaded benchmark profile {profile['name']}.",
        detail=f"profile_path={profile_path.name}",
    )

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
        "peak_gpu_memory_process_mb": None,
        "peak_gpu_memory_device_mb": None,
        "peak_host_rss_mb": None,
        "gpu_memory_backend": None,
        "gpu_memory_scope": None,
        "gpu_memory_process_source": None,
        "gpu_memory_process_supported": None,
        "gpu_memory_sample_interval_seconds": None,
        "gpu_memory_sample_count": None,
        "gpu_memory_observed_gpu_count": None,
        "gpu_sampler_error": None,
        "jax_device_memory_profile_path": None,
        "jax_device_memory_profile_error": None,
        "summary_image_path": None,
        "summary_image_error": None,
        "quality_threshold_metric": None,
        "quality_threshold_value": None,
        "quality_threshold_met": None,
        "time_budget_seconds": None,
        "required_warm_successes": None,
        "warm_threshold_hit_count": None,
        "warm_threshold_total_runs": None,
        "warm_threshold_success_rate": None,
        "warm_gt_mse_mean": None,
        "warm_gt_mse_median": None,
        "warm_gt_mse_std": None,
        "warm_trans_rmse_px_mean": None,
        "warm_trans_rmse_px_median": None,
        "stopped_on_threshold": None,
        "stopped_on_plateau": None,
        "stopped_on_budget": None,
        "warm_stopped_on_threshold_count": None,
        "warm_stopped_on_plateau_count": None,
        "warm_stopped_on_budget_count": None,
        "final_stop_reason": None,
        "final_stop_level_factor": None,
        "first_threshold_crossing_level_factor": None,
        "cold_seconds_to_quality_threshold": None,
        "warm_seconds_to_quality_threshold": None,
        "cold_outer_iters_to_quality_threshold": None,
        "warm_outer_iters_to_quality_threshold": None,
        "best_quality_value": None,
        "best_quality_elapsed_seconds": None,
        "total_outer_iters_executed": None,
        "cold_convergence_trace": [],
        "warm_convergence_trace": [],
        "warm_convergence_traces": [],
        "quality": {},
        "device": {},
        "oom": False,
        "error": None,
        "profile_path": str(profile_path),
        "env": env_updates,
    }

    try:
        mods = _import_modules(profile)
        fixture, fixture_generated, fixture_path = _ensure_fixture(
            profile,
            mods,
            progress_callback=progress_callback,
        )
        if str(profile.get("task", "recon")) == "align":
            run_metrics = _run_align_profile(
                fixture,
                profile,
                mods,
                out_path,
                progress_callback=progress_callback,
            )
        else:
            run_metrics = _run_recon_profile(
                fixture,
                profile,
                mods,
                out_path,
                progress_callback=progress_callback,
            )

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
        metrics["success"] = bool(
            run_metrics.get(
                "success",
                metrics.get("objective_value") is not None or objective_name == "warm_run_seconds_mean",
            )
        )
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
    return metrics


def main() -> int:
    args = _parse_args()
    metrics = execute_profile(
        profile_arg=args.profile,
        out_path=Path(args.out),
        profile_root=args.profile_root,
    )
    return 0 if metrics.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
