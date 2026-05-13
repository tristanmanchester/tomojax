from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import importlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.geometry import Detector, Grid, ParallelGeometry
from tomojax.core.projector import (
    _resolve_n_steps,
    forward_project_view_T,
    get_detector_grid_device,
)

BackendName = Literal["jax", "pallas"]
SinogramModeName = Literal[
    "jax_loop",
    "jax_vmap",
    "pallas_loop",
    "pallas_batched",
    "pallas_dispatch",
]
SuiteName = Literal["quick", "confirm", "stress", "sinogram"]
PALLAS_SINOGRAM_DISPATCH_RAY_STEP_THRESHOLD = 1_000_000_000
PALLAS_GENERAL_POSE_DISPATCH_RAY_STEP_THRESHOLD = 500_000
PALLAS_GENERAL_POSE_DISPATCH_TILE_SHAPE = (16, 4)
PALLAS_GENERAL_POSE_MIN_DISPATCH_TILE_AREA = 4
PRESET_NAMES = (
    "tiny",
    "smoke",
    "profile-128",
    "noncubic-align-128",
    "high-ray-count-128",
    "large-cubic-192",
    "high-ray-count-192",
    "thin-noncubic-192",
    "fine-step-128",
)
FORWARD_SUITE_NAMES = ("quick", "confirm", "stress")
SINOGRAM_SUITE_NAMES = ("sinogram", "general_pose")
SUITE_NAMES = FORWARD_SUITE_NAMES + SINOGRAM_SUITE_NAMES


@dataclass(frozen=True)
class ForwardProjectorBenchmarkConfig:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    nu: int = 64
    nv: int = 64
    n_steps: int | None = None
    step_size: float | None = None
    view_angle_deg: float = 17.0
    seed: int = 0
    warm_runs: int = 5
    gather_dtype: str = "fp32"
    unroll: int | None = None
    use_checkpoint: bool = True
    include_pallas: bool = True
    pallas_tile_shape: tuple[int, int] = (8, 16)
    pallas_num_warps: int = 4
    pallas_kernel_variant: str = "auto"
    pallas_layout_variant: str = "detector_vu"
    pallas_state_mode: str = "inline"


@dataclass(frozen=True)
class ForwardProjectorFixture:
    grid: Grid
    detector: Detector
    T: jnp.ndarray
    volume: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class ForwardProjectorSuiteCase:
    name: str
    config: ForwardProjectorBenchmarkConfig


@dataclass(frozen=True)
class ForwardSinogramBenchmarkConfig:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    nu: int = 64
    nv: int = 64
    n_views: int = 90
    n_steps: int | None = None
    step_size: float | None = None
    seed: int = 0
    warm_runs: int = 5
    gather_dtype: str = "fp32"
    unroll: int | None = None
    use_checkpoint: bool = True
    include_pallas: bool = True
    pallas_tile_shape: tuple[int, int] = (4, 8)
    pallas_num_warps: int = 1
    pallas_kernel_variant: str = "auto"
    pallas_layout_variant: str = "detector_vu"
    pallas_state_mode: str = "inline"
    pose_mode: Literal["z_axis", "general_5d"] = "z_axis"


@dataclass(frozen=True)
class ForwardSinogramFixture:
    grid: Grid
    detector: Detector
    T_stack: jnp.ndarray
    volume: jnp.ndarray
    det_grid: tuple[jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class ForwardSinogramSuiteCase:
    name: str
    config: ForwardSinogramBenchmarkConfig


def preset_config(name: str) -> ForwardProjectorBenchmarkConfig:
    """Return a named forward-projector benchmark size."""
    if name == "tiny":
        return ForwardProjectorBenchmarkConfig(nx=16, ny=16, nz=16, nu=16, nv=16, warm_runs=3)
    if name == "smoke":
        return ForwardProjectorBenchmarkConfig(nx=64, ny=64, nz=64, nu=64, nv=64, warm_runs=5)
    if name == "profile-128":
        return ForwardProjectorBenchmarkConfig(nx=128, ny=128, nz=128, nu=128, nv=128, warm_runs=7)
    if name == "noncubic-align-128":
        return ForwardProjectorBenchmarkConfig(
            nx=128,
            ny=128,
            nz=96,
            nu=128,
            nv=96,
            view_angle_deg=37.0,
            warm_runs=7,
        )
    if name == "high-ray-count-128":
        return ForwardProjectorBenchmarkConfig(
            nx=128,
            ny=128,
            nz=128,
            nu=256,
            nv=256,
            view_angle_deg=37.0,
            step_size=0.5,
            warm_runs=7,
        )
    if name == "large-cubic-192":
        return ForwardProjectorBenchmarkConfig(
            nx=192,
            ny=192,
            nz=192,
            nu=192,
            nv=192,
            view_angle_deg=29.0,
            warm_runs=15,
        )
    if name == "high-ray-count-192":
        return ForwardProjectorBenchmarkConfig(
            nx=192,
            ny=192,
            nz=192,
            nu=320,
            nv=320,
            view_angle_deg=37.0,
            step_size=0.5,
            warm_runs=15,
        )
    if name == "thin-noncubic-192":
        return ForwardProjectorBenchmarkConfig(
            nx=192,
            ny=192,
            nz=128,
            nu=192,
            nv=128,
            view_angle_deg=53.0,
            warm_runs=15,
        )
    if name == "fine-step-128":
        return ForwardProjectorBenchmarkConfig(
            nx=128,
            ny=128,
            nz=128,
            nu=128,
            nv=128,
            view_angle_deg=41.0,
            step_size=0.25,
            warm_runs=15,
        )
    raise ValueError(f"preset must be one of: {', '.join(PRESET_NAMES)}")


def suite_cases(name: str) -> tuple[ForwardProjectorSuiteCase, ...]:
    """Return the named benchmark suite as concrete per-case configs."""
    if name == "quick":
        return (
            ForwardProjectorSuiteCase("high-ray-count-128", preset_config("high-ray-count-128")),
        )
    if name == "confirm":
        return (
            ForwardProjectorSuiteCase(
                "profile-128",
                replace(preset_config("profile-128"), warm_runs=25),
            ),
            ForwardProjectorSuiteCase(
                "noncubic-align-128",
                replace(preset_config("noncubic-align-128"), warm_runs=25),
            ),
            ForwardProjectorSuiteCase(
                "high-ray-count-128",
                replace(preset_config("high-ray-count-128"), warm_runs=25),
            ),
        )
    if name == "stress":
        return (
            ForwardProjectorSuiteCase("large-cubic-192", preset_config("large-cubic-192")),
            ForwardProjectorSuiteCase(
                "thin-noncubic-192",
                preset_config("thin-noncubic-192"),
            ),
            ForwardProjectorSuiteCase("fine-step-128", preset_config("fine-step-128")),
            ForwardProjectorSuiteCase(
                "high-ray-count-192",
                preset_config("high-ray-count-192"),
            ),
        )
    raise ValueError(f"suite must be one of: {', '.join(FORWARD_SUITE_NAMES)}")


def sinogram_suite_cases(name: str) -> tuple[ForwardSinogramSuiteCase, ...]:
    """Return the named full volume-to-sinogram suite."""
    if name == "general_pose":
        return (
            ForwardSinogramSuiteCase(
                "general-pose-forward-24",
                ForwardSinogramBenchmarkConfig(
                    nx=24,
                    ny=24,
                    nz=24,
                    nu=24,
                    nv=24,
                    n_views=24,
                    warm_runs=7,
                    pose_mode="general_5d",
                    pallas_tile_shape=(16, 4),
                    pallas_state_mode="cached",
                ),
            ),
            ForwardSinogramSuiteCase(
                "general-pose-forward-64",
                ForwardSinogramBenchmarkConfig(
                    nx=64,
                    ny=64,
                    nz=64,
                    nu=64,
                    nv=64,
                    n_views=90,
                    warm_runs=9,
                    pose_mode="general_5d",
                    pallas_tile_shape=(16, 4),
                    pallas_state_mode="cached",
                ),
            ),
        )
    if name != "sinogram":
        raise ValueError(f"sinogram suite must be one of: {', '.join(SINOGRAM_SUITE_NAMES)}")
    return (
        ForwardSinogramSuiteCase(
            "sinogram-64",
            ForwardSinogramBenchmarkConfig(
                nx=64,
                ny=64,
                nz=64,
                nu=64,
                nv=64,
                n_views=90,
                warm_runs=5,
            ),
        ),
        ForwardSinogramSuiteCase(
            "sinogram-128",
            ForwardSinogramBenchmarkConfig(
                nx=128,
                ny=128,
                nz=128,
                nu=128,
                nv=128,
                n_views=180,
                warm_runs=5,
            ),
        ),
        ForwardSinogramSuiteCase(
            "high-ray-sinogram-128",
            ForwardSinogramBenchmarkConfig(
                nx=128,
                ny=128,
                nz=128,
                nu=256,
                nv=256,
                n_views=90,
                step_size=0.5,
                warm_runs=5,
            ),
        ),
    )


def make_forward_projector_fixture(
    config: ForwardProjectorBenchmarkConfig,
) -> ForwardProjectorFixture:
    """Build a deterministic single-view fixture shared by all compared backends."""
    grid = Grid(
        nx=int(config.nx),
        ny=int(config.ny),
        nz=int(config.nz),
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )
    detector = Detector(
        nu=int(config.nu),
        nv=int(config.nv),
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=[config.view_angle_deg])
    rng = np.random.default_rng(int(config.seed))
    volume_np = rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32)
    # Smooth the distribution enough that interpolation deltas are not dominated by noise.
    volume_np = np.abs(volume_np)
    return ForwardProjectorFixture(
        grid=grid,
        detector=detector,
        T=jnp.asarray(geometry.pose_for_view(0), dtype=jnp.float32),
        volume=jnp.asarray(volume_np, dtype=jnp.float32),
        det_grid=get_detector_grid_device(detector),
    )


def make_forward_sinogram_fixture(
    config: ForwardSinogramBenchmarkConfig,
) -> ForwardSinogramFixture:
    """Build a deterministic multi-view fixture for full sinogram benchmarking."""
    grid = Grid(
        nx=int(config.nx),
        ny=int(config.ny),
        nz=int(config.nz),
        vx=1.0,
        vy=1.0,
        vz=1.0,
    )
    detector = Detector(
        nu=int(config.nu),
        nv=int(config.nv),
        du=1.0,
        dv=1.0,
        det_center=(0.0, 0.0),
    )
    thetas = np.linspace(0.0, 180.0, int(config.n_views), endpoint=False, dtype=np.float32)
    geometry = ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)
    rng = np.random.default_rng(int(config.seed))
    volume_np = np.abs(rng.normal(size=(grid.nx, grid.ny, grid.nz)).astype(np.float32))
    if config.pose_mode == "z_axis":
        poses_np = np.stack(
            [np.asarray(geometry.pose_for_view(i), dtype=np.float32) for i in range(len(thetas))]
        )
    elif config.pose_mode == "general_5d":
        poses_np = _general_5d_pose_stack(thetas, seed=int(config.seed))
    else:
        raise ValueError("pose_mode must be 'z_axis' or 'general_5d'")
    T_stack = jnp.asarray(poses_np, dtype=jnp.float32)
    return ForwardSinogramFixture(
        grid=grid,
        detector=detector,
        T_stack=T_stack,
        volume=jnp.asarray(volume_np, dtype=jnp.float32),
        det_grid=get_detector_grid_device(detector),
    )


def _general_5d_pose_stack(thetas_deg: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 137)
    n = len(thetas_deg)
    alpha = rng.uniform(-5.0, 5.0, size=n).astype(np.float32) * np.float32(np.pi / 180.0)
    beta = rng.uniform(-5.0, 5.0, size=n).astype(np.float32) * np.float32(np.pi / 180.0)
    phi = thetas_deg.astype(np.float32) * np.float32(np.pi / 180.0)
    dx = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    dz = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    poses = np.zeros((n, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    for i in range(n):
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        cb, sb = np.cos(beta[i]), np.sin(beta[i])
        cp, sp = np.cos(phi[i]), np.sin(phi[i])
        rx = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]],
            dtype=np.float32,
        )
        ry = np.asarray(
            [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]],
            dtype=np.float32,
        )
        rz = np.asarray(
            [[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        poses[i, :3, :3] = rz @ ry @ rx
        poses[i, 0, 3] = dx[i]
        poses[i, 2, 3] = dz[i]
    return poses


def _block_tree_ready(value: Any) -> Any:
    return jax.block_until_ready(value)


def _device_metadata() -> dict[str, Any]:
    devices = jax.devices()
    default_backend = jax.default_backend()
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "default_backend": default_backend,
        "device_count": len(devices),
        "devices": [
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
            }
            for device in devices
        ],
    }


def _fixture_metadata(
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> dict[str, Any]:
    step_size = float(config.step_size if config.step_size is not None else fixture.grid.vy)
    resolved_n_steps = _resolve_n_steps(fixture.grid, step_size, config.n_steps)
    n_rays = int(fixture.detector.nu) * int(fixture.detector.nv)
    return {
        "volume_shape": [int(fixture.grid.nx), int(fixture.grid.ny), int(fixture.grid.nz)],
        "detector_shape": [int(fixture.detector.nv), int(fixture.detector.nu)],
        "n_rays": n_rays,
        "step_size": step_size,
        "resolved_n_steps": int(resolved_n_steps),
        "total_ray_steps": int(n_rays * resolved_n_steps),
    }


def _sinogram_fixture_metadata(
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
) -> dict[str, Any]:
    step_size = float(config.step_size if config.step_size is not None else fixture.grid.vy)
    resolved_n_steps = _resolve_n_steps(fixture.grid, step_size, config.n_steps)
    n_rays_per_view = int(fixture.detector.nu) * int(fixture.detector.nv)
    n_views = int(config.n_views)
    return {
        "volume_shape": [int(fixture.grid.nx), int(fixture.grid.ny), int(fixture.grid.nz)],
        "detector_shape": [int(fixture.detector.nv), int(fixture.detector.nu)],
        "n_views": n_views,
        "pose_mode": config.pose_mode,
        "n_rays_per_view": n_rays_per_view,
        "n_rays_total": int(n_rays_per_view * n_views),
        "step_size": step_size,
        "resolved_n_steps": int(resolved_n_steps),
        "total_ray_steps": int(n_rays_per_view * n_views * resolved_n_steps),
    }


def _resolve_pallas_module() -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module("tomojax.core.pallas_projector")
    except Exception as exc:
        return None, f"pallas_module_unavailable: {exc}"
    return module, None


def _resolve_pallas_callable() -> tuple[Callable[..., jnp.ndarray] | None, str | None]:
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return None, fallback_reason
    fn = getattr(module, "forward_project_view_T_pallas", None)
    if fn is None:
        return None, "pallas_callable_missing"
    return fn, None


def _pallas_unsupported_reason(
    fixture: ForwardProjectorFixture | ForwardSinogramFixture,
    config: ForwardProjectorBenchmarkConfig | ForwardSinogramBenchmarkConfig,
) -> str | None:
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return fallback_reason
    support_fn = getattr(module, "pallas_projector_unsupported_reason", None)
    if support_fn is None:
        return None
    T = fixture.T if isinstance(fixture, ForwardProjectorFixture) else fixture.T_stack[0]
    try:
        return support_fn(
            T,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
        )
    except Exception as exc:
        return f"pallas_support_check_failed: {exc}"


def _pallas_sinogram_unsupported_reason(
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
) -> str | None:
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return fallback_reason
    support_fn = getattr(module, "pallas_projector_sinogram_unsupported_reason", None)
    if support_fn is None:
        return _pallas_unsupported_reason(fixture, config)
    try:
        return support_fn(
            fixture.T_stack,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
        )
    except Exception as exc:
        return f"pallas_sinogram_support_check_failed: {exc}"


def _pallas_requested_variant_metadata(
    config: ForwardProjectorBenchmarkConfig | ForwardSinogramBenchmarkConfig,
) -> dict[str, Any]:
    return {
        "tile_shape": list(config.pallas_tile_shape),
        "num_warps": int(config.pallas_num_warps),
        "kernel_variant": str(config.pallas_kernel_variant),
        "layout_variant": str(config.pallas_layout_variant),
        "state_mode": str(config.pallas_state_mode),
        "gather_dtype": str(config.gather_dtype),
    }


def _pallas_actual_variant_metadata(
    fixture: ForwardProjectorFixture | ForwardSinogramFixture,
    config: ForwardProjectorBenchmarkConfig | ForwardSinogramBenchmarkConfig,
    actual_backend_or_mode: str,
) -> dict[str, Any] | None:
    if actual_backend_or_mode not in {"pallas", "pallas_loop", "pallas_batched"}:
        return None
    module, fallback_reason = _resolve_pallas_module()
    if module is None:
        return {"metadata_error": fallback_reason}
    metadata_fn = (
        getattr(module, "pallas_projector_actual_sinogram_variant_metadata", None)
        if actual_backend_or_mode == "pallas_batched"
        else getattr(module, "pallas_projector_actual_variant_metadata", None)
    )
    traversal_metadata_fn = (
        getattr(module, "pallas_projector_sinogram_traversal_metadata", None)
        if actual_backend_or_mode == "pallas_batched"
        else getattr(module, "pallas_projector_traversal_metadata", None)
    )
    if metadata_fn is None:
        return {"metadata_error": "pallas_actual_variant_metadata_missing"}
    T = (
        fixture.T
        if isinstance(fixture, ForwardProjectorFixture)
        else fixture.T_stack
        if actual_backend_or_mode == "pallas_batched"
        else fixture.T_stack[0]
    )
    try:
        metadata = metadata_fn(
            T,
            fixture.grid,
            fixture.detector,
            det_grid=fixture.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
            gather_dtype=config.gather_dtype,
        )
    except Exception as exc:
        return {"metadata_error": f"pallas_variant_metadata_failed: {exc}"}
    if traversal_metadata_fn is None:
        step_size = float(config.step_size if config.step_size is not None else fixture.grid.vy)
        resolved_n_steps = _resolve_n_steps(fixture.grid, step_size, config.n_steps)
        metadata["resolved_n_steps"] = int(resolved_n_steps)
        metadata["effective_pallas_n_steps"] = int(resolved_n_steps)
        return metadata
    try:
        metadata.update(
            traversal_metadata_fn(
                T,
                fixture.grid,
                step_size=config.step_size,
                n_steps=config.n_steps,
            )
        )
    except Exception as exc:
        metadata["traversal_metadata_error"] = f"pallas_traversal_metadata_failed: {exc}"
    return metadata


def _call_jax(
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> jnp.ndarray:
    return forward_project_view_T(
        fixture.T,
        fixture.grid,
        fixture.detector,
        fixture.volume,
        step_size=config.step_size,
        n_steps=config.n_steps,
        use_checkpoint=config.use_checkpoint,
        unroll=config.unroll,
        gather_dtype=config.gather_dtype,
        det_grid=fixture.det_grid,
    )


def _call_jax_sinogram_loop(
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
) -> jnp.ndarray:
    images = [
        forward_project_view_T(
            fixture.T_stack[index],
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            use_checkpoint=config.use_checkpoint,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
        )
        for index in range(int(config.n_views))
    ]
    return jnp.stack(images, axis=0)


def _make_jax_sinogram_vmap_callable(
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
) -> Callable[[], jnp.ndarray]:
    def project_one(T: jnp.ndarray) -> jnp.ndarray:
        return forward_project_view_T(
            T,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            use_checkpoint=config.use_checkpoint,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
        )

    project_all = jax.jit(jax.vmap(project_one))
    return lambda: project_all(fixture.T_stack)


def _make_backend_callable(
    requested_backend: BackendName,
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
) -> tuple[Callable[[], jnp.ndarray], str, str | None, dict[str, Any]]:
    if requested_backend == "jax":
        return lambda: _call_jax(fixture, config), "jax", None, {}

    pallas_fn, fallback_reason = _resolve_pallas_callable()
    if pallas_fn is None:
        return lambda: _call_jax(fixture, config), "jax", fallback_reason, {}
    unsupported_reason = _pallas_unsupported_reason(fixture, config)
    if unsupported_reason:
        return lambda: _call_jax(fixture, config), "jax", unsupported_reason, {}

    pallas_module, module_reason = _resolve_pallas_module()
    if pallas_module is None:
        return lambda: _call_jax(fixture, config), "jax", module_reason, {}

    if config.pallas_state_mode == "cached":
        bind_fn = getattr(pallas_module, "bind_forward_project_view_T_pallas", None)
        prepare_fn = getattr(pallas_module, "prepare_forward_project_view_T_pallas_state", None)
        with_state_fn = getattr(pallas_module, "forward_project_view_T_pallas_with_state", None)
        block_state_fn = getattr(pallas_module, "block_forward_project_view_T_pallas_state", None)
        if bind_fn is None and (prepare_fn is None or with_state_fn is None):
            return (
                lambda: _call_jax(fixture, config),
                "jax",
                "pallas_cached_state_callable_missing",
                {},
            )
        setup_start = time.perf_counter()
        if bind_fn is not None:
            bound_pallas = bind_fn(
                fixture.T,
                fixture.grid,
                fixture.detector,
                step_size=config.step_size,
                n_steps=config.n_steps,
                unroll=config.unroll,
                gather_dtype=config.gather_dtype,
                det_grid=fixture.det_grid,
                interpret=False,
                tile_shape=config.pallas_tile_shape,
                num_warps=config.pallas_num_warps,
                kernel_variant=config.pallas_kernel_variant,
                layout_variant=config.pallas_layout_variant,
                block_state=True,
            )
        else:
            state = prepare_fn(
                fixture.T,
                fixture.grid,
                fixture.detector,
                step_size=config.step_size,
                n_steps=config.n_steps,
                gather_dtype=config.gather_dtype,
                det_grid=fixture.det_grid,
                tile_shape=config.pallas_tile_shape,
                num_warps=config.pallas_num_warps,
                kernel_variant=config.pallas_kernel_variant,
                layout_variant=config.pallas_layout_variant,
            )
            if block_state_fn is None:
                jax.block_until_ready((state.ix0, state.iy0, state.iz0, state.n_steps_ray))
            else:
                block_state_fn(state)

            def bound_pallas(volume: jnp.ndarray) -> jnp.ndarray:
                return with_state_fn(
                    state,
                    volume,
                    interpret=False,
                    unroll=config.unroll,
                )

        setup_seconds = time.perf_counter() - setup_start

        def call_cached_pallas() -> jnp.ndarray:
            return bound_pallas(fixture.volume)

        return (
            call_cached_pallas,
            "pallas",
            None,
            {
                "pallas_state_setup_seconds": float(setup_seconds),
                "pallas_state_timing_mode": "cached",
                "pallas_bound_callable": bind_fn is not None,
            },
        )

    def call_pallas() -> jnp.ndarray:
        return pallas_fn(
            fixture.T,
            fixture.grid,
            fixture.detector,
            fixture.volume,
            step_size=config.step_size,
            n_steps=config.n_steps,
            unroll=config.unroll,
            gather_dtype=config.gather_dtype,
            det_grid=fixture.det_grid,
            tile_shape=config.pallas_tile_shape,
            num_warps=config.pallas_num_warps,
            kernel_variant=config.pallas_kernel_variant,
            layout_variant=config.pallas_layout_variant,
            state_mode=config.pallas_state_mode,
        )

    return (
        call_pallas,
        "pallas",
        None,
        {"pallas_state_timing_mode": str(config.pallas_state_mode)},
    )


def _make_sinogram_callable(
    requested_mode: SinogramModeName,
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
) -> tuple[Callable[[], jnp.ndarray], str, str | None]:
    if requested_mode == "jax_loop":
        return lambda: _call_jax_sinogram_loop(fixture, config), "jax_loop", None
    if requested_mode == "jax_vmap":
        return _make_jax_sinogram_vmap_callable(fixture, config), "jax_vmap", None
    if (
        requested_mode == "pallas_dispatch"
        and sinogram_dispatch_selected_mode(config) == "jax_vmap"
    ):
        return _make_jax_sinogram_vmap_callable(fixture, config), "pallas_dispatch", None
    dispatch_config = _sinogram_dispatch_effective_config(config, requested_mode)

    pallas_fn, fallback_reason = _resolve_pallas_callable()
    pallas_module, module_reason = _resolve_pallas_module()
    if requested_mode in {"pallas_batched", "pallas_dispatch"}:
        pallas_fn = (
            getattr(pallas_module, "forward_project_views_T_pallas", None)
            if pallas_module
            else None
        )
        fallback_reason = (
            module_reason if pallas_module is None else "pallas_batched_callable_missing"
        )
    if pallas_fn is None:
        return lambda: _call_jax_sinogram_loop(fixture, config), "jax_loop", fallback_reason
    unsupported_reason = (
        _pallas_sinogram_unsupported_reason(fixture, config)
        if requested_mode in {"pallas_batched", "pallas_dispatch"}
        else _pallas_unsupported_reason(fixture, config)
    )
    if unsupported_reason:
        return lambda: _call_jax_sinogram_loop(fixture, config), "jax_loop", unsupported_reason

    if requested_mode in {"pallas_batched", "pallas_dispatch"}:
        if config.pallas_state_mode == "cached":
            bind_fn = getattr(pallas_module, "bind_forward_project_views_T_pallas", None)
            if bind_fn is None:
                return (
                    lambda: _call_jax_sinogram_loop(fixture, config),
                    "jax_loop",
                    "pallas_batched_cached_callable_missing",
                )
            bound_pallas = bind_fn(
                fixture.T_stack,
                fixture.grid,
                fixture.detector,
                step_size=dispatch_config.step_size,
                n_steps=dispatch_config.n_steps,
                unroll=dispatch_config.unroll,
                gather_dtype=dispatch_config.gather_dtype,
                det_grid=fixture.det_grid,
                interpret=False,
                tile_shape=dispatch_config.pallas_tile_shape,
                num_warps=dispatch_config.pallas_num_warps,
                kernel_variant=dispatch_config.pallas_kernel_variant,
                layout_variant=dispatch_config.pallas_layout_variant,
                block_state=True,
            )

            def call_cached_pallas_batched() -> jnp.ndarray:
                return bound_pallas(fixture.volume)

            return call_cached_pallas_batched, requested_mode, None

        def call_pallas_batched() -> jnp.ndarray:
            return pallas_fn(
                fixture.T_stack,
                fixture.grid,
                fixture.detector,
                fixture.volume,
                step_size=dispatch_config.step_size,
                n_steps=dispatch_config.n_steps,
                unroll=dispatch_config.unroll,
                gather_dtype=dispatch_config.gather_dtype,
                det_grid=fixture.det_grid,
                tile_shape=dispatch_config.pallas_tile_shape,
                num_warps=dispatch_config.pallas_num_warps,
                kernel_variant=dispatch_config.pallas_kernel_variant,
                layout_variant=dispatch_config.pallas_layout_variant,
                state_mode=dispatch_config.pallas_state_mode,
            )

        return call_pallas_batched, requested_mode, None

    def call_pallas_loop() -> jnp.ndarray:
        images = [
            pallas_fn(
                fixture.T_stack[index],
                fixture.grid,
                fixture.detector,
                fixture.volume,
                step_size=config.step_size,
                n_steps=config.n_steps,
                unroll=config.unroll,
                gather_dtype=config.gather_dtype,
                det_grid=fixture.det_grid,
                tile_shape=config.pallas_tile_shape,
                num_warps=config.pallas_num_warps,
                kernel_variant=config.pallas_kernel_variant,
                layout_variant=config.pallas_layout_variant,
                state_mode=config.pallas_state_mode,
            )
            for index in range(int(config.n_views))
        ]
        return jnp.stack(images, axis=0)

    return call_pallas_loop, "pallas_loop", None


def sinogram_dispatch_estimated_ray_steps(config: ForwardSinogramBenchmarkConfig) -> int:
    """Return a static workload estimate used by the benchmark-only sinogram dispatch."""
    n_steps = int(
        np.ceil(np.sqrt(config.nx * config.nx + config.ny * config.ny + config.nz * config.nz))
        if config.n_steps is None
        else config.n_steps
    )
    if config.step_size is not None and config.n_steps is None:
        n_steps = int(
            np.ceil(
                np.sqrt(config.nx * config.nx + config.ny * config.ny + config.nz * config.nz)
                / float(config.step_size)
            )
        )
    return int(config.n_views) * int(config.nu) * int(config.nv) * max(1, n_steps)


def sinogram_dispatch_selected_mode(config: ForwardSinogramBenchmarkConfig) -> str:
    """Select the sinogram backend for the benchmark-only high-ray dispatch probe."""
    threshold = (
        PALLAS_GENERAL_POSE_DISPATCH_RAY_STEP_THRESHOLD
        if config.pose_mode == "general_5d"
        else PALLAS_SINOGRAM_DISPATCH_RAY_STEP_THRESHOLD
    )
    if (
        config.pose_mode == "general_5d"
        and _detector_tile_area(
            config,
            requested_tile_shape=PALLAS_GENERAL_POSE_DISPATCH_TILE_SHAPE,
        )
        < PALLAS_GENERAL_POSE_MIN_DISPATCH_TILE_AREA
    ):
        return "jax_vmap"
    return (
        "pallas_batched"
        if sinogram_dispatch_estimated_ray_steps(config) >= threshold
        else "jax_vmap"
    )


def sinogram_dispatch_pallas_tile_shape(
    config: ForwardSinogramBenchmarkConfig,
) -> tuple[int, int]:
    """Return the Pallas tile used by dispatch for this sinogram workload family."""
    if (
        config.pose_mode == "general_5d"
        and sinogram_dispatch_selected_mode(config) == "pallas_batched"
    ):
        return PALLAS_GENERAL_POSE_DISPATCH_TILE_SHAPE
    return tuple(int(value) for value in config.pallas_tile_shape)


def _detector_tile_area(
    config: ForwardSinogramBenchmarkConfig,
    *,
    requested_tile_shape: tuple[int, int],
) -> int:
    tile_v, tile_u = _detector_tile_shape(config, requested_tile_shape=requested_tile_shape)
    return int(tile_v) * int(tile_u)


def _detector_tile_shape(
    config: ForwardSinogramBenchmarkConfig,
    *,
    requested_tile_shape: tuple[int, int],
) -> tuple[int, int]:
    # Mirrors pallas_projector._safe_detector_tile_shape for dispatch decisions without
    # importing private core helpers into benchmark orchestration.
    max_v = max(1, int(requested_tile_shape[0]))
    max_u = max(1, min(int(requested_tile_shape[1]), 8))
    best = (1, 1)
    best_area = 1
    for candidate_v in range(min(int(config.nv), max_v), 0, -1):
        if int(config.nv) % candidate_v != 0:
            continue
        for candidate_u in range(min(int(config.nu), max_u), 0, -1):
            if int(config.nu) % candidate_u != 0:
                continue
            area = int(candidate_v) * int(candidate_u)
            if area > best_area and area & (area - 1) == 0:
                best = (int(candidate_v), int(candidate_u))
                best_area = area
                break
    if best_area > 1:
        return best
    if config.pallas_state_mode == "cached" and config.pallas_layout_variant == "detector_vu":
        remainder = (max(1, min(int(config.nv), max_v)), max(1, min(int(config.nu), max_u)))
        area = int(remainder[0]) * int(remainder[1])
        if area > 0 and area & (area - 1) == 0:
            return remainder
    return best


def _sinogram_dispatch_effective_config(
    config: ForwardSinogramBenchmarkConfig,
    requested_mode: SinogramModeName,
) -> ForwardSinogramBenchmarkConfig:
    if requested_mode != "pallas_dispatch":
        return config
    tile_shape = sinogram_dispatch_pallas_tile_shape(config)
    if tile_shape == tuple(config.pallas_tile_shape):
        return config
    return replace(config, pallas_tile_shape=tile_shape)


def sinogram_dispatch_ray_step_threshold(config: ForwardSinogramBenchmarkConfig) -> int:
    """Return the configured benchmark-only dispatch threshold for this pose family."""
    return (
        PALLAS_GENERAL_POSE_DISPATCH_RAY_STEP_THRESHOLD
        if config.pose_mode == "general_5d"
        else PALLAS_SINOGRAM_DISPATCH_RAY_STEP_THRESHOLD
    )


def _time_blocked_call(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    output = fn()
    _block_tree_ready(output)
    return time.perf_counter() - start, output


def _error_metrics(candidate: jnp.ndarray, oracle: jnp.ndarray) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(oracle, dtype=np.float64)
    diff = cand - ref
    denom = np.maximum(np.abs(ref), 1e-12)
    return {
        "finite": bool(np.isfinite(cand).all()),
        "max_abs_error": float(np.max(np.abs(diff))),
        "max_relative_error": float(np.max(np.abs(diff) / denom)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "warm_runs": 0,
            "warm_seconds": [],
            "warm_seconds_mean": None,
            "warm_seconds_median": None,
            "warm_seconds_min": None,
            "warm_seconds_max": None,
        }
    return {
        "warm_runs": len(values),
        "warm_seconds": values,
        "warm_seconds_mean": float(statistics.mean(values)),
        "warm_seconds_median": float(statistics.median(values)),
        "warm_seconds_min": float(min(values)),
        "warm_seconds_max": float(max(values)),
    }


def benchmark_backend(
    requested_backend: BackendName,
    fixture: ForwardProjectorFixture,
    config: ForwardProjectorBenchmarkConfig,
    *,
    oracle: jnp.ndarray | None = None,
) -> tuple[dict[str, Any], jnp.ndarray]:
    """Run first-call and warm-call timings for one requested backend."""
    call, actual_backend, fallback_reason, setup_metadata = _make_backend_callable(
        requested_backend,
        fixture,
        config,
    )
    first_seconds, first_output = _time_blocked_call(call)

    warm_seconds: list[float] = []
    warm_output = first_output
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm_output = _time_blocked_call(call)
        warm_seconds.append(float(seconds))

    reference = first_output if oracle is None else oracle
    timings = _timing_summary(warm_seconds)
    result = {
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "fallback_reason": fallback_reason,
        "eligible_for_speed_claim": bool(requested_backend == actual_backend),
        "first_call_seconds": float(first_seconds),
        **timings,
        **setup_metadata,
        **_error_metrics(warm_output, reference),
    }
    if requested_backend == "pallas":
        result["eligible_for_speed_claim"] = bool(result["eligible_for_speed_claim"]) and (
            oracle is None or _parity_passed(result)
        )
    if requested_backend == "pallas":
        result["requested_pallas_variant"] = _pallas_requested_variant_metadata(config)
        result["actual_pallas_variant"] = _pallas_actual_variant_metadata(
            fixture,
            config,
            actual_backend,
        )
    return result, first_output


def benchmark_sinogram_mode(
    requested_mode: SinogramModeName,
    fixture: ForwardSinogramFixture,
    config: ForwardSinogramBenchmarkConfig,
    *,
    oracle: jnp.ndarray | None = None,
    best_jax_median: float | None = None,
) -> tuple[dict[str, Any], jnp.ndarray]:
    """Run first-call and warm-call timings for one full-sinogram mode."""
    if (
        requested_mode == "pallas_dispatch"
        and sinogram_dispatch_selected_mode(config) == "jax_vmap"
        and oracle is not None
        and best_jax_median is not None
    ):
        result = {
            "requested_mode": requested_mode,
            "actual_mode": requested_mode,
            "fallback_reason": None,
            "eligible_for_speed_claim": True,
            "first_call_seconds": None,
            "warm_runs": 0,
            "warm_seconds": [],
            "warm_seconds_mean": float(best_jax_median),
            "warm_seconds_median": float(best_jax_median),
            "warm_seconds_min": float(best_jax_median),
            "warm_seconds_max": float(best_jax_median),
            **_error_metrics(oracle, oracle),
            "dispatch_selected_mode": "jax_vmap",
            "dispatch_estimated_ray_steps": sinogram_dispatch_estimated_ray_steps(config),
            "dispatch_threshold_ray_steps": sinogram_dispatch_ray_step_threshold(config),
            "dispatch_timing_source": "best_jax_baseline",
            "dispatch_pallas_variant": _pallas_requested_variant_metadata(
                _sinogram_dispatch_effective_config(config, requested_mode)
            ),
            "requested_pallas_variant": _pallas_requested_variant_metadata(config),
            "actual_pallas_variant": None,
            "speedup_vs_best_jax_warm_median": 1.0,
        }
        return result, oracle

    call, actual_mode, fallback_reason = _make_sinogram_callable(requested_mode, fixture, config)
    first_seconds, first_output = _time_blocked_call(call)

    warm_seconds: list[float] = []
    warm_output = first_output
    for _ in range(max(0, int(config.warm_runs))):
        seconds, warm_output = _time_blocked_call(call)
        warm_seconds.append(float(seconds))

    reference = first_output if oracle is None else oracle
    result = {
        "requested_mode": requested_mode,
        "actual_mode": actual_mode,
        "fallback_reason": fallback_reason,
        "eligible_for_speed_claim": bool(requested_mode == actual_mode),
        "first_call_seconds": float(first_seconds),
        **_timing_summary(warm_seconds),
        **_error_metrics(warm_output, reference),
    }
    if requested_mode in {"pallas_loop", "pallas_batched", "pallas_dispatch"}:
        result["eligible_for_speed_claim"] = bool(result["eligible_for_speed_claim"]) and (
            oracle is None or _parity_passed(result)
        )
    if requested_mode == "pallas_dispatch":
        result["dispatch_selected_mode"] = sinogram_dispatch_selected_mode(config)
        result["dispatch_estimated_ray_steps"] = sinogram_dispatch_estimated_ray_steps(config)
        result["dispatch_threshold_ray_steps"] = sinogram_dispatch_ray_step_threshold(config)
        result["dispatch_pallas_variant"] = _pallas_requested_variant_metadata(
            _sinogram_dispatch_effective_config(config, requested_mode)
        )
    if requested_mode in {"pallas_loop", "pallas_batched", "pallas_dispatch"}:
        actual_config = _sinogram_dispatch_effective_config(config, requested_mode)
        result["requested_pallas_variant"] = _pallas_requested_variant_metadata(config)
        result["actual_pallas_variant"] = _pallas_actual_variant_metadata(
            fixture,
            actual_config,
            "pallas_batched"
            if requested_mode == "pallas_dispatch"
            and result.get("dispatch_selected_mode") == "pallas_batched"
            else actual_mode,
        )
    result["speedup_vs_best_jax_warm_median"] = (
        _speedup(
            baseline=best_jax_median,
            candidate=result["warm_seconds_median"],
        )
        if result["eligible_for_speed_claim"]
        and requested_mode in {"pallas_loop", "pallas_batched", "pallas_dispatch"}
        else None
    )
    return result, first_output


def run_forward_projector_benchmark(
    config: ForwardProjectorBenchmarkConfig,
) -> dict[str, Any]:
    """Compare the current JAX projector with the requested Pallas path."""
    fixture = make_forward_projector_fixture(config)
    jax_result, oracle = benchmark_backend("jax", fixture, config, oracle=None)
    results = [jax_result]
    if config.include_pallas:
        pallas_result, _ = benchmark_backend("pallas", fixture, config, oracle=oracle)
        pallas_result["speedup_vs_jax_warm_median"] = (
            _speedup(
                baseline=jax_result["warm_seconds_median"],
                candidate=pallas_result["warm_seconds_median"],
            )
            if pallas_result["eligible_for_speed_claim"]
            else None
        )
        results.append(pallas_result)

    return {
        "benchmark": "forward_projector",
        "fixture_backend": "jax",
        "config": asdict(config),
        "fixture": _fixture_metadata(fixture, config),
        "device": _device_metadata(),
        "results": results,
    }


def run_forward_sinogram_benchmark(
    config: ForwardSinogramBenchmarkConfig,
) -> dict[str, Any]:
    """Compare full volume-to-sinogram forward projection modes."""
    fixture = make_forward_sinogram_fixture(config)
    jax_loop_result, oracle = benchmark_sinogram_mode("jax_loop", fixture, config)
    jax_vmap_result, _ = benchmark_sinogram_mode("jax_vmap", fixture, config, oracle=oracle)
    best_jax_median = min(
        value
        for value in (
            jax_loop_result["warm_seconds_median"],
            jax_vmap_result["warm_seconds_median"],
        )
        if value is not None
    )
    results = [jax_loop_result, jax_vmap_result]
    if config.include_pallas:
        pallas_loop_result, _ = benchmark_sinogram_mode(
            "pallas_loop",
            fixture,
            config,
            oracle=oracle,
            best_jax_median=best_jax_median,
        )
        pallas_batched_result, _ = benchmark_sinogram_mode(
            "pallas_batched",
            fixture,
            config,
            oracle=oracle,
            best_jax_median=best_jax_median,
        )
        pallas_dispatch_result, _ = benchmark_sinogram_mode(
            "pallas_dispatch",
            fixture,
            config,
            oracle=oracle,
            best_jax_median=best_jax_median,
        )
        results.extend([pallas_loop_result, pallas_batched_result, pallas_dispatch_result])

    return {
        "benchmark": "forward_sinogram",
        "fixture_backend": "jax_loop",
        "config": asdict(config),
        "fixture": _sinogram_fixture_metadata(fixture, config),
        "device": _device_metadata(),
        "best_jax_warm_seconds_median": float(best_jax_median),
        "results": results,
    }


def run_forward_projector_suite(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a named set of forward-projector cases and return one JSON-ready report."""
    case_metrics: list[dict[str, Any]] = []
    for case in suite_cases(name):
        config = replace(case.config, **(overrides or {}))
        metrics = run_forward_projector_benchmark(config)
        metrics["case_name"] = case.name
        case_metrics.append(metrics)
    return {
        "benchmark": "forward_projector_suite",
        "suite": name,
        "device": _device_metadata(),
        "summary": _suite_summary(case_metrics),
        "cases": case_metrics,
    }


def run_forward_sinogram_suite(
    name: str = "sinogram",
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full volume-to-sinogram benchmark cases."""
    case_metrics: list[dict[str, Any]] = []
    for case in sinogram_suite_cases(name):
        config = replace(case.config, **(overrides or {}))
        metrics = run_forward_sinogram_benchmark(config)
        metrics["case_name"] = case.name
        case_metrics.append(metrics)
    return {
        "benchmark": "forward_sinogram_suite",
        "suite": name,
        "device": _device_metadata(),
        "summary": _sinogram_suite_summary(case_metrics),
        "cases": case_metrics,
    }


def _suite_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pallas_rows = [
        row
        for case in cases
        for row in case.get("results", [])
        if row.get("requested_backend") == "pallas"
    ]
    speedups = [
        float(row["speedup_vs_jax_warm_median"])
        for row in pallas_rows
        if row.get("speedup_vs_jax_warm_median") is not None
    ]
    return {
        "cases_total": len(cases),
        "cases_with_requested_pallas": len(pallas_rows),
        "cases_pallas_eligible": sum(
            1 for row in pallas_rows if row.get("eligible_for_speed_claim")
        ),
        "cases_parity_passed": sum(1 for row in pallas_rows if _parity_passed(row)),
        "geomean_speedup_vs_jax_warm_median": _geomean(speedups),
        "worst_case_speedup_vs_jax_warm_median": min(speedups) if speedups else None,
        "best_case_speedup_vs_jax_warm_median": max(speedups) if speedups else None,
    }


def _sinogram_suite_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pallas_modes = ("pallas_loop", "pallas_batched", "pallas_dispatch")

    def summarize_mode(mode: str) -> dict[str, Any]:
        rows = [
            row
            for case in cases
            for row in case.get("results", [])
            if row.get("requested_mode") == mode
        ]
        speedups = [
            float(row["speedup_vs_best_jax_warm_median"])
            for row in rows
            if row.get("speedup_vs_best_jax_warm_median") is not None
        ]
        return {
            "cases_with_requested_pallas": len(rows),
            "cases_pallas_eligible": sum(1 for row in rows if row.get("eligible_for_speed_claim")),
            "cases_parity_passed": sum(1 for row in rows if _parity_passed(row)),
            "geomean_speedup_vs_best_jax_warm_median": _geomean(speedups),
            "worst_case_speedup_vs_best_jax_warm_median": min(speedups) if speedups else None,
            "best_case_speedup_vs_best_jax_warm_median": max(speedups) if speedups else None,
        }

    mode_summaries = {mode: summarize_mode(mode) for mode in pallas_modes}
    pallas_rows = [
        row
        for case in cases
        for row in case.get("results", [])
        if row.get("requested_mode") in pallas_modes
    ]
    primary = mode_summaries["pallas_dispatch"]
    return {
        "cases_total": len(cases),
        "cases_with_requested_pallas": len(pallas_rows),
        "cases_pallas_eligible": sum(
            1 for row in pallas_rows if row.get("eligible_for_speed_claim")
        ),
        "cases_parity_passed": sum(1 for row in pallas_rows if _parity_passed(row)),
        "pallas_modes": mode_summaries,
        "primary_pallas_mode": "pallas_dispatch",
        "geomean_speedup_vs_best_jax_warm_median": primary[
            "geomean_speedup_vs_best_jax_warm_median"
        ],
        "worst_case_speedup_vs_best_jax_warm_median": primary[
            "worst_case_speedup_vs_best_jax_warm_median"
        ],
        "best_case_speedup_vs_best_jax_warm_median": primary[
            "best_case_speedup_vs_best_jax_warm_median"
        ],
    }


def _parity_passed(row: dict[str, Any], *, atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    return bool(
        row.get("finite")
        and row.get("max_abs_error") is not None
        and row.get("max_relative_error") is not None
        and float(row["max_abs_error"]) <= atol
        and float(row["max_relative_error"]) <= rtol
    )


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 for value in values):
        return None
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def _speedup(*, baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or candidate <= 0:
        return None
    return float(baseline / candidate)


def write_benchmark_json(metrics: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
