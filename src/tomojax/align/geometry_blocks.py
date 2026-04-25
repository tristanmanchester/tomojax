from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.calibration.axis_geometry import (
    axis_pose_stack,
    axis_unit_from_rotations,
    default_active_axis_dofs,
    nominal_axis_unit_from_inputs,
)
from tomojax.calibration.detector_grid import detector_grid_from_calibration
from tomojax.calibration.gauge import validate_calibration_gauges
from tomojax.calibration.state import CalibrationState, CalibrationVariable
from tomojax.core.geometry import Detector, Geometry, Grid, RotationAxisGeometry
from tomojax.core.geometry.lamino import LaminographyGeometry
from tomojax.core.geometry.parallel import ParallelGeometry
from tomojax.core.geometry.views import stack_view_poses
from tomojax.core.projector import forward_project_view_T
from tomojax.recon.fista_tv import FistaConfig, fista_tv


GEOMETRY_DOFS: tuple[str, ...] = (
    "det_u_px",
    "det_v_px",
    "detector_roll_deg",
    "axis_rot_x_deg",
    "axis_rot_y_deg",
    "tilt_deg",
)

GEOMETRY_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("detector_center", ("det_u_px", "det_v_px")),
    ("detector_roll", ("detector_roll_deg",)),
    ("axis_direction", ("axis_rot_x_deg", "axis_rot_y_deg")),
)


def normalize_geometry_dofs(
    values: Iterable[str] | None,
    *,
    geometry: Geometry | None = None,
) -> tuple[str, ...]:
    """Normalize detector/instrument geometry DOF names for staged alignment."""
    if values is None:
        return ()
    names: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            name = part.strip()
            if not name:
                continue
            if name == "tilt_deg":
                name = _tilt_alias_for_geometry(geometry)
            if name not in GEOMETRY_DOFS and name not in {"axis_rot_x_deg", "axis_rot_y_deg"}:
                allowed = ", ".join(GEOMETRY_DOFS)
                raise ValueError(f"Unknown geometry DOF {name!r}; expected one of: {allowed}")
            if name not in names:
                names.append(name)
    return tuple(names)


def _tilt_alias_for_geometry(geometry: Geometry | None) -> str:
    tilt_about = getattr(geometry, "tilt_about", "x")
    return "axis_rot_y_deg" if str(tilt_about) == "z" else "axis_rot_x_deg"


@dataclass(frozen=True)
class GeometryCalibrationState:
    """Native-resolution detector/instrument geometry state carried across levels."""

    det_u_px: float = 0.0
    det_v_px: float = 0.0
    detector_roll_deg: float = 0.0
    axis_rot_x_deg: float = 0.0
    axis_rot_y_deg: float = 0.0
    nominal_axis_unit: tuple[float, float, float] = (0.0, 0.0, 1.0)
    active_geometry_dofs: tuple[str, ...] = ()

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        *,
        active_geometry_dofs: Iterable[str] | None = None,
    ) -> "GeometryCalibrationState":
        geometry_inputs = geometry_inputs_from_geometry(geometry)
        active = normalize_geometry_dofs(active_geometry_dofs, geometry=geometry)
        nominal = tuple(float(v) for v in nominal_axis_unit_from_inputs(geometry_inputs))
        roll = float(geometry_inputs.get("detector_roll_deg", 0.0))
        state = cls(
            detector_roll_deg=roll,
            nominal_axis_unit=nominal,
            active_geometry_dofs=active,
        )
        validate_calibration_gauges(state.to_calibration_state())
        return state

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, object] | None,
        geometry: Geometry,
        *,
        active_geometry_dofs: Iterable[str] | None = None,
    ) -> "GeometryCalibrationState":
        state = cls.from_geometry(geometry, active_geometry_dofs=active_geometry_dofs)
        if not isinstance(payload, Mapping):
            return state

        values: dict[str, float] = {}
        for section_name in ("detector", "scan"):
            section = payload.get(section_name)
            if not isinstance(section, list | tuple):
                continue
            for item in section:
                if not isinstance(item, Mapping):
                    continue
                name = item.get("name")
                if name in {
                    "det_u_px",
                    "det_v_px",
                    "detector_roll_deg",
                    "axis_rot_x_deg",
                    "axis_rot_y_deg",
                }:
                    raw_value = item.get("value")
                    if isinstance(raw_value, int | float):
                        values[str(name)] = float(raw_value)
        return replace(state, **values)

    def replace_values(self, names: Sequence[str], values: jnp.ndarray) -> "GeometryCalibrationState":
        updates: dict[str, float] = {}
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        for idx, name in enumerate(names):
            updates[str(name)] = float(arr[idx])
        return replace(self, **updates)

    def values_for(self, names: Sequence[str]) -> jnp.ndarray:
        return jnp.asarray([float(getattr(self, name)) for name in names], dtype=jnp.float32)

    def to_calibration_state(self) -> CalibrationState:
        active = set(self.active_geometry_dofs)
        axis_unit = [float(v) for v in self.axis_unit_lab()]
        return CalibrationState(
            detector=(
                CalibrationVariable(
                    name="det_u_px",
                    value=float(self.det_u_px),
                    unit="native_detector_px",
                    status="estimated" if "det_u_px" in active else "frozen",
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="det_v_px",
                    value=float(self.det_v_px),
                    unit="native_detector_px",
                    status="estimated" if "det_v_px" in active else "frozen",
                    frame="detector",
                    gauge="detector_ray_grid_center",
                ),
                CalibrationVariable(
                    name="detector_roll_deg",
                    value=float(self.detector_roll_deg),
                    unit="deg",
                    status="estimated" if "detector_roll_deg" in active else "frozen",
                    frame="detector_plane",
                    gauge="detector_plane_roll",
                ),
            ),
            scan=(
                CalibrationVariable(
                    name="axis_rot_x_deg",
                    value=float(self.axis_rot_x_deg),
                    unit="deg",
                    status="estimated" if "axis_rot_x_deg" in active else "frozen",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_rot_y_deg",
                    value=float(self.axis_rot_y_deg),
                    unit="deg",
                    status="estimated" if "axis_rot_y_deg" in active else "frozen",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
                CalibrationVariable(
                    name="axis_unit_lab",
                    value=axis_unit,
                    unit="unit_vector",
                    status="derived",
                    frame="scan",
                    gauge="rotation_axis_direction",
                ),
            ),
        )

    def axis_unit_lab(self) -> tuple[float, float, float]:
        axis = axis_unit_from_rotations(
            self.nominal_axis_unit,
            axis_rot_x_deg=float(self.axis_rot_x_deg),
            axis_rot_y_deg=float(self.axis_rot_y_deg),
        )
        return tuple(float(v) for v in np.asarray(axis))


def geometry_inputs_from_geometry(geometry: Geometry) -> dict[str, object]:
    detector = getattr(geometry, "detector")
    grid = getattr(geometry, "grid")
    payload: dict[str, object] = {
        "grid": grid.to_dict(),
        "detector": detector.to_dict(),
        "thetas_deg": np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32),
        "geometry_type": "parallel",
    }
    if isinstance(geometry, LaminographyGeometry):
        payload["geometry_type"] = "lamino"
        payload["tilt_deg"] = float(geometry.tilt_deg)
        payload["tilt_about"] = str(geometry.tilt_about)
    elif isinstance(geometry, RotationAxisGeometry):
        payload["axis_unit_lab"] = list(getattr(geometry, "axis_unit_lab"))
    if getattr(geometry, "detector_roll_deg", None) is not None:
        payload["detector_roll_deg"] = float(getattr(geometry, "detector_roll_deg"))
    return payload


def geometry_with_axis_state(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    state: GeometryCalibrationState,
) -> Geometry:
    thetas = np.asarray(getattr(geometry, "thetas_deg"), dtype=np.float32)
    axis_active = (
        abs(float(state.axis_rot_x_deg)) > 1e-7
        or abs(float(state.axis_rot_y_deg)) > 1e-7
        or isinstance(geometry, RotationAxisGeometry)
    )
    if axis_active:
        return RotationAxisGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            axis_unit_lab=state.axis_unit_lab(),
        )
    if isinstance(geometry, LaminographyGeometry):
        return LaminographyGeometry(
            grid=grid,
            detector=detector,
            thetas_deg=thetas,
            tilt_deg=float(geometry.tilt_deg),
            tilt_about=str(geometry.tilt_about),
        )
    return ParallelGeometry(grid=grid, detector=detector, thetas_deg=thetas)


def level_detector_grid(
    detector: Detector,
    *,
    state: GeometryCalibrationState,
    factor: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    factor_f = float(max(1, int(factor)))
    return detector_grid_from_calibration(
        detector,
        det_u_px=float(detector.det_center[0]) / float(detector.du)
        + jnp.asarray(state.det_u_px, dtype=jnp.float32) / jnp.float32(factor_f),
        det_v_px=float(detector.det_center[1]) / float(detector.dv)
        + jnp.asarray(state.det_v_px, dtype=jnp.float32) / jnp.float32(factor_f),
        detector_roll_deg=state.detector_roll_deg,
    )


def _state_with_active_values(
    state: GeometryCalibrationState,
    active_names: Sequence[str],
    values: jnp.ndarray,
) -> GeometryCalibrationState:
    data = {
        "det_u_px": state.det_u_px,
        "det_v_px": state.det_v_px,
        "detector_roll_deg": state.detector_roll_deg,
        "axis_rot_x_deg": state.axis_rot_x_deg,
        "axis_rot_y_deg": state.axis_rot_y_deg,
    }
    for idx, name in enumerate(active_names):
        data[str(name)] = values[idx]
    return GeometryCalibrationState(
        det_u_px=data["det_u_px"],
        det_v_px=data["det_v_px"],
        detector_roll_deg=data["detector_roll_deg"],
        axis_rot_x_deg=data["axis_rot_x_deg"],
        axis_rot_y_deg=data["axis_rot_y_deg"],
        nominal_axis_unit=state.nominal_axis_unit,
        active_geometry_dofs=state.active_geometry_dofs,
    )


def _pose_stack_for_state(
    geometry: Geometry,
    state: GeometryCalibrationState,
) -> jnp.ndarray:
    thetas = jnp.asarray(getattr(geometry, "thetas_deg"), dtype=jnp.float32)
    axis = axis_unit_from_rotations(
        state.nominal_axis_unit,
        axis_rot_x_deg=state.axis_rot_x_deg,
        axis_rot_y_deg=state.axis_rot_y_deg,
    )
    return axis_pose_stack(thetas, axis)


def _block_names(active_geometry_dofs: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    active = set(active_geometry_dofs)
    return [
        (label, tuple(name for name in names if name in active))
        for label, names in GEOMETRY_BLOCKS
        if any(name in active for name in names)
    ]


def optimize_geometry_blocks_for_level(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    init_x: jnp.ndarray | None,
    state: GeometryCalibrationState,
    factor: int,
    recon_iters: int,
    lambda_tv: float,
    regulariser: str,
    huber_delta: float,
    tv_prox_iters: int,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    gn_damping: float,
    outer_iters: int,
    max_step_px: float = 2.0,
    max_step_deg: float = 2.0,
) -> tuple[jnp.ndarray, GeometryCalibrationState, list[dict[str, float | int | str | bool]]]:
    """Run staged global geometry GN blocks at one multires level.

    The update differentiates fixed-volume reprojection residuals only. It does not
    differentiate through reconstruction, keeping the memory profile aligned with
    the existing alternating alignment loop.
    """
    if not state.active_geometry_dofs:
        if init_x is None:
            geom = geometry_with_axis_state(geometry, grid, detector, state)
            det_grid = level_detector_grid(detector, state=state, factor=factor)
            x, _ = _reconstruct(
                geom,
                grid,
                detector,
                projections,
                init_x=None,
                det_grid=det_grid,
                recon_iters=recon_iters,
                lambda_tv=lambda_tv,
                regulariser=regulariser,
                huber_delta=huber_delta,
                tv_prox_iters=tv_prox_iters,
                views_per_batch=views_per_batch,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
            )
            return x, state, []
        return init_x, state, []

    x = init_x
    current = state
    stats: list[dict[str, float | int | str | bool]] = []
    for outer_idx in range(1, int(outer_iters) + 1):
        geom = geometry_with_axis_state(geometry, grid, detector, current)
        det_grid = level_detector_grid(detector, state=current, factor=factor)
        x, _ = _reconstruct(
            geom,
            grid,
            detector,
            projections,
            init_x=x,
            det_grid=det_grid,
            recon_iters=recon_iters,
            lambda_tv=lambda_tv,
            regulariser=regulariser,
            huber_delta=huber_delta,
            tv_prox_iters=tv_prox_iters,
            views_per_batch=views_per_batch,
            projector_unroll=projector_unroll,
            checkpoint_projector=checkpoint_projector,
            gather_dtype=gather_dtype,
        )
        for block_name, active_names in _block_names(current.active_geometry_dofs):
            current, block_stat = _optimize_one_block(
                geometry=geometry,
                grid=grid,
                detector=detector,
                projections=projections,
                volume=x,
                state=current,
                active_names=active_names,
                factor=factor,
                block_name=block_name,
                views_per_batch=views_per_batch,
                projector_unroll=projector_unroll,
                checkpoint_projector=checkpoint_projector,
                gather_dtype=gather_dtype,
                gn_damping=gn_damping,
                max_step_px=max_step_px,
                max_step_deg=max_step_deg,
            )
            block_stat["geometry_outer_idx"] = int(outer_idx)
            stats.append(block_stat)
    geom = geometry_with_axis_state(geometry, grid, detector, current)
    det_grid = level_detector_grid(detector, state=current, factor=factor)
    x, _ = _reconstruct(
        geom,
        grid,
        detector,
        projections,
        init_x=x,
        det_grid=det_grid,
        recon_iters=recon_iters,
        lambda_tv=lambda_tv,
        regulariser=regulariser,
        huber_delta=huber_delta,
        tv_prox_iters=tv_prox_iters,
        views_per_batch=views_per_batch,
        projector_unroll=projector_unroll,
        checkpoint_projector=checkpoint_projector,
        gather_dtype=gather_dtype,
    )
    return x, current, stats


def _reconstruct(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    *,
    init_x: jnp.ndarray | None,
    det_grid: tuple[jnp.ndarray, jnp.ndarray],
    recon_iters: int,
    lambda_tv: float,
    regulariser: str,
    huber_delta: float,
    tv_prox_iters: int,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
) -> tuple[jnp.ndarray, dict]:
    cfg = FistaConfig(
        iters=int(recon_iters),
        lambda_tv=float(lambda_tv),
        regulariser=regulariser,  # type: ignore[arg-type]
        huber_delta=float(huber_delta),
        views_per_batch=max(1, int(views_per_batch)),
        projector_unroll=int(projector_unroll),
        checkpoint_projector=bool(checkpoint_projector),
        gather_dtype=str(gather_dtype),
        tv_prox_iters=int(tv_prox_iters),
    )
    return fista_tv(
        geometry,
        grid,
        detector,
        projections,
        init_x=init_x,
        config=cfg,
        det_grid=det_grid,
    )


def _optimize_one_block(
    *,
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: jnp.ndarray,
    volume: jnp.ndarray,
    state: GeometryCalibrationState,
    active_names: Sequence[str],
    factor: int,
    block_name: str,
    views_per_batch: int,
    projector_unroll: int,
    checkpoint_projector: bool,
    gather_dtype: str,
    gn_damping: float,
    max_step_px: float,
    max_step_deg: float,
) -> tuple[GeometryCalibrationState, dict[str, float | int | str | bool]]:
    y = jnp.asarray(projections, dtype=jnp.float32)
    n_views, nv, nu = y.shape
    chunk_size = max(1, min(int(views_per_batch), int(n_views)))
    n_chunks = int(math.ceil(int(n_views) / chunk_size))
    pad_views = n_chunks * chunk_size - int(n_views)
    if pad_views:
        y_pad = jnp.pad(y, ((0, pad_views), (0, 0), (0, 0)), mode="edge")
        valid = jnp.concatenate(
            [
                jnp.ones((int(n_views),), dtype=jnp.float32),
                jnp.zeros((pad_views,), dtype=jnp.float32),
            ]
        )
    else:
        y_pad = y
        valid = jnp.ones((int(n_views),), dtype=jnp.float32)
    valid = valid.reshape((n_chunks, chunk_size, 1, 1))
    denom = jnp.sqrt(jnp.maximum(jnp.mean(y.astype(jnp.float32) ** 2), jnp.float32(1e-6)))
    initial_values = state.values_for(active_names)

    def residual_for_values(values: jnp.ndarray) -> jnp.ndarray:
        candidate = _state_with_active_values(state, active_names, values)
        T_all = _pose_stack_for_state(geometry, candidate)
        if pad_views:
            T_all = jnp.pad(T_all, ((0, pad_views), (0, 0), (0, 0)), mode="edge")
        T_chunks = T_all.reshape((n_chunks, chunk_size, 4, 4))
        y_chunks = y_pad.reshape((n_chunks, chunk_size, nv, nu))
        det_grid = level_detector_grid(detector, state=candidate, factor=factor)

        def body(_, idx):
            T_chunk = T_chunks[idx]
            y_chunk = y_chunks[idx]
            valid_chunk = valid[idx]
            pred = jax.vmap(
                lambda T: forward_project_view_T(
                    T,
                    grid,
                    detector,
                    volume,
                    use_checkpoint=checkpoint_projector,
                    unroll=int(projector_unroll),
                    gather_dtype=gather_dtype,
                    det_grid=det_grid,
                )
            )(T_chunk)
            return None, ((pred - y_chunk) * valid_chunk / denom).reshape(-1)

        _, chunks = jax.lax.scan(body, None, jnp.arange(n_chunks, dtype=jnp.int32))
        return chunks.reshape(-1)

    residual = residual_for_values(initial_values)
    loss_before = float(jnp.mean(jnp.square(residual)))
    _, pullback = jax.vjp(residual_for_values, initial_values)
    gradient = pullback(residual)[0] * jnp.float32(2.0 / max(int(residual.size), 1))
    eye = jnp.eye(int(initial_values.size), dtype=jnp.float32)

    def jvp_col(direction: jnp.ndarray) -> jnp.ndarray:
        return jax.jvp(residual_for_values, (initial_values,), (direction,))[1]

    jac_cols = jax.vmap(jvp_col)(eye)
    curvature = (jac_cols @ jac_cols.T) * jnp.float32(2.0 / max(int(residual.size), 1))
    system = curvature + jnp.eye(int(initial_values.size), dtype=jnp.float32) * jnp.float32(
        gn_damping
    )
    raw_step = jnp.linalg.solve(system, -gradient)
    max_step = _max_step_for_names(active_names, max_step_px=max_step_px, max_step_deg=max_step_deg)
    raw_norm = jnp.linalg.norm(raw_step)
    clipped_step = jnp.where(
        raw_norm > max_step,
        raw_step * (max_step / jnp.maximum(raw_norm, jnp.float32(1e-6))),
        raw_step,
    )

    best_values = initial_values
    best_loss = loss_before
    best_scale = 0.0
    best_step = jnp.zeros_like(initial_values)
    for scale in (1.0, 0.5, 0.25):
        trial_step = clipped_step * jnp.float32(scale)
        trial_values = initial_values + trial_step
        trial_residual = residual_for_values(trial_values)
        trial_loss = float(jnp.mean(jnp.square(trial_residual)))
        if math.isfinite(trial_loss) and trial_loss < best_loss:
            best_values = trial_values
            best_loss = trial_loss
            best_scale = float(scale)
            best_step = trial_step
            break

    next_state = state.replace_values(active_names, best_values)
    return next_state, {
        "geometry_block": block_name,
        "geometry_active_dofs": ",".join(active_names),
        "geometry_loss_before": loss_before,
        "geometry_loss_after": float(best_loss),
        "geometry_accepted": bool(best_scale > 0.0),
        "geometry_step_norm": float(jnp.linalg.norm(best_step)),
        "geometry_gradient_norm": float(jnp.linalg.norm(gradient)),
    }


def _max_step_for_names(
    names: Sequence[str],
    *,
    max_step_px: float,
    max_step_deg: float,
) -> jnp.ndarray:
    steps = [
        float(max_step_px) if name in {"det_u_px", "det_v_px"} else float(max_step_deg)
        for name in names
    ]
    return jnp.asarray(max(steps) if steps else 0.0, dtype=jnp.float32)
