from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

import jax.numpy as jnp

from .dofs import (
    ALL_ALIGNMENT_DOF_NAMES,
    DOF_INDEX,
    DofBounds,
    normalize_alignment_dofs,
)
from .state import AlignmentState


DofScope = Literal["setup", "pose"]


@dataclass(frozen=True, slots=True)
class ParameterScale:
    """Affine whitening rule for one physical active value."""

    reference: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("ParameterScale.scale must be finite and positive")

    def to_whitened(self, value: jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(value, dtype=jnp.float32) - jnp.float32(self.reference)) / jnp.float32(
            self.scale
        )

    def from_whitened(self, value: jnp.ndarray) -> jnp.ndarray:
        return jnp.float32(self.reference) + jnp.asarray(value, dtype=jnp.float32) * jnp.float32(
            self.scale
        )


@dataclass(frozen=True, slots=True)
class DofSpec:
    """Registry entry for a public alignment DOF."""

    name: str
    scope: DofScope
    unit: str
    scale: ParameterScale
    state_attr: str | None = None
    pose_index: int | None = None
    lower: float = -math.inf
    upper: float = math.inf
    display_unit: str | None = None
    gauge_group: str | None = None
    description: str = ""

    @property
    def is_pose(self) -> bool:
        return self.scope == "pose"

    @property
    def is_setup(self) -> bool:
        return self.scope == "setup"


def _deg(value: float) -> float:
    return math.radians(float(value))


def _pose_spec(name: str, index: int, *, unit: str, scale: float, gauge: str) -> DofSpec:
    return DofSpec(
        name=name,
        scope="pose",
        unit=unit,
        scale=ParameterScale(scale=scale),
        pose_index=int(index),
        gauge_group=gauge,
    )


DOF_SPECS: dict[str, DofSpec] = {
    "alpha": _pose_spec("alpha", DOF_INDEX["alpha"], unit="rad", scale=1e-3, gauge="pose_rot"),
    "beta": _pose_spec("beta", DOF_INDEX["beta"], unit="rad", scale=1e-3, gauge="pose_rot"),
    "phi": _pose_spec("phi", DOF_INDEX["phi"], unit="rad", scale=1e-3, gauge="pose_rot"),
    "dx": _pose_spec("dx", DOF_INDEX["dx"], unit="world", scale=1e-1, gauge="pose_translation"),
    "dz": _pose_spec("dz", DOF_INDEX["dz"], unit="world", scale=1e-1, gauge="pose_translation"),
    "det_u_px": DofSpec(
        name="det_u_px",
        scope="setup",
        state_attr="det_u_px",
        unit="native_detector_px",
        scale=ParameterScale(scale=1.0),
        gauge_group="detector_ray_grid_center",
    ),
    "det_v_px": DofSpec(
        name="det_v_px",
        scope="setup",
        state_attr="det_v_px",
        unit="native_detector_px",
        scale=ParameterScale(scale=1.0),
        gauge_group="detector_ray_grid_center",
    ),
    "detector_roll_deg": DofSpec(
        name="detector_roll_deg",
        scope="setup",
        state_attr="detector_roll_rad",
        unit="rad",
        display_unit="deg",
        scale=ParameterScale(scale=_deg(1.0)),
        lower=-math.pi,
        upper=math.pi,
        gauge_group="detector_plane_roll",
    ),
    "axis_rot_x_deg": DofSpec(
        name="axis_rot_x_deg",
        scope="setup",
        state_attr="axis_rot_x_rad",
        unit="rad",
        display_unit="deg",
        scale=ParameterScale(scale=_deg(1.0)),
        lower=_deg(-60.0),
        upper=_deg(60.0),
        gauge_group="rotation_axis_direction",
    ),
    "axis_rot_y_deg": DofSpec(
        name="axis_rot_y_deg",
        scope="setup",
        state_attr="axis_rot_y_rad",
        unit="rad",
        display_unit="deg",
        scale=ParameterScale(scale=_deg(1.0)),
        lower=_deg(-60.0),
        upper=_deg(60.0),
        gauge_group="rotation_axis_direction",
    ),
    "tilt_deg": DofSpec(
        name="tilt_deg",
        scope="setup",
        state_attr="tilt_rad",
        unit="rad",
        display_unit="deg",
        scale=ParameterScale(scale=_deg(1.0)),
        lower=_deg(-90.0),
        upper=_deg(90.0),
        gauge_group="rotation_axis_direction",
    ),
}


def dof_spec(name: str) -> DofSpec:
    try:
        return DOF_SPECS[str(name)]
    except KeyError as exc:
        valid = ", ".join(DOF_SPECS)
        raise ValueError(f"Unknown alignment DOF {name!r}; valid DOFs: {valid}") from exc


def ordered_dofs(names: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(names)
    requested_set = set(requested)
    return tuple(name for name in ALL_ALIGNMENT_DOF_NAMES if name in requested_set)


@dataclass(frozen=True, slots=True)
class ActiveParameterView:
    """Pack/unpack active DOFs as a whitened vector."""

    dofs: tuple[str, ...]

    @classmethod
    def from_dofs(
        cls,
        values: str | Iterable[str] | None,
        *,
        geometry: object | None = None,
    ) -> "ActiveParameterView":
        names = normalize_alignment_dofs(values, option_name="active_dofs", geometry=geometry)
        return cls(ordered_dofs(names))

    def __post_init__(self) -> None:
        normalized = ordered_dofs(
            normalize_alignment_dofs(self.dofs, option_name="active_dofs")
        )
        if len(normalized) != len(tuple(self.dofs)):
            raise ValueError("Duplicate active DOFs are not allowed")
        object.__setattr__(self, "dofs", normalized)
        if not normalized:
            raise ValueError("ActiveParameterView requires at least one active DOF")
        for name in normalized:
            dof_spec(name)

    @property
    def specs(self) -> tuple[DofSpec, ...]:
        return tuple(dof_spec(name) for name in self.dofs)

    @property
    def active_pose_dofs(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs if spec.is_pose)

    @property
    def active_setup_dofs(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs if spec.is_setup)

    def pack(self, state: AlignmentState) -> jnp.ndarray:
        parts: list[jnp.ndarray] = []
        for spec in self.specs:
            values = _values_for_spec(state, spec)
            parts.append(spec.scale.to_whitened(values).reshape(-1))
        return jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0,), dtype=jnp.float32)

    def unpack(self, state: AlignmentState, whitened: jnp.ndarray) -> AlignmentState:
        values = jnp.asarray(whitened, dtype=jnp.float32).reshape(-1)
        cursor = 0
        setup_updates: dict[str, jnp.ndarray] = {}
        params5 = state.pose.params5
        for spec in self.specs:
            size = _value_size_for_spec(state, spec)
            chunk = values[cursor : cursor + size]
            cursor += size
            physical = spec.scale.from_whitened(chunk)
            if spec.is_setup:
                if spec.state_attr is None:
                    raise ValueError(f"Setup DOF {spec.name!r} is missing state_attr")
                setup_updates[spec.state_attr] = physical.reshape(())
            else:
                if spec.pose_index is None:
                    raise ValueError(f"Pose DOF {spec.name!r} is missing pose_index")
                params5 = params5.at[:, int(spec.pose_index)].set(
                    physical.reshape((int(params5.shape[0]),))
                )
        if cursor != int(values.size):
            raise ValueError(
                f"Active vector has {int(values.size)} values but {cursor} were consumed"
            )
        setup = state.setup.replace(**setup_updates) if setup_updates else state.setup
        pose = state.pose.replace(params5=params5)
        return state.replace(setup=setup, pose=pose)

    def bounds_whitened(
        self,
        state: AlignmentState,
        bounds: DofBounds | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        overrides = {} if bounds is None else {name: (lo, hi) for name, lo, hi in bounds}
        lower_parts: list[jnp.ndarray] = []
        upper_parts: list[jnp.ndarray] = []
        for spec in self.specs:
            size = _value_size_for_spec(state, spec)
            lower_value, upper_value = overrides.get(spec.name, (spec.lower, spec.upper))
            lower = jnp.full((size,), lower_value, dtype=jnp.float32)
            upper = jnp.full((size,), upper_value, dtype=jnp.float32)
            lower_parts.append(spec.scale.to_whitened(lower))
            upper_parts.append(spec.scale.to_whitened(upper))
        return jnp.concatenate(lower_parts), jnp.concatenate(upper_parts)

    def native_delta_by_dof(
        self,
        before: AlignmentState,
        after: AlignmentState,
    ) -> dict[str, list[float] | float]:
        deltas: dict[str, list[float] | float] = {}
        for spec in self.specs:
            delta = _values_for_spec(after, spec) - _values_for_spec(before, spec)
            arr = jnp.asarray(delta, dtype=jnp.float32).reshape(-1)
            if int(arr.size) == 1:
                deltas[spec.name] = float(arr[0])
            else:
                deltas[spec.name] = [float(v) for v in arr]
        return deltas


def _values_for_spec(state: AlignmentState, spec: DofSpec) -> jnp.ndarray:
    if spec.is_setup:
        if spec.state_attr is None:
            raise ValueError(f"Setup DOF {spec.name!r} is missing state_attr")
        return jnp.asarray(getattr(state.setup, spec.state_attr), dtype=jnp.float32)
    if spec.pose_index is None:
        raise ValueError(f"Pose DOF {spec.name!r} is missing pose_index")
    return jnp.asarray(state.pose.params5[:, int(spec.pose_index)], dtype=jnp.float32)


def _value_size_for_spec(state: AlignmentState, spec: DofSpec) -> int:
    if spec.is_setup:
        return 1
    return int(state.pose.params5.shape[0])


def optimizer_step_stats(
    *,
    view: ActiveParameterView,
    before: AlignmentState,
    after: AlignmentState,
    grad_whitened: jnp.ndarray | None = None,
) -> dict[str, object]:
    """Common per-stage movement diagnostics in whitened and native units."""
    z_before = view.pack(before)
    z_after = view.pack(after)
    step = z_after - z_before
    stats: dict[str, object] = {
        "step_norm_whitened": float(jnp.linalg.norm(step)),
        "step_by_dof_native_units": view.native_delta_by_dof(before, after),
        "active_dofs": list(view.dofs),
    }
    if grad_whitened is not None:
        grad = jnp.asarray(grad_whitened, dtype=jnp.float32).reshape(-1)
        stats["grad_norm_whitened"] = float(jnp.linalg.norm(grad))
    return stats


def active_view_from_scopes(
    *,
    pose_dofs: Sequence[str] = (),
    setup_dofs: Sequence[str] = (),
) -> ActiveParameterView:
    return ActiveParameterView(ordered_dofs(tuple(pose_dofs) + tuple(setup_dofs)))
