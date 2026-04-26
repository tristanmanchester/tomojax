from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

import jax.numpy as jnp


DOF_NAMES = ("alpha", "beta", "phi", "dx", "dz")
DOF_INDEX = {name: idx for idx, name in enumerate(DOF_NAMES)}
GEOMETRY_DOF_NAMES = (
    "det_u_px",
    "det_v_px",
    "detector_roll_deg",
    "axis_rot_x_deg",
    "axis_rot_y_deg",
    "tilt_deg",
)
GEOMETRY_DOF_INDEX = {name: idx for idx, name in enumerate(GEOMETRY_DOF_NAMES)}
ALL_ALIGNMENT_DOF_NAMES = DOF_NAMES + GEOMETRY_DOF_NAMES
ALL_ALIGNMENT_DOF_INDEX = {name: idx for idx, name in enumerate(ALL_ALIGNMENT_DOF_NAMES)}
type DofBounds = tuple[tuple[str, float, float], ...]


@dataclass(frozen=True, slots=True)
class ScopedAlignmentDofs:
    """Resolved public alignment DOFs split into pose and geometry scopes."""

    active_pose_dofs: tuple[str, ...]
    active_geometry_dofs: tuple[str, ...]
    frozen_pose_dofs: tuple[str, ...]
    frozen_geometry_dofs: tuple[str, ...]

    @property
    def pose_mask(self) -> tuple[bool, bool, bool, bool, bool]:
        active = set(self.active_pose_dofs)
        return tuple(name in active for name in DOF_NAMES)  # type: ignore[return-value]

    @property
    def active_dofs(self) -> tuple[str, ...]:
        return self.active_pose_dofs + self.active_geometry_dofs


def _iter_dof_tokens(value: str | Iterable[str] | None) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, str):
        return value.split(",")
    tokens: list[str] = []
    for item in value:
        tokens.extend(str(item).split(","))
    return tokens


def normalize_dofs(
    value: str | Iterable[str] | None,
    *,
    option_name: str = "dofs",
) -> tuple[str, ...]:
    """Normalize named alignment DOFs from CLI/config/Python inputs."""
    names: list[str] = []
    seen: set[str] = set()
    valid = ", ".join(DOF_NAMES)
    for raw in _iter_dof_tokens(value):
        name = str(raw).strip().lower()
        if not name:
            continue
        if name not in DOF_INDEX:
            raise ValueError(
                f"Unknown alignment DOF for {option_name}: {name!r}; valid DOFs: {valid}"
            )
        if name in seen:
            raise ValueError(f"Duplicate alignment DOF for {option_name}: {name!r}")
        seen.add(name)
        names.append(name)
    return tuple(names)


def _tilt_alias_for_geometry(geometry: object | None) -> str:
    if geometry is None:
        return "tilt_deg"
    tilt_about = getattr(geometry, "tilt_about", "x")
    return "axis_rot_y_deg" if str(tilt_about) == "z" else "axis_rot_x_deg"


def normalize_alignment_dofs(
    value: str | Iterable[str] | None,
    *,
    option_name: str = "dofs",
    geometry: object | None = None,
) -> tuple[str, ...]:
    """Normalize public alignment DOFs across pose and geometry scopes."""
    names: list[str] = []
    seen: set[str] = set()
    valid = ", ".join(ALL_ALIGNMENT_DOF_NAMES)
    for raw in _iter_dof_tokens(value):
        name = str(raw).strip().lower()
        if not name:
            continue
        if name == "tilt_deg":
            name = _tilt_alias_for_geometry(geometry)
        if name not in ALL_ALIGNMENT_DOF_INDEX and name != "tilt_deg":
            raise ValueError(
                f"Unknown alignment DOF for {option_name}: {name!r}; valid DOFs: {valid}"
            )
        if name in seen:
            raise ValueError(f"Duplicate alignment DOF for {option_name}: {name!r}")
        seen.add(name)
        names.append(name)
    return tuple(names)


def resolve_scoped_alignment_dofs(
    *,
    optimise_dofs: str | Iterable[str] | None = None,
    freeze_dofs: str | Iterable[str] | None = None,
    geometry_dofs: str | Iterable[str] | None = None,
    geometry: object | None = None,
) -> ScopedAlignmentDofs:
    """Resolve effective active/frozen alignment DOFs into pose and geometry scopes."""
    optimise = (
        None
        if optimise_dofs is None
        else normalize_alignment_dofs(
            optimise_dofs,
            option_name="optimise_dofs",
            geometry=geometry,
        )
    )
    legacy_geometry = normalize_alignment_dofs(
        geometry_dofs,
        option_name="geometry_dofs",
        geometry=geometry,
    )
    for name in legacy_geometry:
        if name in DOF_INDEX:
            raise ValueError(
                f"Pose DOF {name!r} is not valid for geometry_dofs; "
                f"valid geometry DOFs: {', '.join(GEOMETRY_DOF_NAMES)}"
            )
    freeze = normalize_alignment_dofs(
        freeze_dofs,
        option_name="freeze_dofs",
        geometry=geometry,
    )
    frozen = set(freeze)

    if optimise is None:
        base = DOF_NAMES + legacy_geometry
    else:
        base = optimise + tuple(name for name in legacy_geometry if name not in optimise)

    active: list[str] = []
    seen: set[str] = set()
    for name in base:
        if name in frozen or name in seen:
            continue
        seen.add(name)
        active.append(name)

    if not active:
        raise ValueError(
            "No active alignment DOFs remain after applying optimise_dofs/freeze_dofs; "
            f"valid DOFs: {', '.join(ALL_ALIGNMENT_DOF_NAMES)}"
        )

    active_pose = tuple(name for name in DOF_NAMES if name in active)
    active_geometry = tuple(name for name in GEOMETRY_DOF_NAMES if name in active)
    frozen_pose = tuple(name for name in DOF_NAMES if name in frozen)
    frozen_geometry = tuple(name for name in GEOMETRY_DOF_NAMES if name in frozen)
    return ScopedAlignmentDofs(
        active_pose_dofs=active_pose,
        active_geometry_dofs=active_geometry,
        frozen_pose_dofs=frozen_pose,
        frozen_geometry_dofs=frozen_geometry,
    )


def _parse_bound_float(raw: object, *, option_name: str, dof_name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid alignment bound for {option_name} {dof_name!r}: "
            f"{raw!r} is not numeric"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Invalid alignment bound for {option_name} {dof_name!r}: "
            f"{raw!r} must be finite"
        )
    return value


def _validate_bound_name(raw_name: object, *, option_name: str) -> str:
    name = str(raw_name).strip().lower()
    valid = ", ".join(DOF_NAMES)
    if not name:
        raise ValueError(f"Missing alignment DOF name for {option_name}; valid DOFs: {valid}")
    if name not in DOF_INDEX:
        raise ValueError(
            f"Unknown alignment DOF for {option_name}: {name!r}; valid DOFs: {valid}"
        )
    return name


def _parse_bound_pair(raw_pair: object, *, option_name: str, dof_name: str) -> tuple[float, float]:
    if isinstance(raw_pair, str):
        if ":" not in raw_pair:
            raise ValueError(
                f"Invalid alignment bounds for {option_name} {dof_name!r}: "
                "expected lower:upper"
            )
        lower_raw, upper_raw = raw_pair.split(":", 1)
        pair: Sequence[object] = (lower_raw.strip(), upper_raw.strip())
    elif isinstance(raw_pair, Sequence):
        pair = raw_pair
    else:
        raise ValueError(
            f"Invalid alignment bounds for {option_name} {dof_name!r}: "
            "expected a two-item sequence or lower:upper string"
        )

    if len(pair) != 2:
        raise ValueError(
            f"Invalid alignment bounds for {option_name} {dof_name!r}: "
            f"expected 2 values, got {len(pair)}"
        )
    lower = _parse_bound_float(pair[0], option_name=option_name, dof_name=dof_name)
    upper = _parse_bound_float(pair[1], option_name=option_name, dof_name=dof_name)
    if lower >= upper:
        raise ValueError(
            f"Invalid alignment bounds for {option_name} {dof_name!r}: "
            f"lower bound {lower:g} must be less than upper bound {upper:g}"
        )
    return lower, upper


def _iter_bound_items(value: object, *, option_name: str) -> Iterable[tuple[object, object]]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: list[tuple[object, object]] = []
        for raw_token in value.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(
                    f"Invalid alignment bounds for {option_name}: "
                    f"{token!r} must use DOF=lower:upper"
                )
            raw_name, raw_pair = token.split("=", 1)
            items.append((raw_name, raw_pair))
        return items
    if isinstance(value, Mapping):
        return value.items()
    if isinstance(value, Iterable):
        items = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 3:
                raise ValueError(
                    f"Invalid alignment bounds for {option_name}: "
                    "expected entries as (DOF, lower, upper)"
                )
            items.append((item[0], (item[1], item[2])))
        return items
    raise ValueError(
        f"Invalid alignment bounds for {option_name}: "
        "expected a string, mapping, sequence, or None"
    )


def normalize_bounds(value: object, *, option_name: str = "bounds") -> DofBounds:
    """Normalize finite per-DOF alignment bounds from CLI/config/Python inputs."""
    parsed: dict[str, tuple[float, float]] = {}
    for raw_name, raw_pair in _iter_bound_items(value, option_name=option_name):
        name = _validate_bound_name(raw_name, option_name=option_name)
        if name in parsed:
            raise ValueError(f"Duplicate alignment bounds for {option_name}: {name!r}")
        parsed[name] = _parse_bound_pair(raw_pair, option_name=option_name, dof_name=name)
    return tuple((name, *parsed[name]) for name in DOF_NAMES if name in parsed)


def bounds_vectors(bounds: DofBounds) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build 5-column lower/upper bound vectors for normalized per-DOF bounds."""
    lower = jnp.full((len(DOF_NAMES),), -jnp.inf, dtype=jnp.float32)
    upper = jnp.full((len(DOF_NAMES),), jnp.inf, dtype=jnp.float32)
    for name, lo, hi in bounds:
        idx = DOF_INDEX[name]
        lower = lower.at[idx].set(jnp.float32(lo))
        upper = upper.at[idx].set(jnp.float32(hi))
    return lower, upper


def active_dofs(
    *,
    optimise_dofs: str | Iterable[str] | None = None,
    freeze_dofs: str | Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return effective active DOFs after applying optimise and freeze selections."""
    optimise = (
        None
        if optimise_dofs is None
        else normalize_dofs(optimise_dofs, option_name="optimise_dofs")
    )
    freeze = normalize_dofs(freeze_dofs, option_name="freeze_dofs")
    frozen = set(freeze)
    base = DOF_NAMES if optimise is None else optimise
    active = tuple(name for name in base if name not in frozen)
    if not active:
        raise ValueError(
            "No active alignment DOFs remain after applying optimise_dofs/freeze_dofs; "
            f"valid DOFs: {', '.join(DOF_NAMES)}"
        )
    return active


def active_dof_mask(
    *,
    optimise_dofs: str | Iterable[str] | None = None,
    freeze_dofs: str | Iterable[str] | None = None,
) -> tuple[bool, bool, bool, bool, bool]:
    """Build a 5-column boolean mask for active alignment DOFs."""
    active = set(active_dofs(optimise_dofs=optimise_dofs, freeze_dofs=freeze_dofs))
    return tuple(name in active for name in DOF_NAMES)  # type: ignore[return-value]


def active_dof_mask_array(
    *,
    optimise_dofs: str | Iterable[str] | None = None,
    freeze_dofs: str | Iterable[str] | None = None,
) -> jnp.ndarray:
    """Build a float32 JAX mask for active alignment DOFs."""
    return jnp.asarray(
        active_dof_mask(optimise_dofs=optimise_dofs, freeze_dofs=freeze_dofs),
        dtype=jnp.float32,
    )
