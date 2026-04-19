from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp


DOF_NAMES = ("alpha", "beta", "phi", "dx", "dz")
DOF_INDEX = {name: idx for idx, name in enumerate(DOF_NAMES)}


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
