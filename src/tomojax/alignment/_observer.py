from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import jax.numpy as jnp

ObserverAction = Literal["continue", "advance_level", "stop_run"]
type OuterStatValue = float | int | bool | str | list[object] | dict[str, object] | None
type OuterStat = dict[str, OuterStatValue]

ObserverCallback = Callable[[jnp.ndarray, jnp.ndarray, OuterStat], ObserverAction | None]
BoolCompatibleObserverCallback = Callable[
    [jnp.ndarray, jnp.ndarray, OuterStat],
    ObserverAction | bool | None,
]


def _normalize_observer_action(
    action: ObserverAction | str | None,
) -> ObserverAction:
    if action is None:
        return "continue"
    if isinstance(action, str):
        lowered = action.strip().lower()
        if lowered in {"continue", "advance_level", "stop_run"}:
            return cast("ObserverAction", lowered)
    raise ValueError(f"Unsupported observer action: {action!r}")


def _scalar_bool_action(action: object) -> bool | None:
    if isinstance(action, bool):
        return action

    dtype = getattr(action, "dtype", None)
    shape = getattr(action, "shape", None)
    if dtype is None or str(dtype) != "bool" or shape != ():
        return None
    return bool(action)


def adapt_observer_callback(
    observer: BoolCompatibleObserverCallback | None,
) -> ObserverCallback | None:
    """Wrap bool-compatible observers in the explicit ObserverAction contract."""
    if observer is None:
        return None

    def _wrapped(x: jnp.ndarray, params5: jnp.ndarray, stat: OuterStat) -> ObserverAction | None:
        action = observer(x, params5, stat)
        if action is None:
            return None
        bool_action = _scalar_bool_action(action)
        if bool_action is not None:
            return "stop_run" if bool_action else None
        return _normalize_observer_action(cast("ObserverAction | str | None", action))

    return _wrapped


__all__ = [
    "BoolCompatibleObserverCallback",
    "ObserverAction",
    "ObserverCallback",
    "OuterStat",
    "OuterStatValue",
    "_normalize_observer_action",
    "adapt_observer_callback",
]
