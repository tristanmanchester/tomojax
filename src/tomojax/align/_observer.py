from __future__ import annotations

from typing import Callable, Literal

import jax.numpy as jnp

ObserverAction = Literal["continue", "advance_level", "stop_run"]
type OuterStatValue = float | int | bool | str | None
type OuterStat = dict[str, OuterStatValue]

ObserverCallback = Callable[[jnp.ndarray, jnp.ndarray, OuterStat], ObserverAction | None]
LegacyObserverCallback = Callable[
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
            return lowered  # type: ignore[return-value]
    raise ValueError(f"Unsupported observer action: {action!r}")


def adapt_legacy_observer(observer: LegacyObserverCallback | None) -> ObserverCallback | None:
    """Wrap a legacy bool observer in the explicit ObserverAction contract."""
    if observer is None:
        return None

    def _wrapped(x: jnp.ndarray, params5: jnp.ndarray, stat: OuterStat) -> ObserverAction | None:
        action = observer(x, params5, stat)
        if action is None or action is False:
            return None
        if action is True:
            return "stop_run"
        return _normalize_observer_action(action)

    return _wrapped


__all__ = [
    "LegacyObserverCallback",
    "ObserverAction",
    "ObserverCallback",
    "OuterStat",
    "OuterStatValue",
    "_normalize_observer_action",
    "adapt_legacy_observer",
]
