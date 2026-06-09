"""JSON normalization helpers for calibration metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def normalize_json(value: Any) -> JsonValue:
    """Convert common runtime objects into strict JSON-compatible values."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    try:
        import jax
    except ImportError:  # pragma: no cover - JAX is a project dependency
        jax = None  # type: ignore[assignment]
    if jax is not None and isinstance(value, jax.Array):
        try:
            return normalize_json(value.tolist())
        except Exception as exc:
            raise TypeError(
                f"could not normalize JAX array with shape {getattr(value, 'shape', None)} to JSON"
            ) from exc
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a project dependency
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return normalize_json(value.item())
        if isinstance(value, np.ndarray):
            return normalize_json(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [normalize_json(v) for v in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return normalize_json(to_dict())
    return str(value)


def drop_none(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Return a JSON-compatible dict without keys whose value is ``None``."""
    return {str(k): normalize_json(v) for k, v in payload.items() if v is not None}
