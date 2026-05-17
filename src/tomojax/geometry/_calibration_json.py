"""JSON normalization helpers for calibration metadata."""

# ruff: noqa: PLR0911

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
