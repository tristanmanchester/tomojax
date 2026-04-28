from __future__ import annotations

from typing import Any, Mapping

from tomojax.utils.json import JsonValue, drop_none as _drop_none, normalize_json as _normalize_json


def normalize_json(value: Any) -> JsonValue:
    """Convert common runtime objects into strict JSON-compatible values."""
    return _normalize_json(value)


def drop_none(payload: Mapping[str, Any]) -> dict[str, JsonValue]:
    """Return a JSON-compatible dict without keys whose value is ``None``."""
    return _drop_none(payload)
