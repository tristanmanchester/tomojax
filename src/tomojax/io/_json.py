from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def normalize_json(
    value: object,
    *,
    namespace: bool = False,
    sort_mapping_keys: bool = False,
    catch_to_dict_errors: bool = False,
) -> JsonValue:
    """Convert common runtime objects into strict JSON-compatible values."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if namespace and _is_argparse_namespace(value):
        return normalize_json(
            vars(value),
            namespace=namespace,
            sort_mapping_keys=sort_mapping_keys,
            catch_to_dict_errors=catch_to_dict_errors,
        )
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_json(
            asdict(value),
            namespace=namespace,
            sort_mapping_keys=sort_mapping_keys,
            catch_to_dict_errors=catch_to_dict_errors,
        )
    if isinstance(value, Mapping):
        items: Iterable[tuple[object, object]] = cast("Mapping[object, object]", value).items()
        if sort_mapping_keys:
            items = sorted(items, key=lambda item: str(item[0]))
        return {
            str(k): normalize_json(
                v,
                namespace=namespace,
                sort_mapping_keys=sort_mapping_keys,
                catch_to_dict_errors=catch_to_dict_errors,
            )
            for k, v in items
        }
    if isinstance(value, tuple | list | set | frozenset):
        return [
            normalize_json(
                v,
                namespace=namespace,
                sort_mapping_keys=sort_mapping_keys,
                catch_to_dict_errors=catch_to_dict_errors,
            )
            for v in cast("Iterable[object]", value)
        ]

    array_value = _normalize_array(value)
    if array_value is not _UNHANDLED:
        return normalize_json(
            array_value,
            namespace=namespace,
            sort_mapping_keys=sort_mapping_keys,
            catch_to_dict_errors=catch_to_dict_errors,
        )

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return normalize_json(
                to_dict(),
                namespace=namespace,
                sort_mapping_keys=sort_mapping_keys,
                catch_to_dict_errors=catch_to_dict_errors,
            )
        except Exception:
            if not catch_to_dict_errors:
                raise

    return str(value)


def drop_none(payload: Mapping[str, object], **normalize_options: bool) -> dict[str, JsonValue]:
    """Return a JSON-compatible dict without keys whose value is ``None``."""
    return {
        str(k): normalize_json(v, **normalize_options) for k, v in payload.items() if v is not None
    }


def read_json_object(path: Path) -> dict[str, JsonValue]:
    """Read a JSON file that must contain an object."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast("dict[str, JsonValue]", data)


def write_json_object(path: Path, payload: object) -> None:
    """Write a normalized JSON object with deterministic formatting."""
    normalized = normalize_json(payload, sort_mapping_keys=True)
    if not isinstance(normalized, dict):
        raise ValueError("JSON object payload must normalize to a mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _is_argparse_namespace(value: object) -> bool:
    try:
        import argparse

        return isinstance(value, argparse.Namespace)
    except Exception:
        return False


class _Unhandled:
    pass


_UNHANDLED = _Unhandled()


def _normalize_array(value: object) -> object:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return cast("object", value.item())
        if isinstance(value, np.ndarray):
            return cast("object", value.tolist())
    except Exception:
        pass

    try:
        import jax

        if isinstance(value, jax.Array):
            return cast("object", value.tolist())
    except Exception:
        pass

    return _UNHANDLED
