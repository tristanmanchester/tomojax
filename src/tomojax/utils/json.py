from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def normalize_json(
    value: Any,
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
        items = value.items()
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
            for v in value
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


def drop_none(payload: Mapping[str, Any], **normalize_options: bool) -> dict[str, JsonValue]:
    """Return a JSON-compatible dict without keys whose value is ``None``."""
    return {
        str(k): normalize_json(v, **normalize_options)
        for k, v in payload.items()
        if v is not None
    }


def _is_argparse_namespace(value: Any) -> bool:
    try:
        import argparse

        return isinstance(value, argparse.Namespace)
    except Exception:
        return False


class _Unhandled:
    pass


_UNHANDLED = _Unhandled()


def _normalize_array(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        import jax

        if isinstance(value, jax.Array):
            return value.tolist()
    except Exception:
        pass

    return _UNHANDLED
