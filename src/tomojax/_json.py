"""JSON normalisation shared by metadata, manifests and calibration records."""

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
    """Convert common runtime objects into strict JSON-compatible values.

    Parameters
    ----------
    value : object
        Runtime value, including mappings, dataclasses, paths and NumPy/JAX arrays.
    namespace : bool, optional
        Convert argparse namespaces to mappings; defaults to False.
    sort_mapping_keys : bool, optional
        Sort mapping keys by their string representation.
    catch_to_dict_errors : bool, optional
        Fall back to text if an object's to_dict method fails.

    Returns
    -------
    JSON-compatible value
        Normalized value; nonfinite floats and otherwise unsupported objects become strings.

    Raises
    ------
    TypeError
        If a NumPy or JAX value cannot be materialized as JSON data.

    Examples
    --------
    >>> import numpy as np
    >>> from tomojax.io.api import normalize_json
    >>> normalize_json({"angles": np.array([0.0, 90.0]), "missing": float("nan")})
    {'angles': [0.0, 90.0], 'missing': 'nan'}
    """
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
    """Return a JSON-compatible dict without keys whose value is ``None``.

    Parameters
    ----------
    payload : mapping
        Mapping whose top-level None values should be omitted.
    **normalize_options : bool
        Keyword settings forwarded to normalize_json.

    Returns
    -------
    dict
        Normalized mapping with absent top-level values omitted.
    """
    return {
        str(k): normalize_json(v, **normalize_options) for k, v in payload.items() if v is not None
    }


def read_json_object(path: Path) -> dict[str, JsonValue]:
    """Read a JSON file that must contain an object.

    Parameters
    ----------
    path : pathlib.Path
        JSON file containing an object.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the content is invalid JSON or not an object.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        data = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast("dict[str, JsonValue]", data)


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
    except ImportError:
        np = None  # type: ignore[assignment]

    if np is not None:
        if isinstance(value, np.generic):
            try:
                return cast("object", value.item())
            except Exception as exc:
                scalar = cast("object", value)
                raise TypeError(
                    f"could not normalize NumPy scalar {type(scalar).__name__} to JSON"
                ) from exc
        if isinstance(value, np.ndarray):
            try:
                return cast("object", value.tolist())
            except Exception as exc:
                shape = cast("object", value.shape)
                raise TypeError(
                    f"could not normalize NumPy array with shape {shape} to JSON"
                ) from exc

    try:
        import jax
    except ImportError:
        jax = None  # type: ignore[assignment]

    if jax is not None and isinstance(value, jax.Array):
        try:
            return cast("object", value.tolist())
        except Exception as exc:
            raise TypeError(
                f"could not normalize JAX array with shape {getattr(value, 'shape', None)} to JSON"
            ) from exc

    return _UNHANDLED
