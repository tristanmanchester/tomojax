from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any


SCHEMA_VERSION = 1


def _normalize_json(value: Any) -> Any:
    """Convert common CLI/runtime objects into strict JSON-compatible values."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return _normalize_json(vars(value))
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_json(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_normalize_json(v) for v in value]

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _normalize_json(value.item())
        if isinstance(value, np.ndarray):
            return _normalize_json(value.tolist())
    except Exception:
        pass

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _normalize_json(to_dict())
        except Exception:
            pass

    return str(value)


def _format_timestamp(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if isinstance(timestamp, str):
        return timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _tomojax_version() -> str | None:
    try:
        from importlib import metadata

        return metadata.version("tomojax")
    except Exception:
        try:
            import tomojax

            return getattr(tomojax, "__version__", None)
        except Exception:
            return None


def _versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "tomojax": _tomojax_version(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "jax": None,
        "jaxlib": None,
    }
    try:
        import jax

        versions["jax"] = getattr(jax, "__version__", None)
    except Exception:
        pass
    try:
        import jaxlib

        versions["jaxlib"] = getattr(jaxlib, "__version__", None)
    except Exception:
        pass
    return versions


def _device_field(device: object, name: str) -> Any:
    value = getattr(device, name, None)
    if callable(value):
        try:
            value = value()
        except Exception as exc:
            return f"<unavailable: {exc}>"
    return _normalize_json(value)


def _jax_runtime() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {"available": False, "backend": None, "devices": [], "error": str(exc)}

    report: dict[str, Any] = {"available": True, "backend": None, "devices": []}
    errors: list[str] = []
    try:
        report["backend"] = _normalize_json(jax.default_backend())
    except Exception as exc:
        errors.append(f"default_backend: {exc}")
    try:
        report["devices"] = [
            {
                "id": _device_field(device, "id"),
                "platform": _device_field(device, "platform"),
                "device_kind": _device_field(device, "device_kind"),
                "process_index": _device_field(device, "process_index"),
                "repr": str(device),
            }
            for device in jax.devices()
        ]
    except Exception as exc:
        errors.append(f"devices: {exc}")
    if errors:
        report["error"] = "; ".join(errors)
    return report


def build_manifest(
    command_name: str,
    argv: list[str],
    cli_args: argparse.Namespace | Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    timestamp: datetime | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable reproducibility manifest for a CLI run."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _format_timestamp(timestamp),
        "command": str(command_name),
        "argv": _normalize_json(list(argv)),
        "cli_args": _normalize_json(cli_args),
        "resolved_config": _normalize_json(resolved_config),
        "versions": _versions(),
        "jax": _jax_runtime(),
    }
    # Keep this helper honest: callers should be able to pass the result straight
    # to json.dump(..., allow_nan=False).
    json.dumps(manifest, allow_nan=False)
    return manifest


def save_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
    """Write a manifest JSON sidecar, creating parent directories as needed."""
    out_path = Path(path)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_normalize_json(manifest), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
