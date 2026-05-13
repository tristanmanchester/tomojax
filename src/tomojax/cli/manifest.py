"""Reproducibility manifest helpers for CLI commands."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING, Any

from tomojax.io import normalize_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping
    import os

SCHEMA_VERSION = 1


def _normalize_json(value: Any) -> Any:
    """Convert common CLI/runtime objects into strict JSON-compatible values."""
    return normalize_json(value, namespace=True, catch_to_dict_errors=True)


def _format_timestamp(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        timestamp = datetime.now(UTC)
    if isinstance(timestamp, str):
        return timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


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
    if out_path.parent != Path():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_normalize_json(manifest), fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
