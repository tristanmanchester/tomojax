"""Reproducibility manifest helpers for CLI commands."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
import platform
import sys
from typing import TYPE_CHECKING, cast

from tomojax._version import __version__
from tomojax.io.api import JsonValue, normalize_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator, Mapping, Sequence
    import os

SCHEMA_VERSION = 1


def _normalize_json(value: object) -> JsonValue:
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


def _versions() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {
        "tomojax": __version__,
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

        versions["jax"] = jax.__version__
    except (ImportError, RuntimeError):
        pass
    try:
        import jaxlib

        version: object = getattr(jaxlib, "__version__", None)
        versions["jaxlib"] = _normalize_json(version)
    except (ImportError, RuntimeError):
        pass
    return versions


def _device_field(device: object, name: str) -> JsonValue:
    value: object = getattr(device, name, None)
    if callable(value):
        try:
            value = value()
        except RuntimeError as exc:
            return f"<unavailable: {exc}>"
    return _normalize_json(value)


def _jax_runtime() -> dict[str, JsonValue]:
    try:
        import jax
    except (ImportError, RuntimeError) as exc:
        return {"available": False, "backend": None, "devices": [], "error": str(exc)}

    report: dict[str, JsonValue] = {"available": True, "backend": None, "devices": []}
    errors: list[str] = []
    try:
        report["backend"] = _normalize_json(jax.default_backend())
    except RuntimeError as exc:
        errors.append(f"default_backend: {exc}")
    try:
        # jaxlib device objects have no stubs, and the manifest uses only their metadata.
        list_devices = cast("Callable[[], Sequence[object]]", jax.devices)
        devices = list_devices()
        report["devices"] = [
            {
                "id": _device_field(device, "id"),
                "platform": _device_field(device, "platform"),
                "device_kind": _device_field(device, "device_kind"),
                "process_index": _device_field(device, "process_index"),
                "repr": str(device),
            }
            for device in devices
        ]
    except RuntimeError as exc:
        errors.append(f"devices: {exc}")
    if errors:
        report["error"] = "; ".join(errors)
    return report


def build_manifest(
    command_name: str,
    argv: list[str],
    cli_args: argparse.Namespace | Mapping[str, object],
    resolved_config: Mapping[str, object],
    *,
    timestamp: datetime | str | None = None,
) -> dict[str, JsonValue]:
    """Build a JSON-serializable reproducibility manifest for a CLI run."""
    manifest: dict[str, JsonValue] = {
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
    _ = json.dumps(manifest, allow_nan=False)
    return manifest


@contextmanager
def recording(
    command: str, arguments: argparse.Namespace, resolved: Mapping[str, object]
) -> Iterator[dict[str, JsonValue]]:
    """Attach the command and the actual configuration to every file written in this scope."""
    from tomojax.io.api import provenance

    manifest = build_manifest(f"tomojax {command}", list(sys.argv), arguments, resolved)
    with provenance(
        {
            "command": manifest["command"],
            "argv": manifest["argv"],
            "arguments": manifest["cli_args"],
            "resolved_config": manifest["resolved_config"],
            "devices": manifest["jax"],
        }
    ):
        yield manifest


def save_manifest(path: str | os.PathLike[str], manifest: Mapping[str, object]) -> None:
    """Write a manifest JSON sidecar, moved onto ``path`` once whole."""
    from tomojax.core.files import replaced_when_complete

    text = json.dumps(_normalize_json(manifest), indent=2, sort_keys=True, allow_nan=False)
    with replaced_when_complete(path, "manifest") as temporary:
        _ = temporary.write_text(text + "\n", encoding="utf-8")
