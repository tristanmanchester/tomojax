"""Runtime details recorded alongside the arrays written by commands."""

from __future__ import annotations

from contextlib import contextmanager
import platform
import sys
from typing import TYPE_CHECKING, cast

from tomojax._version import __version__
from tomojax.io.api import JsonValue, normalize_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator, Mapping, Sequence


def _normalize_json(value: object) -> JsonValue:
    """Convert common CLI/runtime objects into strict JSON-compatible values."""
    return normalize_json(value, namespace=True, catch_to_dict_errors=True)


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


def runtime_provenance() -> dict[str, JsonValue]:
    """Report the installed versions and devices for a persisted operation."""
    return {"versions": _versions(), "devices": _jax_runtime()}


@contextmanager
def recording(
    command: str, arguments: argparse.Namespace, resolved: Mapping[str, object]
) -> Iterator[None]:
    """Attach the command and the actual configuration to every file written in this scope."""
    from tomojax.io.api import provenance

    with provenance(
        {
            "command": f"tomojax {command}",
            "argv": list(sys.argv),
            "arguments": _normalize_json(arguments),
            "resolved_config": _normalize_json(resolved),
            **runtime_provenance(),
        }
    ):
        yield
