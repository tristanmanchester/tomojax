"""Optional Pallas projector capability lookup owned by the core projector layer."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class PallasModuleCapability:
    """Result of resolving the optional experimental Pallas projector module."""

    module: Any | None
    unavailable_reason: str | None


def resolve_pallas_module() -> PallasModuleCapability:
    """Resolve the optional Pallas projector module without import-time failure."""
    try:
        module = importlib.import_module("tomojax.core.pallas.api")
    except Exception as exc:
        return PallasModuleCapability(None, f"pallas_module_unavailable: {exc}")
    return PallasModuleCapability(module, None)


def resolve_pallas_callable(
    name: str,
    *,
    missing_reason: str | None = None,
) -> tuple[Callable[..., Any] | None, str | None]:
    """Resolve a callable from the optional Pallas projector module."""
    capability = resolve_pallas_module()
    if capability.module is None:
        return None, capability.unavailable_reason
    fn = getattr(capability.module, name, None)
    if fn is None:
        return None, missing_reason or f"{name}_missing"
    if not callable(fn):
        return None, missing_reason or f"{name}_not_callable"
    return fn, None
