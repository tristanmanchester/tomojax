from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from ._json import JsonValue, drop_none, normalize_json
from .conventions import ConventionAudit
from .objectives import ObjectiveCard
from .state import CalibrationState


CALIBRATION_MANIFEST_SCHEMA_VERSION = 1


def _timestamp_utc(timestamp: datetime | str | None) -> str:
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


def build_calibration_manifest(
    *,
    calibration_state: CalibrationState,
    objective_card: ObjectiveCard | None = None,
    convention_audit: ConventionAudit | None = None,
    calibrated_geometry: Mapping[str, object] | None = None,
    source: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
    timestamp: datetime | str | None = None,
) -> dict[str, JsonValue]:
    """Build a strict JSON-compatible manifest for geometry calibration metadata."""
    manifest = drop_none(
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "created_at": _timestamp_utc(timestamp),
            "tomojax_version": _tomojax_version(),
            "calibration_state": calibration_state.to_dict(),
            "objective_card": objective_card.to_dict() if objective_card is not None else None,
            "convention_audit": convention_audit.to_dict()
            if convention_audit is not None
            else None,
            "calibrated_geometry": normalize_json(calibrated_geometry),
            "source": normalize_json(source),
            "extra": normalize_json(extra),
        }
    )
    return manifest
