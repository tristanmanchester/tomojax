from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping

from ._json import JsonValue, drop_none, normalize_json


VariableStatus = Literal["estimated", "supplied", "frozen", "derived"]
CalibrationFrame = Literal[
    "detector",
    "scan",
    "object",
    "world",
    "detector_plane",
    "angle",
    "metadata",
]

_VALID_STATUSES = {"estimated", "supplied", "frozen", "derived"}
_VALID_FRAMES = {
    "detector",
    "scan",
    "object",
    "world",
    "detector_plane",
    "angle",
    "metadata",
}


@dataclass(frozen=True)
class CalibrationVariable:
    """One calibration variable with explicit frame, status, unit, and gauge metadata."""

    name: str
    value: JsonValue
    unit: str
    status: VariableStatus
    frame: CalibrationFrame
    gauge: str | None = None
    uncertainty: JsonValue | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unknown calibration variable status {self.status!r}")
        if self.frame not in _VALID_FRAMES:
            raise ValueError(f"Unknown calibration variable frame {self.frame!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return drop_none(
            {
                "name": self.name,
                "value": self.value,
                "unit": self.unit,
                "status": self.status,
                "frame": self.frame,
                "gauge": self.gauge,
                "uncertainty": self.uncertainty,
                "description": self.description,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationVariable":
        return cls(
            name=str(payload["name"]),
            value=normalize_json(payload.get("value")),
            unit=str(payload["unit"]),
            status=str(payload["status"]),  # type: ignore[arg-type]
            frame=str(payload["frame"]),  # type: ignore[arg-type]
            gauge=None if payload.get("gauge") is None else str(payload["gauge"]),
            uncertainty=normalize_json(payload.get("uncertainty"))
            if "uncertainty" in payload
            else None,
            description=None
            if payload.get("description") is None
            else str(payload["description"]),
        )


def _tuple_variables(values: Iterable[CalibrationVariable]) -> tuple[CalibrationVariable, ...]:
    return tuple(values)


@dataclass(frozen=True)
class CalibrationState:
    """Grouped calibration state for instrument, scan, and residual geometry."""

    detector: tuple[CalibrationVariable, ...] = field(default_factory=tuple)
    scan: tuple[CalibrationVariable, ...] = field(default_factory=tuple)
    object_residual: tuple[CalibrationVariable, ...] = field(default_factory=tuple)
    world_residual: tuple[CalibrationVariable, ...] = field(default_factory=tuple)
    detector_plane_residual: tuple[CalibrationVariable, ...] = field(default_factory=tuple)
    angle_residual: tuple[CalibrationVariable, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in _SECTION_NAMES:
            object.__setattr__(self, field_name, _tuple_variables(getattr(self, field_name)))
        names = [variable.name for variable in self.variables()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Calibration variable names must be unique: {duplicates}")

    def variables(self) -> tuple[CalibrationVariable, ...]:
        return (
            *self.detector,
            *self.scan,
            *self.object_residual,
            *self.world_residual,
            *self.detector_plane_residual,
            *self.angle_residual,
        )

    def variables_by_name(self) -> dict[str, CalibrationVariable]:
        return {variable.name: variable for variable in self.variables()}

    def estimated_names(self) -> set[str]:
        return {variable.name for variable in self.variables() if variable.status == "estimated"}

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            field_name: [variable.to_dict() for variable in getattr(self, field_name)]
            for field_name in _SECTION_NAMES
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationState":
        sections = {}
        for field_name in _SECTION_NAMES:
            values = payload.get(field_name, [])
            if not isinstance(values, list):
                raise TypeError(f"{field_name} must be a list of calibration variables")
            sections[field_name] = tuple(CalibrationVariable.from_dict(v) for v in values)
        return cls(**sections)


_SECTION_NAMES = (
    "detector",
    "scan",
    "object_residual",
    "world_residual",
    "detector_plane_residual",
    "angle_residual",
)
