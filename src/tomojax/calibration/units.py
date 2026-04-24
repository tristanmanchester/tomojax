from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from tomojax.core.geometry import Detector


DetectorAxis = Literal["u", "v"]


@dataclass(frozen=True)
class DetectorPixelScale:
    """Detector pixel scale used to report calibration variables across resolutions."""

    native_du: float
    native_dv: float
    bin_factor_u: float = 1.0
    bin_factor_v: float = 1.0

    def __post_init__(self) -> None:
        for name in ("native_du", "native_dv", "bin_factor_u", "bin_factor_v"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value!r}")

    @classmethod
    def from_detectors(cls, native: Detector, level: Detector) -> "DetectorPixelScale":
        return cls(
            native_du=float(native.du),
            native_dv=float(native.dv),
            bin_factor_u=float(level.du) / float(native.du),
            bin_factor_v=float(level.dv) / float(native.dv),
        )

    def native_px_to_physical(self, value_px: float, axis: DetectorAxis) -> float:
        spacing = self.native_du if axis == "u" else self.native_dv
        return float(value_px) * spacing

    def physical_to_native_px(self, value_physical: float, axis: DetectorAxis) -> float:
        spacing = self.native_du if axis == "u" else self.native_dv
        return float(value_physical) / spacing

    def native_px_to_level_px(self, value_px: float, axis: DetectorAxis) -> float:
        factor = self.bin_factor_u if axis == "u" else self.bin_factor_v
        return float(value_px) / factor

    def level_px_to_native_px(self, value_px: float, axis: DetectorAxis) -> float:
        factor = self.bin_factor_u if axis == "u" else self.bin_factor_v
        return float(value_px) * factor

    def to_dict(self) -> dict[str, float]:
        return {
            "native_du": float(self.native_du),
            "native_dv": float(self.native_dv),
            "bin_factor_u": float(self.bin_factor_u),
            "bin_factor_v": float(self.bin_factor_v),
        }


@dataclass(frozen=True)
class DetectorPixelValue:
    """A detector-plane value stored canonically in native detector pixels."""

    axis: DetectorAxis
    native_px: float

    def report(self, scale: DetectorPixelScale) -> dict[str, float | str]:
        return {
            "axis": self.axis,
            "native_px": float(self.native_px),
            "level_px": scale.native_px_to_level_px(self.native_px, self.axis),
            "physical": scale.native_px_to_physical(self.native_px, self.axis),
        }
