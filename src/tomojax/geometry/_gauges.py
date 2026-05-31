"""Gauge canonicalisation for geometry state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tomojax.geometry._state import GeometryState


@dataclass(frozen=True)
class GaugeTransfer:
    source: str
    target: str
    value: float
    unit: str
    applied: bool
    reason: str | None = None


@dataclass(frozen=True)
class GaugeReport:
    transfers: tuple[GaugeTransfer, ...]

    @property
    def applied_transfers(self) -> tuple[GaugeTransfer, ...]:
        return tuple(transfer for transfer in self.transfers if transfer.applied)


@dataclass(frozen=True)
class CanonicalizedGeometry:
    state: GeometryState
    report: GaugeReport


def canonicalize_geometry_gauges(state: GeometryState) -> CanonicalizedGeometry:
    """Transfer mean residual pose gauges into setup parameters."""
    setup = state.setup
    pose = state.pose
    transfers: list[GaugeTransfer] = []

    mean_dx = float(np.mean(pose.dx_px))
    setup = setup.replace_parameter(
        "det_u_px",
        setup.det_u_px.with_value(setup.det_u_px.value + mean_dx),
    )
    pose = pose.with_updates(dx_px=pose.dx_px - mean_dx)
    transfers.append(GaugeTransfer("pose.dx_px.mean", "setup.det_u_px", mean_dx, "px", True))

    mean_phi = float(np.mean(pose.phi_residual_rad))
    setup = setup.replace_parameter(
        "theta_offset_rad",
        setup.theta_offset_rad.with_value(setup.theta_offset_rad.value + mean_phi),
    )
    pose = pose.with_updates(phi_residual_rad=pose.phi_residual_rad - mean_phi)
    transfers.append(
        GaugeTransfer("pose.phi_residual_rad.mean", "setup.theta_offset_rad", mean_phi, "rad", True)
    )

    mean_dz = float(np.mean(pose.dz_px))
    if setup.det_v_px.active:
        setup = setup.replace_parameter(
            "det_v_px",
            setup.det_v_px.with_value(setup.det_v_px.value + mean_dz),
        )
        pose = pose.with_updates(dz_px=pose.dz_px - mean_dz)
        transfers.append(GaugeTransfer("pose.dz_px.mean", "setup.det_v_px", mean_dz, "px", True))
    else:
        transfers.append(
            GaugeTransfer(
                "pose.dz_px.mean",
                "setup.det_v_px",
                mean_dz,
                "px",
                False,
                reason="det_v_px inactive",
            )
        )

    return CanonicalizedGeometry(
        state=GeometryState(setup=setup, pose=pose, acquisition=state.acquisition),
        report=GaugeReport(tuple(transfers)),
    )
