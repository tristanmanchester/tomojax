"""Reconstruction and alignment plans as text: what will run, on what grid, in what memory.

The placement is an estimate made before the work starts; allocators, compilation
and other programs can change the eventual strategy and peak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tomojax.cli._results import unit_of

if TYPE_CHECKING:
    import tomojax as tj
    from tomojax._planning import ResolvedReconstruction
    from tomojax._smooth_alignment import CentreSelection, JointSelection
    from tomojax.geometry import Grid
    from tomojax.wire.resources import Placement

    type Work = ResolvedReconstruction | JointSelection | CentreSelection


def size(count: float) -> str:
    """A byte count in the largest binary unit that keeps it at or above one."""
    value, unit = float(count), "B"
    for larger in ("KiB", "MiB", "GiB"):
        if value < 1024:
            break
        value, unit = value / 1024, larger
    return f"{value:.0f} B" if unit == "B" else f"{value:.1f} {unit}"


def _unit(unit: str) -> str:
    return "scan units" if unit == "scan_length" else unit


def _length(value: float, unit: str) -> str:
    return f"{value:.4g} {_unit(unit)}"


def grid_text(grid: Grid, unit: str) -> str:
    """``nx x ny x nz`` voxels, their pitch and the volume's extent."""
    pitches = (grid.vx, grid.vy, grid.vz)
    pitch = " x ".join(f"{v:.4g}" for v in pitches) if len(set(pitches)) > 1 else f"{grid.vx:.4g}"
    extent = " x ".join(f"{n * v:.4g}" for n, v in zip(grid.shape[::-1], pitches, strict=True))
    return (
        f"{grid.nx} x {grid.ny} x {grid.nz} voxels at {pitch} {_unit(unit)} "
        f"({extent} {_unit(unit)})"
    )


def _pairs(count: int) -> str:
    return f"{count} near-opposite pair{'' if count == 1 else 's'}"


def _available(placement: Placement) -> str:
    known = [s.available for s in placement.snapshot if s.available is not None]
    return "unknown" if not known else size(min(known))


def placement_text(placement: Placement) -> str:
    """The estimated placement: strategy, devices, memory needed and memory free."""
    devices = ", ".join(placement.devices)
    needed = f"host {size(placement.host_bytes)}"
    if placement.device_bytes:
        needed += f", device {size(placement.device_bytes)}"
    return (
        f"estimated: {placement.strategy} on {devices}; needs {needed}; "
        f"available {_available(placement)}"
    )


def plan_lines(scan: tj.Scan, work: Work, placement: Placement) -> list[tuple[str, str]]:
    """The plan as (heading, text) rows."""
    from tomojax._planning import ResolvedReconstruction
    from tomojax._smooth_alignment import CentreSelection

    unit = unit_of(scan)
    if isinstance(work, CentreSelection):
        return [
            ("method", f"opposite views: the centre from {_pairs(len(work.samples))}"),
            ("placement", placement_text(placement)),
        ]
    if isinstance(work, ResolvedReconstruction):
        rows = [("method", f"{method_name(work)}: {'; '.join(work.reasons)}")]
        iterations = getattr(work.config, "iterations", None)
        if iterations is not None:
            rows.append(("iterations", str(iterations)))
        rows.append(("grid", grid_text(work.scan.grid, unit)))
        region = work.info.get("region")
        if region is not None:
            zeroed = ", zeroed outside the cylinder" if work.cylinder is not None else ""
            rows.append(("region", f"{region}{zeroed}"))
        rows.append(("placement", placement_text(placement)))
        return rows
    rows = [("method", "joint image and geometry fit")]
    if work.grid is not None:
        rows.append(("grid", grid_text(work.grid, unit)))
    for level in work.schedule.levels:
        budget = level.budget
        rows.append(
            (
                f"binning {level.binning}",
                f"{grid_text(level.grid, unit)}; {budget.iterations} steps, "
                f"{budget.image_iterations} image iterations",
            )
        )
    rows.append(("placement", placement_text(placement)))
    return rows


def method_name(work: ResolvedReconstruction) -> str:
    """The method as users name it: FDK for a cone-beam FBP."""
    from tomojax.recon.api import solver_name

    return solver_name(work.method, work.scan.geometry)


def plan_text(scan: tj.Scan, work: Work, placement: Placement) -> str:
    """The plan for ``--dry-run``: one heading and its text per line."""
    rows = plan_lines(scan, work, placement)
    width = max(len(heading) for heading, _ in rows)
    return "\n".join(f"{heading:<{width}}  {text}" for heading, text in rows)


def plan_header(command: str, scan: tj.Scan, work: Work, placement: Placement) -> str:
    """One line saying what a run is about to do."""
    from tomojax._planning import ResolvedReconstruction
    from tomojax._smooth_alignment import CentreSelection

    unit = unit_of(scan)
    if isinstance(work, CentreSelection):
        what = f"opposite views, {_pairs(len(work.samples))}"
    elif isinstance(work, ResolvedReconstruction):
        what = method_name(work)
        iterations = getattr(work.config, "iterations", None)
        if iterations is not None:
            what += f", {iterations} iterations"
        grid = work.scan.grid
        what += f" · {grid.nx}x{grid.ny}x{grid.nz} at {_length(grid.vx, unit)}"
    else:
        bins = ", ".join(str(level.binning) for level in work.schedule.levels)
        what = f"joint fit at binning {bins}"
    devices = ", ".join(placement.devices)
    free, host, device = _available(placement), placement.host_bytes, placement.device_bytes
    memory = f"{size(device)} of {free} on the device, {size(host)} on the host"
    if all(s.scope == "host" for s in placement.snapshot):  # one memory holds both
        memory = f"{size(host + device)} of {free}"
    return f"{command}: {what} · {placement.strategy} on {devices}, {memory}"


__all__ = [
    "grid_text",
    "method_name",
    "placement_text",
    "plan_header",
    "plan_lines",
    "plan_text",
    "size",
]
