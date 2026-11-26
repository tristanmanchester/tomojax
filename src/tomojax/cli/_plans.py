"""Encode the resolver's decisions in their operation-specific protocol variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from tomojax._planning import ResolvedReconstruction
from tomojax._resources import estimate, sample
from tomojax._smooth_alignment import CenterSelection, JointSelection
from tomojax.cli import products as p
from tomojax.cli.wire_options import Grid

if TYPE_CHECKING:
    from tomojax._scan import Scan
    from tomojax.wire.progress import Result


def plan_product(
    scan: Scan,
    work: CenterSelection | JointSelection | ResolvedReconstruction,
    *,
    snapshot: tuple[p.Snapshot, ...] | None = None,
) -> p.CenterPlan | p.JointPlan | p.ReconPlan:
    """Keep the working arrays and numerical configuration out of the wire record."""
    placement = estimate(scan, work, sample(work) if snapshot is None else snapshot)
    if isinstance(work, CenterSelection):
        return p.CenterPlan(
            len(work.samples), ("Fit the center from near-opposite view pairs.",), placement
        )
    if isinstance(work, JointSelection):
        levels = tuple(
            p.Level(
                msgspec.convert(level.grid, type=Grid, from_attributes=True),
                level.budget.iterations,
                level.budget.image_iterations,
            )
            for level in work.schedule.levels
        )
        return p.JointPlan(
            ("Joint fit. Working grids extending beyond the volume include padding.",),
            levels,
            placement,
        )
    from tomojax.recon.api import solver_name

    return p.ReconPlan(
        solver_name(work.method, work.scan.geometry),
        work.reasons,
        getattr(work.config, "iterations", None),
        msgspec.convert(work.scan.grid, type=Grid, from_attributes=True),
        placement,
    )


def announce(
    command: str,
    scan: Scan,
    work: CenterSelection | JointSelection | ResolvedReconstruction,
    *,
    used: tuple[str, ...] = (),
) -> p.CenterPlan | p.JointPlan | p.ReconPlan:
    """Estimate the plan, say what the run will do, and keep a memory hint for a failure.

    ``used`` names the options the command was given (``--region``, ``--grid``), which
    the hint then does not suggest again. Returns the plan record.
    """
    from tomojax import progress
    from tomojax.cli._plan_view import plan_header, size
    from tomojax.cli._progress import memory_hint

    plan = plan_product(scan, work)
    placement = plan.placement
    progress.note("info", plan_header(command, scan, work, placement))
    available = [s.available for s in placement.snapshot if s.available is not None]
    estimate = f"the plan estimated {size(placement.host_bytes)} on the host"
    if placement.device_bytes:
        estimate += f" and {size(placement.device_bytes)} on the device"
    if available:
        estimate += f", with {size(min(available))} available"
    options = [
        option
        for option, given in (("--region cyl", "--region"), ("a smaller --grid", "--grid"))
        if given not in used
    ]
    options.append("binning the scan first (tomojax correct IN -o OUT --binning 2)")
    memory_hint.set(f"{estimate}; try {', or '.join(options)}")
    return plan


def dry_run(
    mode: str,
    source: str,
    plan: p.CenterPlan | p.JointPlan | p.ReconPlan,
    scan: Scan,
    work: CenterSelection | JointSelection | ResolvedReconstruction,
    *,
    document: dict[str, object] | None = None,
) -> Result | None:
    """Show ``--dry-run``'s plan: text, or the JSON document as the stream's result.

    ``document`` replaces the plan record as the JSON summary (an alignment adds its
    paths, fit and configuration).
    """
    from pathlib import Path

    from tomojax.cli._plan_view import plan_text
    from tomojax.cli._results import report_result

    if mode == "json":
        summary = msgspec.to_builtins(plan) if document is None else document
        return report_result(Path(source), (), summary)
    print(plan_text(scan, work, plan.placement))
    return None
