"""Encode the resolver's decisions in their operation-specific protocol variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from tomojax._planning import ResolvedReconstruction
from tomojax._resources import estimate, sample
from tomojax._smooth_alignment import CentreSelection, JointSelection
from tomojax.cli import products as p
from tomojax.cli.wire_options import Grid

if TYPE_CHECKING:
    from tomojax._scan import Scan


def plan_product(
    scan: Scan,
    work: CentreSelection | JointSelection | ResolvedReconstruction,
    *,
    snapshot: tuple[p.Snapshot, ...] | None = None,
) -> p.CentrePlan | p.JointPlan | p.ReconPlan:
    """Keep the working arrays and numerical configuration out of the wire record."""
    placement = estimate(scan, work, sample(work) if snapshot is None else snapshot)
    if isinstance(work, CentreSelection):
        return p.CentrePlan(
            len(work.samples), ("Fit the centre from near-opposite view pairs.",), placement
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
        return p.JointPlan(("Joint image and selected geometry fit.",), levels, placement)
    from tomojax.geometry.api import is_cone_beam

    return p.ReconPlan(
        "fdk" if work.method == "fbp" and is_cone_beam(work.scan.geometry) else work.method,
        work.reasons,
        getattr(work.config, "iterations", None),
        msgspec.convert(work.scan.grid, type=Grid, from_attributes=True),
        placement,
    )
