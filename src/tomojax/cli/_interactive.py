"""Register small products of the Python interactive operations for a session."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np

from tomojax import interactive
from tomojax._scan.model import detector_window
from tomojax._workflow.planning import plane_dimension
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._handles import stage_array
from tomojax.cli._results import unit_of
from tomojax.cli.wire_options import Grid
from tomojax.core.registration import registered_plane, scan_axes, volume_axes
from tomojax.wire.arrays import Array, Axis, Coordinates

if TYPE_CHECKING:
    import tomojax as tj
    from tomojax._workflow.planning import ResolvedReconstruction
    from tomojax.corrections import Step


def slice_product(
    command: op.Slice, plans: tuple[ResolvedReconstruction, ...], context: ex.ExecutionContext
) -> p.PlaneResult | p.Sweep:
    """Stage one spatial registration, optionally with signed detector coordinates."""
    plan = plans[0]
    axis, index = command.axis, command.index
    write = partial(stage_array, context)
    if isinstance(command, op.CenterSweep):
        pixels = interactive.resolved_sweep(plans, axis, index)
        leading = (
            Axis("center", "detector", "pixel", Coordinates(write(np.asarray(command.centers)))),
        )
        product = p.Sweep
    else:
        pixels = interactive.resolved_slice(plan, axis, index)
        leading = ()
        product = p.PlaneResult
    unit = unit_of(plan.scan)
    sample = registered_plane(
        pixels,
        volume_axes(plan.scan.grid, unit),
        plane_dimension(plan.scan.grid, axis, index),
        index,
        unit,
        leading_axes=leading,
    )
    return product(sample.store(write))


def correction_product(
    command: op.CorrectionPreview,
    frames: tj.Frames | tj.Scan,
    steps: tuple[Step, ...],
    settings: dict[str, Any],
    context: ex.ExecutionContext,
) -> p.CorrectionResult:
    """Publish native and binned rows in their own detector and acquisition coordinates.

    A scan's input rows are line integrals; raw frames' are counts.
    """
    band = interactive.correction_preview(frames, command.rows, *steps, **settings)
    write = partial(stage_array, context)
    angles = np.asarray(frames.geometry.angles)
    unit = unit_of(frames)
    native = Axis("view", "acquisition", "degree", Coordinates(write(angles)))
    kept = Axis("view", "acquisition", "degree", Coordinates(write(angles[band.views])))
    lower, upper = command.rows
    factor = command.binning
    detector = frames.geometry.detector
    before = detector_window(slice(lower * factor, upper * factor), slice(None), detector)(detector)
    after = detector_window(slice(lower, upper), slice(None), band.detector)(band.detector)
    return p.CorrectionResult(
        Array(
            write(band.before),
            band.before.shape,
            band.before.dtype.str,
            scan_axes(before, native, unit),
            "count" if band.domain == "counts" else "1",
            band.domain,
        ),
        Array(
            write(band.after),
            band.after.shape,
            band.after.dtype.str,
            scan_axes(after, kept, unit),
            "1",
            "line_integrals",
        ),
        Array(
            write(band.views),
            band.views.shape,
            band.views.dtype.str,
            (kept,),
            "index",
            "view_indices",
        ),
    )


def region_product(scan: tj.Scan, context: ex.ExecutionContext) -> ex.NewHandle:
    """Stage registered overlays and retain the estimator result for subsequent operations."""
    overlay = interactive.region_preview(scan)
    estimate = overlay.estimate
    grid = msgspec.convert(estimate.grid, type=Grid, from_attributes=True)
    arrays = {
        axis: plane.store(partial(stage_array, context)) for axis, plane in overlay.images.items()
    }
    bounds = overlay.bounds

    def bind(identifier: str) -> ex.BoundHandle:
        return ex.BoundHandle(p.Region(identifier, grid, bounds, arrays), {"self": estimate})

    return ex.NewHandle(scan.projections.nbytes, bind, lambda: None)
