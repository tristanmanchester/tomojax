"""Preflight and apply a worker projection or backprojection with its registered axes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import tomojax as tj
from tomojax._loading.registration import unit_of
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._arguments import arguments, source
from tomojax.cli._browse import operator_input
from tomojax.cli._handles import stage_array
from tomojax.core.array_grid import Grid
from tomojax.core.registration import scan_axes
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Axis, Coordinates, Member


def project_work(
    operation: op.Project | op.Backproject, context: ex.ExecutionContext, *, validating: bool
) -> Callable[[], p.Projections | p.Volume]:
    """Apply an operator in the source's geometry; a path source never loads its pixels."""
    forward = isinstance(operation, op.Project)
    name = "volume" if forward else "projections"
    options = arguments(operation, context, omit=("source", name), layout_only=validating)
    scan: tj.Scan = source(operation.source, context, kinds=(tj.Scan,), layout_only=True)
    grid: tj.Grid = options["grid"] or scan.grid
    unit = unit_of(scan)
    if isinstance(operation, op.Project):
        layout = Grid.volume(grid, unit)
        data = operator_input(operation.volume, layout, context, field=name, shape_only=validating)
    else:
        view = Axis("view", "acquisition", "degree", Coordinates(Member("source", "angles")))
        layout = Grid(scan.projections.shape, scan_axes(scan.detector, view, unit), "1")
        data = operator_input(
            operation.projections,
            layout,
            context,
            field=name,
            shape_only=validating,
            read_grid=lambda _: np.asarray(scan.angles),
        )
    expected = layout.shape
    if grid.nz == scan.detector.nv == 1 and data.ndim == 2:
        data = data[None, :, :] if forward else data[:, None, :]
    if tuple(data.shape) != expected:
        raise InvalidParameter(name, f"{name} shape {data.shape} does not match {expected}")

    def apply() -> p.Projections | p.Volume:
        if forward:
            pixels = np.asarray(tj.project(scan.geometry, data, **options))
            view = Axis(
                "view",
                "acquisition",
                "degree",
                Coordinates(stage_array(context, scan.angles)),
            )
            return p.Projections(
                Array(
                    stage_array(context, pixels),
                    pixels.shape,
                    pixels.dtype.str,
                    scan_axes(scan.detector, view, unit),
                    "1",
                    "line_integrals",
                )
            )
        pixels = np.asarray(tj.backproject(scan.geometry, data, **options))
        return p.Volume(
            ex.volume_metadata(
                grid,
                storage=stage_array(context, pixels),
                dtype=pixels.dtype.str,
                length_unit=unit,
            )
        )

    return apply
