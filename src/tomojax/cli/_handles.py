"""Register numerical results with their coordinate vectors and reusable members."""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import numpy as np

import tomojax as tj
from tomojax.cli import execution as ex, products as p
from tomojax.cli._results import scan_axes, unit_of
from tomojax.core.registration import pose_columns
from tomojax.recon.api import solver_name
from tomojax.wire.arrays import (
    Array,
    Axis,
    Coordinates,
    Member,
    Npy,
    Storage,
    Uniform,
)

type Root = tj.Scan | tj.Frames | tj.Simulation | tj.Alignment | tj.Reconstruction


def stage_array(context: ex.ExecutionContext, data: object) -> Npy:
    """Write only to this request's private reservation."""
    stage = context.staging.array()
    with stage.path.open("wb") as f:
        np.save(f, np.asarray(data), allow_pickle=False)
    return stage.reference


def handle(item: Root, *, volume_storage: Storage | None = None) -> ex.NewHandle:
    """Prepare registrations on the executor; binding only assigns the owner's id."""
    source = item if isinstance(item, tj.Scan | tj.Frames) else item.scan
    unit = unit_of(item if isinstance(item, tj.Reconstruction) else source)
    pixels: dict[str, Any] = {}
    if source is not None:
        pixels["angles"] = np.asarray(source.geometry.angles)
        key = "counts" if isinstance(source, tj.Frames) else "projections"
        pixels[key] = source.counts if isinstance(source, tj.Frames) else source.projections
    grid = None
    if isinstance(item, tj.Reconstruction):
        grid, pixels["volume"] = item.grid, item.volume
    elif isinstance(item, tj.Simulation):
        grid, pixels["phantom"] = item.scan.grid, item.phantom
    elif isinstance(item, tj.Alignment):
        pixels["poses"] = item.poses
        if item.image is not None:
            grid, pixels["image"] = item.grid, item.image
    # A handle owns logical references; the backing NumPy/JAX objects may be shared.
    logical = list(pixels.values())
    if isinstance(source, tj.Frames):
        logical.extend(
            v
            for v in (
                source.flats,
                source.darks,
                source.exposure_times,
                source.flat_exposure_times,
                source.flat_positions,
                source.view_positions,
            )
            if hasattr(v, "shape")
        )
    nbytes = sum(
        math.prod(v.shape) * np.dtype(v.dtype).itemsize
        for v in {id(v): v for v in logical}.values()
    )
    method = solver_name(item.method, item.geometry) if isinstance(item, tj.Reconstruction) else ""

    bind = partial(
        bind_result,
        item=item,
        source=source,
        grid=grid,
        volume_storage=volume_storage,
        unit=unit,
        pixels=pixels,
        method=method,
    )
    return ex.NewHandle(nbytes, bind, lambda: None)


def bind_result(
    identifier: str,
    *,
    item: Root,
    source: tj.Scan | tj.Frames | None,
    grid: tj.Grid | None,
    volume_storage: Storage | None,
    unit: str,
    pixels: dict[str, Any],
    method: str,
) -> ex.BoundHandle:
    members: dict[str, ex.HandleValue] = {"self": item}
    arrays: dict[str, Array] = {}
    view = Axis("view", "acquisition", "degree", Coordinates(Member(identifier, "angles")))
    scan_ref = Member(identifier, "scan")
    if source is not None:
        if isinstance(source, tj.Scan):
            members["scan"] = source
        key = "counts" if isinstance(source, tj.Frames) else "projections"
        data = pixels[key]
        arrays[key] = Array(
            Member(identifier, key),
            data.shape,
            np.dtype(data.dtype).str,
            scan_axes(source.geometry.detector, view, unit),
            "counts" if key == "counts" else "1",
            "counts" if key == "counts" else "line_integrals",
        )
        angles = pixels["angles"]
        arrays["angles"] = Array(
            Member(identifier, "angles"),
            angles.shape,
            angles.dtype.str,
            (Axis("view", "acquisition", "index", Uniform(0, 1)),),
            "degree",
            "angle",
        )
    if grid is not None:
        key = (
            "volume"
            if isinstance(item, tj.Reconstruction)
            else "image"
            if isinstance(item, tj.Alignment)
            else "phantom"
        )
        arrays[key] = ex.volume_metadata(
            grid,
            storage=volume_storage or Member(identifier, key),
            dtype=np.dtype(pixels[key].dtype).str,
            length_unit=unit,
        )
    product: p.HandleProduct
    if isinstance(item, tj.Frames):
        product = p.Frames(identifier, arrays["counts"])
    elif isinstance(item, tj.Scan):
        product = p.Scan(identifier, arrays["projections"])
    elif isinstance(item, tj.Simulation):
        product = p.Simulation(identifier, scan_ref, arrays["phantom"])
    elif isinstance(item, tj.Reconstruction):
        product = p.Reconstruction(
            identifier, arrays["volume"], scan_ref if source is not None else None, method
        )
    else:
        poses = pixels["poses"]
        arrays["poses"] = Array(
            Member(identifier, "poses"),
            poses.shape,
            poses.dtype.str,
            (view, Axis("column", "pose", "index", Uniform(0, 1))),
            "mixed",
            "poses",
            columns=pose_columns(unit),
        )
        product = p.Alignment(identifier, scan_ref, arrays["poses"], arrays.get("image"))
    members.update({k: ex.RegisteredArray(pixels[k], a) for k, a in arrays.items()})
    return ex.BoundHandle(product, members)
