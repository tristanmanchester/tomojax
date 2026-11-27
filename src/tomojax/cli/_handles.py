"""Register numerical results with their coordinate vectors and reusable members."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
import math
from typing import Any

import msgspec
import numpy as np

import tomojax as tj
from tomojax._loading.registration import RegisteredMember, registered_member, unit_of
from tomojax.cli import execution as ex, products as p
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


def owned(
    registration: Array, values: Callable[[Storage], np.ndarray], context: ex.ExecutionContext
) -> Array:
    """Copy axis vectors into this result's artifacts so it can outlive the source."""
    axes = []
    for axis in registration.axes:
        c = axis.coordinates
        if isinstance(c, Coordinates):
            c = Coordinates(stage_array(context, values(c.values)))
        axes.append(msgspec.structs.replace(axis, coordinates=c))
    return msgspec.structs.replace(registration, axes=tuple(axes))


def handle(item: Root, *, volume_storage: Storage | None = None) -> ex.NewHandle:
    """Prepare registrations on the executor; binding only assigns the owner's id."""
    source = item if isinstance(item, tj.Scan | tj.Frames) else item.scan
    unit = unit_of(item if isinstance(item, tj.Reconstruction) else source)
    pixels: dict[str, Any] = {}
    if source is not None:
        pixels["angles"] = np.asarray(source.geometry.angles)
        key = "counts" if isinstance(source, tj.Frames) else "projections"
        pixels[key] = source.counts if isinstance(source, tj.Frames) else source.projections
    volume = None if isinstance(item, tj.Scan | tj.Frames) else registered_member(item, unit)
    if volume is not None:
        pixels[volume.name] = volume.pixels
    if isinstance(item, tj.Alignment):
        pixels["poses"] = item.poses
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
        volume=volume,
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
    volume: RegisteredMember | None,
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
        detector = registered_member(source, unit, view=view)
        assert detector is not None
        arrays[detector.name] = detector.at(Member(identifier, detector.name))
        angles = pixels["angles"]
        arrays["angles"] = Array(
            Member(identifier, "angles"),
            angles.shape,
            angles.dtype.str,
            (Axis("view", "acquisition", "index", Uniform(0, 1)),),
            "degree",
            "angle",
        )
    if volume is not None:
        arrays[volume.name] = volume.at(volume_storage or Member(identifier, volume.name))
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
