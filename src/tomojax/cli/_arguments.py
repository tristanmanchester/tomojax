"""Translate admitted references and typed wire fields to public Python parameters."""

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import msgspec
import numpy as np

import tomojax as tj
from tomojax.cli import execution as ex, operations as op
from tomojax.cli._browse import pixels, source_array
from tomojax.cli._parameter_types import option_types
from tomojax.core.devices import as_devices
from tomojax.errors import InvalidParameter, within
from tomojax.wire.arrays import Hdf5, Member, Npy, Record


@cache
def _constructors() -> dict[str, type]:
    return {cls.__name__: cls for cls in option_types()}


def arguments(
    record: Record, context: ex.ExecutionContext, *, omit: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Construct option objects at their enclosing field and resolve bulk inputs."""

    def decode(name: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Member | Npy | Hdf5 | op.PathInput):
            return read_reference(name, value, context)
        if isinstance(value, Record):
            with within(name):
                return _constructors()[type(value).__name__](**arguments(value, context))
        if name == "devices":
            inventory = {f"{d.platform}:{d.id}": d for d in jax.devices()}
            if any(d not in inventory for d in value):
                raise InvalidParameter(
                    name, f"expected distinct device ids from {tuple(inventory)}"
                )
            return as_devices(tuple(inventory[d] for d in value))
        if isinstance(value, tuple):
            return tuple(decode(f"{name}[{i}]", item) for i, item in enumerate(value))
        return value

    return {
        field.name: decode(field.name, getattr(record, field.name))
        for field in msgspec.structs.fields(record)
        if field.name not in omit
    }


def read_reference(
    name: str, value: Member | Hdf5 | Npy | op.PathInput, context: ex.ExecutionContext
) -> Any:
    """Resolve stored pixels, a scan geometry, or a region estimate at execution."""
    if isinstance(value, Member):
        member = context.leases[value].value
        if name == "geometry" and isinstance(member, tj.Frames | tj.Scan):
            return member.geometry
        if name == "region" and isinstance(member, tj.RegionEstimate):
            return member
        if name not in ("region", "geometry") and isinstance(member, ex.RegisteredArray):
            return member.data
        raise InvalidParameter(name, f"{value.handle}.{value.member} has the wrong kind for {name}")
    if isinstance(value, Npy):
        if value.artifact is not None:
            pinned = context.leases[value].value
            if not isinstance(pinned, Npy):
                raise InvalidParameter(name, "expected an admitted NumPy file")
            value = pinned
        return np.load(value.path, mmap_mode="r", allow_pickle=False)
    with source_array(value, context, member=name) as (data, meta, _):
        return pixels(data, tuple(map(range, meta.shape)))


def source(
    ref: op.Source,
    context: ex.ExecutionContext,
    *,
    kinds: tuple[type, ...],
    field: str = "source",
    raw: bool | None = False,
) -> Any:
    """Resolve a pinned input or file. With raw=None keep the file's native domain."""
    if isinstance(ref, op.PathInput):
        if raw is None:
            from tomojax._loading import is_raw_input

            raw = is_raw_input(ref.path)
        return tj.load_frames(ref.path) if raw else tj.load(ref.path)
    item = context.leases[ref].value
    if not isinstance(item, kinds):
        raise InvalidParameter(
            field, f"{ref.handle}.{ref.member} must be a {' or '.join(k.__name__ for k in kinds)}"
        )
    return item
