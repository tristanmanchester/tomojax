"""Translate admitted references and typed wire fields to public Python parameters."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import h5py
import jax
import msgspec
import numpy as np

import tomojax as tj
from tomojax.cli import execution as ex, operations as op
from tomojax.cli._parameter_types import option_types
from tomojax.cli._results import stored_axes
from tomojax.cli.products import VOLUME_DATASET
from tomojax.core.devices import as_devices
from tomojax.core.files import read_hdf5, read_path
from tomojax.errors import InvalidParameter, within
from tomojax.wire.arrays import Hdf5, Member, Npy, Record


@cache
def _constructors() -> dict[str, type]:
    return {cls.__name__: cls for cls in option_types()}


def arguments(
    record: Record,
    context: ex.ExecutionContext,
    *,
    omit: tuple[str, ...] = (),
    layout_only: bool = False,
) -> dict[str, Any]:
    """Construct option objects at their enclosing field and resolve bulk inputs."""

    def decode(name: str, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Member | Npy | Hdf5 | op.PathInput):
            return read_reference(
                name,
                value,
                context,
                shape_only=layout_only and name in ("flats", "darks", "support"),
            )
        if isinstance(value, Record):
            with within(name):
                return _constructors()[type(value).__name__](
                    **arguments(value, context, layout_only=layout_only)
                )
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


def handle_member(name: str, value: Member, context: ex.ExecutionContext) -> Any:
    """Resolve a handle member in the domain expected by the operation field."""
    member = context.leases[value].value
    if name == "geometry" and isinstance(member, tj.Frames | tj.Scan):
        return member.geometry
    if name == "region" and isinstance(member, tj.RegionEstimate):
        return member
    if name not in ("region", "geometry") and isinstance(member, ex.RegisteredArray):
        return member.data
    raise InvalidParameter(name, f"{value.handle}.{value.member} has the wrong kind for {name}")


def read_reference(
    name: str,
    value: Member | Hdf5 | Npy | op.PathInput,
    context: ex.ExecutionContext,
    *,
    shape_only: bool,
) -> Any:
    """Resolve stored pixels, a scan geometry, or a region estimate at execution."""
    if isinstance(value, Member):
        member = handle_member(name, value, context)
        return np.broadcast_to(np.zeros((), member.dtype), member.shape) if shape_only else member
    if isinstance(value, Npy):
        if value.artifact is not None:
            pinned = context.leases[value].value
            if not isinstance(pinned, Npy):
                raise InvalidParameter(name, "expected an admitted NumPy file")
            value = pinned
        return np.load(value.path, mmap_mode="r", allow_pickle=False)
    if isinstance(value, Hdf5):
        with read_hdf5(value.path) as file:
            data = file.get(value.dataset)
            if not isinstance(data, h5py.Dataset):
                raise InvalidParameter(name, f"no dataset {value.dataset} in {value.path}")
            return (
                np.broadcast_to(np.zeros((), data.dtype), data.shape) if shape_only else data[...]
            )
    if name == "projections":
        return tj.load(value.path).projections
    return saved_volume(Path(value.path), name)


def saved_volume(path: Path, name: str) -> np.ndarray:
    """Read only the requested volume, transposed into logical voxel order."""
    if h5py.is_hdf5(path):
        with read_hdf5(path) as file:
            data = file.get(VOLUME_DATASET)
            if isinstance(data, h5py.Dataset):
                axes = stored_axes(file, path)
                return np.asarray(data[...]).transpose(tuple(axes.index(a) for a in "zyx"))
    elif path.suffix.lower() == ".npz":
        with np.load(read_path(path), allow_pickle=False) as fields:
            if "volume" in fields.files:
                if str(fields.get("volume_axes_order", "")) != "zyx":
                    raise ValueError(f"{path}: volume has no zyx axes tag")
                return fields["volume"]
    raise InvalidParameter(name, f"{path} holds no saved volume")


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
