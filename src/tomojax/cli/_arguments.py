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
from tomojax.core.devices import as_devices
from tomojax.core.files import read_hdf5
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
                shape_only=layout_only and name in ("flats", "darks", "support", "volume"),
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
            return hdf5_values(data, name, shape_only=shape_only)
    if name != "volume":
        raise InvalidParameter(name, op.ACQUISITION_ARRAY_ERROR)
    from tomojax.cli._browse import project_volume

    return project_volume(Path(value.path), context, shape_only=shape_only)


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


_SUPPORT_BYTES = 32 << 20


def hdf5_values(data: h5py.Dataset, name: str, *, shape_only: bool) -> np.ndarray:
    """Keep volume-sized support masks in seekable host storage."""
    if shape_only:
        return np.broadcast_to(np.zeros((), data.dtype), data.shape)
    if name == "support" and data.nbytes > _SUPPORT_BYTES:
        from tempfile import gettempdir

        from tomojax import progress
        from tomojax.core.host_volume import host_volume, release

        mask = host_volume(data.shape, Path(gettempdir()), dtype=data.dtype)
        plane = mask[0:1].nbytes
        batch = max(1, _SUPPORT_BYTES // plane)
        for start in range(0, len(mask), batch):
            progress.check_cancelled()
            stop = min(len(mask), start + batch)
            selection = slice(start, stop)
            data.read_direct(mask, selection, selection)
            release(mask, start * plane, stop * plane)
        return mask
    return np.asarray(data[...])
