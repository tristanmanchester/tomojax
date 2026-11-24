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
from tomojax._loading import projection_frames
from tomojax._scan import frame_output_grid
from tomojax.cli import execution as ex, operations as op
from tomojax.cli._parameter_types import option_types
from tomojax.core.devices import as_devices
from tomojax.core.files import read_hdf5, validate_virtual_pixels
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
            return read_reference(name, value, context, shape_only=layout_only)
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


def handle_member(
    name: str, value: Member, context: ex.ExecutionContext, *, shape_only: bool
) -> Any:
    """Resolve a handle member in the domain expected by the operation field."""
    member = context.leases[value].value
    if name == "geometry" and isinstance(member, tj.Frames | tj.Scan):
        return member.geometry
    if name == "region" and isinstance(member, tj.RegionEstimate):
        return member
    if name not in ("region", "geometry") and isinstance(member, ex.RegisteredArray):
        return array_layout(member.data) if shape_only else member.data
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
        return handle_member(name, value, context, shape_only=shape_only)
    if isinstance(value, Npy):
        if value.artifact is not None:
            pinned = context.leases[value].value
            if not isinstance(pinned, Npy):
                raise InvalidParameter(name, "expected an admitted NumPy file")
            value = pinned
        data = np.load(value.path, mmap_mode="r", allow_pickle=False)
        return array_layout(data) if shape_only else data
    if isinstance(value, Hdf5):
        with read_hdf5(value.path) as file:
            data = explicit_dataset(file, value, name)
            return hdf5_values(data, name, shape_only=shape_only)
    if name != "volume":
        raise InvalidParameter(name, op.ACQUISITION_ARRAY_ERROR)
    from tomojax.cli._browse import project_volume

    return project_volume(Path(value.path), context, shape_only=shape_only)


def explicit_dataset(file: h5py.File, ref: Hdf5, field: str) -> h5py.Dataset:
    """Resolve a dataset and verify its virtual mappings before exposing its pixels."""
    data = file.get(ref.dataset)
    if not isinstance(data, h5py.Dataset):
        raise InvalidParameter(field, f"no dataset {ref.dataset} in {ref.path}")
    validate_virtual_pixels(data)
    return data


def array_layout(data: Any) -> np.ndarray:
    """Represent dimensions and dtype without accessing samples, including positive exposures."""
    return np.broadcast_to(np.ones((), data.dtype), data.shape)


def scan_layout(frames: tj.Frames, *, raw: bool) -> tj.Scan:
    """Represent the acquisition geometry without materialising line integrals."""
    return tj.Scan(
        np.broadcast_to(
            np.zeros((), np.float32 if raw else frames.counts.dtype), frames.counts.shape
        ),
        frame_output_grid(frames.geometry, frames.source) if raw else frames.geometry,
        name=frames.name,
        corrections=frames.corrections,
        source=frames.source,
        view_positions=frames.view_positions,
    )


def source(
    ref: op.Source,
    context: ex.ExecutionContext,
    *,
    kinds: tuple[type, ...],
    field: str = "source",
    raw: bool | None = False,
    layout_only: bool = False,
) -> Any:
    """Resolve a pinned input or file. With raw=None keep the file's native domain."""
    if isinstance(ref, op.PathInput):
        if layout_only:
            frames, is_raw = projection_frames(Path(ref.path), require_raw=raw is True)
            return frames if raw or (raw is None and is_raw) else scan_layout(frames, raw=is_raw)
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
        return array_layout(data)
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
