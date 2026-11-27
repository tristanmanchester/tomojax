"""Write a registered array to a NeXus default plot."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tomojax._data import nxdata
from tomojax.errors import InvalidParameter
from tomojax.io._export_data import Blocks, Selection


def nexus_settings(selection: Selection, dataset: str | None) -> dict[str, object]:
    target = "/entry/data/data" if dataset is None else dataset
    parts = target[1:].split("/")
    if not target.startswith("/") or len(parts) < 2 or any(p in ("", ".", "..") for p in parts):
        raise InvalidParameter("path", "the HDF5 dataset needs an absolute /group/signal path")
    names = {"rotation_angle" if a.name == "view" else a.name for a in selection.axes}
    if parts[-1] in names or parts[1] == "record":
        raise InvalidParameter(
            "path", "the HDF5 signal or group collides with a coordinate or record"
        )
    return {"dataset": target, "replacement": "whole file"}


def write_nexus(path: Path, blocks: Blocks, record: dict[str, object]) -> None:
    encoding = record["export"]
    assert isinstance(encoding, dict)
    parts = encoding["dataset"][1:].split("/")
    shape = blocks.selection.shape
    with h5py.File(path, "w") as file:
        parent = file
        for i, name in enumerate(parts[:-1]):
            parent.attrs["default"] = name
            parent = parent.create_group(name)
            parent.attrs["NX_class"] = "NXentry" if i == 0 else "NXcollection"
        chunks = tuple(min(n, c) for n, c in zip(shape, (16, 256, 256), strict=True))
        signal = parent.create_dataset(
            parts[-1],
            shape,
            dtype=np.dtype(blocks.values.dtype).newbyteorder("<"),
            chunks=chunks if encoding["compress"] else None,
            compression="gzip" if encoding["compress"] else None,
            shuffle=encoding["compress"],
        )
        nxdata(
            parent,
            signal,
            blocks.selection.axes,
            unit=blocks.values.unit,
            domain=blocks.selection.source.domain,
        )
        for region, block in blocks:
            signal[region] = block
        if blocks.values.range is not None:
            parent[parts[-1] + "_scaling_factor"] = blocks.values.scale
            parent[parts[-1] + "_offset"] = blocks.values.offset / blocks.values.scale
        record_field = file.create_dataset(
            f"/{parts[0]}/record",
            data=json.dumps(record, allow_nan=False),
            dtype=h5py.string_dtype(),
        )
        file.attrs["tomojax_export_record"] = record_field.name


def nexus_record(path: Path) -> dict[str, object] | None:
    if not h5py.is_hdf5(path):
        return None
    with h5py.File(path) as file:
        stored = file.attrs.get("tomojax_export_record")
        if not isinstance(stored, str):
            return None
        field = file.get(stored)
        if not isinstance(field, h5py.Dataset) or field.shape != ():
            raise ValueError(f"incomplete NeXus export at {path}: missing record at {stored}")
        record = json.loads(field[()])
        return record if isinstance(record, dict) and record.get("kind") == "export" else None
