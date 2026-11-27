"""Stream scalar volumes into an OME-NGFF 0.5 multiscale image."""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tomojax.errors import InvalidParameter
from tomojax.io._export_data import Blocks, Selection
from tomojax.wire.arrays import Uniform

if TYPE_CHECKING:
    from zarr import Array


_LENGTH_UNITS = {
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "um": "micrometer",
    "µm": "micrometer",
    "nm": "nanometer",
}


def zarr_settings(selection: Selection, dataset: str | None) -> dict[str, object]:
    if dataset is not None:
        raise InvalidParameter("path", "only HDF5 accepts ::/group/signal")
    try:
        import_module("zarr")
    except ImportError as exc:
        raise InvalidParameter("path", "install tomojax[zarr] for OME-Zarr export") from exc
    levels = [list(selection.shape)]
    while max(levels[-1]) > 256 and selection.source.domain != "mask":
        levels.append([max(1, n // 2) for n in levels[-1]])
    return {"levels": levels, "replacement": "whole directory"}


def write_zarr(path: Path, blocks: Blocks, record: dict[str, object]) -> None:
    import zarr
    from zarr.codecs import ZstdCodec

    encoding = record["export"]
    assert isinstance(encoding, dict)
    levels = encoding["levels"]
    axes = []
    scale, origin = [], []
    for axis in blocks.selection.axes:
        coordinates = axis.coordinates
        assert isinstance(coordinates, Uniform)
        describes = {"type": "space", "name": axis.name}
        if axis.unit in _LENGTH_UNITS:
            describes["unit"] = _LENGTH_UNITS[axis.unit]
        axes.append(describes)
        scale.append(coordinates.spacing)
        origin.append(coordinates.origin)

    group = zarr.open_group(path, mode="w", zarr_format=3)
    datasets = []
    arrays = []
    for i, shape in enumerate(levels):
        if i:
            factors = [2 if n > 1 else 1 for n in levels[i - 1]]
            origin = [o + s * (f - 1) / 2 for o, s, f in zip(origin, scale, factors, strict=True)]
            scale = [s * f for s, f in zip(scale, factors, strict=True)]
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": [
                    {"type": "scale", "scale": scale},
                    {"type": "translation", "translation": origin},
                ],
            }
        )
        arrays.append(
            group.create_array(
                str(i),
                shape=tuple(shape),
                chunks=tuple(min(c, n) for c, n in zip((32, 256, 256), shape, strict=True)),
                dtype=blocks.values.dtype if i == 0 else "float32",
                dimension_names=[a["name"] for a in axes],
                compressors=[ZstdCodec()] if encoding["compress"] else None,
            )
        )

    write_levels(arrays, blocks)
    group.update_attributes(
        {
            "tomojax": record,
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "name": blocks.selection.source.name,
                        "axes": axes,
                        "datasets": datasets,
                        "type": "mean",
                    }
                ],
            },
        }
    )


def write_levels(arrays: list[Array], blocks: Blocks) -> None:
    """Fill each registered level in one forward pass of the source."""
    # Consume the same storage-order slabs as the other writers. An odd plane
    # at a slab boundary pairs with the next slab, independently at each level.
    stream_axis = blocks.selection.order.index(0)
    written = [0] * len(arrays)
    carry: dict[int, np.ndarray] = {}
    for _, source_block in blocks:
        block = source_block
        for i, target in enumerate(arrays):
            n = block.shape[stream_axis]
            region = [slice(None)] * 3
            region[stream_axis] = slice(written[i], written[i] + n)
            target[tuple(region)] = block
            written[i] += n
            if i == len(arrays) - 1:
                break
            if i in carry:
                block = np.concatenate((carry.pop(i), block), axis=stream_axis)
            if target.shape[stream_axis] > 1 and block.shape[stream_axis] % 2:
                carry[i] = np.take(block, [-1], axis=stream_axis)
            factors = tuple(2 if n > 1 else 1 for n in target.shape)
            pairs = tuple(n // f for n, f in zip(block.shape, factors, strict=True))
            if not all(pairs):
                break
            bins = tuple(x for p in zip(pairs, factors, strict=True) for x in p)
            section = tuple(slice(0, n * f) for n, f in zip(pairs, factors, strict=True))
            block = block[section].reshape(bins).mean(axis=(1, 3, 5), dtype=np.float32)


def zarr_record(path: Path) -> dict[str, object] | None:
    metadata = path / "zarr.json"
    if not metadata.is_file():
        return None
    record = json.loads(metadata.read_bytes()).get("attributes", {}).get("tomojax")
    return record if isinstance(record, dict) and record.get("kind") == "export" else None
