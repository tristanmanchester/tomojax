"""Numerical members of a TomoJAX result, before choosing an output or handle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tomojax._loading.loader import Frames
from tomojax._scan.model import Alignment, Reconstruction, Scan
from tomojax._scan.simulation import Simulation
from tomojax.core.registration import (
    projection_values,
    recorded_length_unit,
    scan_axes,
    volume_axes,
    volume_values,
)
from tomojax.wire.arrays import Array, Axis, Storage

if TYPE_CHECKING:
    import jax

    from tomojax.io import ProjectionDataset


@dataclass(frozen=True)
class RegisteredMember:
    name: str
    pixels: np.ndarray | jax.Array
    axes: tuple[Axis, ...]
    unit: str
    domain: str

    def at(self, storage: Storage) -> Array:
        """Bind the same registration to its location in a file or a live handle."""
        return Array(
            storage,
            self.pixels.shape,
            np.dtype(self.pixels.dtype).str,
            self.axes,
            self.unit,
            self.domain,
        )


def source_record(item: Scan | Frames | Reconstruction | None) -> ProjectionDataset | None:
    if isinstance(item, Reconstruction):
        return item.scan.source if item.scan is not None else item.source_record
    return item.source if item is not None else None


def unit_of(item: Scan | Frames | Reconstruction | None) -> str:
    """The spatial unit of the result or the acquisition it retains."""
    if isinstance(item, Reconstruction) and item.length_unit:
        return item.length_unit
    record = source_record(item)
    return (recorded_length_unit(record.geometry_metadata) if record else None) or "scan_length"


def registered_member(
    item: Scan | Frames | Reconstruction | Alignment | Simulation,
    length_unit: str,
    *,
    view: Axis | None = None,
) -> RegisteredMember | None:
    """Describe the primary pixels without materializing them or reading other members."""
    if isinstance(item, Scan | Frames):
        if view is None:
            raise ValueError("detector registration requires view coordinates")
        counts = isinstance(item, Frames)
        pixels = item.counts if counts else item.projections
        return RegisteredMember(
            "counts" if counts else "projections",
            pixels,
            scan_axes(item.geometry.detector, view, length_unit),
            *projection_values(counts=counts),
        )
    if isinstance(item, Reconstruction):
        name, pixels, grid = "volume", item.volume, item.grid
        record = source_record(item)
    elif isinstance(item, Simulation):
        name, pixels, grid = "phantom", item.phantom, item.scan.grid
        record = item.scan.source
    else:
        if item.image is None:
            return None
        name, pixels, grid = "image", item.image, item.grid or item.scan.grid
        record = item.scan.source
    if pixels.shape != grid.shape:
        raise ValueError(f"{name} shape {pixels.shape} differs from grid {grid.shape}")
    return RegisteredMember(
        name,
        pixels,
        volume_axes(grid, length_unit),
        *volume_values(
            np.dtype(pixels.dtype), length_unit, record.geometry_metadata if record else {}
        ),
    )
