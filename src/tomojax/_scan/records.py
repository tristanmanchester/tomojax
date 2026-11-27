"""The TomoJAX dataset behind a save or a registered-array export."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
from typing import TYPE_CHECKING, cast

from tomojax._scan.model import (
    Alignment,
    Reconstruction,
    Scan,
    geometry_record,
    record_of,
    source_of,
)
from tomojax._scan.simulation import Simulation

if TYPE_CHECKING:
    from tomojax.core.json_values import JsonValue
    from tomojax.io import ProjectionDataset


def result_dataset(
    item: Scan | Reconstruction | Alignment | Simulation,
    *,
    include_scan: bool = False,
) -> ProjectionDataset:
    """Collect effective geometry, pixels and provenance for one saved object."""
    if isinstance(item, Scan):
        record = record_of(item)
        for key in ("alignment_info", "reconstruction_info", "reconstruction_method"):
            record.geometry_metadata.pop(key, None)
    elif isinstance(item, Simulation):
        record = record_of(item.scan, include_projections=True)
        record.volume = item.phantom
        record.geometry_metadata["volume_role"] = "phantom"
        record.geometry_metadata.pop("reconstruction_method", None)
        record.geometry_metadata.pop("reconstruction_info", None)
    elif isinstance(item, Reconstruction):
        if item.scan is None:
            source_record = item.source_record
            record = geometry_record(
                item.geometry,
                source=source_record,
                name=(source_record.sample_name or "sample") if source_record else "sample",
            )
            source = item.info.get("source")
        else:
            record = record_of(
                item.scan,
                grid=None if include_scan else item.grid,
                include_projections=include_scan,
            )
            if include_scan and item.grid != item.scan.grid:
                record.geometry_metadata["volume_grid"] = item.grid.to_dict()
            source = source_of(item.scan)
        if item.length_unit is not None:
            record.geometry_metadata["length_unit"] = item.length_unit
        record.volume = item.volume
        record.geometry_metadata["reconstruction_method"] = item.method
        record.geometry_metadata["reconstruction_info"] = saved_info(
            {**item.info, "source": source}
        )
    elif isinstance(item, Alignment):  # pyright: ignore[reportUnnecessaryIsInstance]
        record = record_of(item.scan, alignment_poses=item.poses)
        record.volume = item.image
        # The alignment made this volume: an earlier reconstruction's method
        # and settings no longer describe it.
        record.geometry_metadata.pop("reconstruction_method", None)
        record.geometry_metadata.pop("reconstruction_info", None)
        # The image's grid, which need not be the scan's: load_alignment/export read it back.
        info = {**item.info} if item.grid is None else {**item.info, "working_grid": item.grid}
        record.geometry_metadata["alignment_info"] = saved_info(info)
    else:
        raise TypeError(
            f"save takes a Scan, Reconstruction, Alignment or Simulation, not {type(item)}"
        )
    if not isinstance(item, Simulation):  # a scan loaded from a simulation carries its tag
        _ = record.geometry_metadata.pop("volume_role", None)
    return record


def saved_info(info: dict[str, object]) -> dict[str, JsonValue]:
    """Convert result metadata to JSON, recording dataclass settings by field."""
    return cast("dict[str, JsonValue]", json.loads(json.dumps(info, default=_json_value)))


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return str(value)
