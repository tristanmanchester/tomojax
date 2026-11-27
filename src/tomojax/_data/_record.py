"""Portable metadata shared by saved containers and interchange exports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from importlib.metadata import version
import platform
from typing import Any, TypedDict

import msgspec
import numpy as np

from tomojax._data._io_types import NXTomoMetadata
from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.json_values import normalize_json
from tomojax.core.registration import pose_columns, recorded_length_unit
from tomojax.wire.arrays import Array, Axis, Coordinates

_provenance = ContextVar[Mapping[str, object] | None]("tomojax_file_provenance", default=None)


@contextmanager
def provenance(fields: Mapping[str, object]) -> Iterator[None]:
    """Attach the calling operation to records written in this context.

    Parameters
    ----------
    fields : mapping
        Command, normalized arguments, resolved settings or worker lineage.

    Yields
    ------
    None
        Saves and exports inside this context inherit the fields.
    """
    token = _provenance.set({**(_provenance.get() or {}), **fields})
    try:
        yield
    finally:
        _provenance.reset(token)


class ArrayRecord(TypedDict):
    shape: list[int]
    dtype: str
    axes: list[dict[str, object]]
    unit: str
    domain: str


def array_record(
    shape: Sequence[int], dtype: str, axes: Sequence[Axis], unit: str, domain: str
) -> ArrayRecord:
    """Describe the stored pixels in either a saved file or an export."""
    registered = [msgspec.to_builtins(a) for a in axes]
    for axis, description in zip(axes, registered, strict=True):
        if isinstance(axis.coordinates, Coordinates):
            description["coordinates"] = {"table": "angles" if axis.name == "view" else axis.name}
    return {
        "shape": list(shape),
        "dtype": dtype,
        "axes": registered,
        "unit": unit,
        "domain": domain,
    }


def metadata_record(meta: NXTomoMetadata) -> dict[str, Any]:
    """Build the portable metadata for a TomoJAX object.

    Parameters
    ----------
    meta : NXTomoMetadata
        Geometry, tables, physical units and processing metadata.

    Returns
    -------
    dict
        JSON-compatible record with inline tables.
    """
    geometry = meta.geometry_meta or {}
    unit = recorded_length_unit(geometry) or "scan_length"
    normalized = normalize_json(geometry)
    assert isinstance(normalized, dict)
    info = normalized.get("simulation_info")
    simulation = info.get("simulation") if isinstance(info, dict) else None
    truth = simulation.pop("true_poses", None) if isinstance(simulation, dict) else None
    tables = {}
    for name, data, units in (
        ("angles", meta.angles, "degree"),
        ("image_key", meta.image_key, "1"),
        ("angle_offset", meta.angle_offset, "degree"),
        ("poses", meta.align_params, "mixed"),
    ):
        if data is not None:
            tables[name] = {"values": np.asarray(data).tolist(), "unit": units}
    columns = [{"name": c.name, "unit": c.unit} for c in pose_columns(unit)]
    for name, poses in (("true_poses", truth), ("alignment_poses", meta.alignment_poses)):
        if poses is not None:
            tables[name] = {
                "values": np.asarray(poses).tolist(),
                "unit": "mixed",
                "columns": columns,
                "frame": "detector",
            }
    if "poses" in tables:
        tables["poses"].update(
            columns=[{**c, "unit": "radian"} for c in columns[:3]] + columns[3:],
            frame=(meta.align_gauge or {}).get(
                "pose_translation_frame", "detector" if geometry.get("cone_segments") else "object"
            ),
        )
    return {
        "length_unit": unit,
        "geometry": {
            "type": meta.geometry_type,
            "meta": normalized,
            "grid": meta.grid.to_dict() if isinstance(meta.grid, Grid) else meta.grid,
            "detector": meta.detector.to_dict()
            if isinstance(meta.detector, Detector)
            else meta.detector,
        },
        "tables": tables,
        "calibration": meta.geometry_calibration,
        "alignment": {"gauge": meta.align_gauge},
        "simulation": {"misalign_spec": meta.misalign_spec, "artifacts": meta.simulation_artifacts},
        "sample": {"name": meta.sample_name},
        "source": {"name": meta.source_name, "type": meta.source_type, "probe": meta.source_probe},
        "frame": meta.frame,
        "history": meta.history,
        "provenance": {
            "tomojax": version("tomojax"),
            "python": platform.python_version(),
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
            "written": datetime.now(UTC).isoformat(),
            **(_provenance.get() or {}),
        },
    }


def feature_record(registration: Array, name: str, recipe: dict, history: list) -> dict[str, Any]:
    """Describe a feature from its owned registration and processing history."""
    length = next(
        (
            a.unit
            for a in (*registration.axes, *registration.fixed)
            if a.frame in ("volume", "detector")
        ),
        "scan_length",
    )
    meta = NXTomoMetadata(history=history, geometry_meta={"length_unit": length})
    record = metadata_record(meta)
    record["feature"] = {"name": name, "recipe": recipe}
    return record
