"""Portable metadata shared by saved containers and interchange exports."""

from __future__ import annotations

from importlib.metadata import version
import platform

import numpy as np

from tomojax._data._io_types import NXTomoMetadata
from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.json_values import normalize_json
from tomojax.core.registration import pose_columns, recorded_length_unit


def metadata_record(meta: NXTomoMetadata) -> dict[str, object]:
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
    tables = {}
    for name, data, units in (
        ("angles", meta.angles, "degree"),
        ("image_key", meta.image_key, "1"),
        ("angle_offset", meta.angle_offset, "degree"),
        ("poses", meta.align_params, "mixed"),
    ):
        if data is not None:
            tables[name] = {"values": np.asarray(data).tolist(), "unit": units}
    if "poses" in tables:
        tables["poses"].update(
            columns=[{"name": c.name, "unit": c.unit} for c in pose_columns(unit)],
            frame=(meta.align_gauge or {}).get("pose_translation_frame", "acquisition"),
        )
    return {
        "length_unit": unit,
        "geometry": {
            "type": meta.geometry_type,
            "meta": normalize_json(geometry),
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
        "history": [],
        "provenance": {
            "tomojax": version("tomojax"),
            "python": platform.python_version(),
            "jax": version("jax"),
            "jaxlib": version("jaxlib"),
        },
    }
