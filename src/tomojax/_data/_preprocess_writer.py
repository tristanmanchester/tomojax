from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(val) for val in value]
    return value


def _write_preprocess_provenance(
    output_path: str | Path,
    *,
    provenance: dict[str, Any],
    flat_mean: np.ndarray,
    dark_mean: np.ndarray,
) -> None:
    with h5py.File(output_path, "r+") as file:
        entry = file.require_group("entry")
        processing = entry.require_group("processing")
        processing.attrs["NX_class"] = "NXprocess"
        tomojax = processing.require_group("tomojax")
        tomojax.attrs["NX_class"] = "NXcollection"
        if "preprocess" in tomojax:
            del tomojax["preprocess"]
        group = tomojax.create_group("preprocess")
        group.attrs["NX_class"] = "NXcollection"
        for key, value in provenance.items():
            if isinstance(value, dict | list | tuple):
                group.attrs[key] = json.dumps(_json_safe(value), sort_keys=True)
            elif value is None:
                group.attrs[key] = "null"
            else:
                group.attrs[key] = value
        group.create_dataset(
            "flat_mean",
            data=np.asarray(flat_mean),
            chunks=True,
            compression="lzf",
        )
        group.create_dataset(
            "dark_mean",
            data=np.asarray(dark_mean),
            chunks=True,
            compression="lzf",
        )
