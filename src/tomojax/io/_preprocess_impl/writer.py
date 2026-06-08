from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tomojax.io._json import JsonValue, normalize_json


def _write_preprocess_provenance(
    output_path: str | Path,
    *,
    provenance: dict[str, JsonValue],
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
            normalized = normalize_json(value)
            if isinstance(normalized, dict | list):
                group.attrs[key] = json.dumps(normalized, sort_keys=True)
            elif value is None:
                group.attrs[key] = "null"
            else:
                group.attrs[key] = normalized
        _ = group.create_dataset(
            "flat_mean",
            data=np.asarray(flat_mean),
            chunks=True,
            compression="lzf",
        )
        _ = group.create_dataset(
            "dark_mean",
            data=np.asarray(dark_mean),
            chunks=True,
            compression="lzf",
        )
