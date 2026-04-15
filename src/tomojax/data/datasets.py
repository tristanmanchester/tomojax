from __future__ import annotations

import numpy as np

from .io_hdf5 import (
    DatasetValue,
    LoadedDataset,
    ValidationReport,
    load_nxtomo,
    load_npz,
    save_nxtomo,
    save_npz,
    validate_nxtomo,
)


def load_dataset(path: str) -> LoadedDataset:
    if path.endswith(".npz"):
        return load_npz(path)
    return load_nxtomo(path)


def save_dataset(path: str, projections: np.ndarray, **meta: DatasetValue) -> None:
    if path.endswith(".npz"):
        return save_npz(path, projections=projections, **meta)
    return save_nxtomo(path, projections, **meta)


def validate_dataset(path: str) -> ValidationReport:
    if path.endswith(".npz"):
        # NPZ: minimal validation
        data = load_npz(path)
        issues: list[str] = []
        if "projections" not in data:
            issues.append("missing 'projections' array in .npz")
        return {"issues": issues}
    return validate_nxtomo(path)

