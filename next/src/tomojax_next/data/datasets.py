from __future__ import annotations

from typing import Any, Dict, Optional

from .io_hdf5 import (
    load_nxtomo,
    save_nxtomo,
    validate_nxtomo,
    load_npz,
    save_npz,
)


def load_dataset(path: str) -> Dict[str, Any]:
    if path.endswith(".npz"):
        return load_npz(path)
    return load_nxtomo(path)


def save_dataset(path: str, projections, **meta) -> None:
    if path.endswith(".npz"):
        return save_npz(path, projections=projections, **meta)
    return save_nxtomo(path, projections, **meta)


def validate_dataset(path: str) -> Dict[str, Any]:
    if path.endswith(".npz"):
        # NPZ: minimal validation
        data = load_npz(path)
        issues = []
        if "projections" not in data:
            issues.append("missing 'projections' array in .npz")
        return {"issues": issues}
    return validate_nxtomo(path)

