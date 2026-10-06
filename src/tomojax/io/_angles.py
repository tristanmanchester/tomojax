"""Read acquisition angles consistently for TIFF import and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_angles(path: str | Path) -> np.ndarray:
    """Load a nonempty, finite FP32 angle vector in degrees without reordering.

    Accept a one-dimensional ``.npy`` array or text/CSV with the angle in the
    first column. Blank lines, ``#`` comments, and leading header lines are
    skipped. Once numeric data starts, a malformed row raises ``ValueError``
    with its line number. The caller must check the number of projection views.
    """
    sidecar = Path(path)
    if sidecar.suffix.lower() == ".npy":
        values = np.asarray(np.load(sidecar, allow_pickle=False), dtype=np.float32)
    else:
        rows: list[float] = []
        for number, line in enumerate(sidecar.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split(",", 1)[0].strip()
            try:
                rows.append(float(token))
            except ValueError as exc:
                if not rows:
                    continue
                raise ValueError(f"invalid angle in {sidecar} at line {number}: {token!r}") from exc
        values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("angle sidecar must be one-dimensional")
    if not values.size or not np.isfinite(values).all():
        raise ValueError("angle sidecar must contain at least one angle and only finite values")
    return values
