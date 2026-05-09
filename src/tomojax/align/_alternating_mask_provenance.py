"""Mask provenance records for alternating alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    import jax
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MaskProvenanceEntry:
    """One mask consumer in an alternating run."""

    caller: str
    stage: str
    level_factor: int
    operation: str
    mask_role: str
    mask_shape: tuple[int, ...]
    valid_fraction: float
    mask_hash: str
    includes_otsu: bool
    includes_train_gating: bool
    normalizer: str
    residual_filters: tuple[str, ...]


def record_mask_provenance(
    entries: list[MaskProvenanceEntry],
    *,
    caller: str,
    stage: str,
    level_factor: int,
    operation: str,
    mask_role: str,
    mask: jax.Array | None,
    includes_otsu: bool,
    includes_train_gating: bool,
    normalizer: str,
    residual_filters: Iterable[str],
) -> None:
    """Append a deterministic mask provenance record."""
    mask_array = _mask_array(mask)
    mask_shape = cast("tuple[int, ...]", mask_array.shape)
    entries.append(
        MaskProvenanceEntry(
            caller=caller,
            stage=stage,
            level_factor=int(level_factor),
            operation=operation,
            mask_role=mask_role,
            mask_shape=mask_shape,
            valid_fraction=_valid_fraction(mask_array),
            mask_hash=_mask_hash(mask_array),
            includes_otsu=bool(includes_otsu),
            includes_train_gating=bool(includes_train_gating),
            normalizer=normalizer,
            residual_filters=tuple(str(kind) for kind in residual_filters),
        )
    )


def mask_provenance_payload(entries: Iterable[MaskProvenanceEntry]) -> dict[str, object]:
    """Return the JSON payload for mask provenance artifacts."""
    return {
        "schema": "tomojax.mask_provenance.v1",
        "entries": [
            {
                "caller": entry.caller,
                "stage": entry.stage,
                "level_factor": entry.level_factor,
                "operation": entry.operation,
                "mask_role": entry.mask_role,
                "mask_shape": list(entry.mask_shape),
                "valid_fraction": entry.valid_fraction,
                "mask_hash": entry.mask_hash,
                "includes_otsu": entry.includes_otsu,
                "includes_train_gating": entry.includes_train_gating,
                "normalizer": entry.normalizer,
                "residual_filters": list(entry.residual_filters),
            }
            for entry in entries
        ],
    }


def _mask_array(mask: jax.Array | None) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones((0,), dtype=np.bool_)
    return cast("NDArray[np.bool_]", np.asarray(mask, dtype=np.float32) != 0.0)


def _valid_fraction(mask: NDArray[np.bool_]) -> float:
    if mask.size == 0:
        return 1.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _mask_hash(mask: NDArray[np.bool_]) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(mask, dtype=np.bool_).tobytes()).hexdigest()
    return digest[:16]
