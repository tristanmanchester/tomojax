from __future__ import annotations

import numpy as np

from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata

# The file's key for the view angles (``NXTomoMetadata.angles``); earlier
# versions wrote it, so it stays.
_ANGLES_KEY = "thetas_deg"


def save_npz(
    path: str,
    projections: np.ndarray,
    *,
    metadata: NXTomoMetadata,
) -> None:
    """Write a typed TomoJAX payload to compressed NPZ.

    ``metadata`` mirrors the required ``save_nxtomo`` persistence contract.
    """
    payload: LoadedDataset = LoadedNXTomo(
        projections=np.asarray(projections),
        metadata=metadata,
    ).to_dataset_dict()
    if "angles" in payload:
        payload[_ANGLES_KEY] = payload.pop("angles")
    np.savez_compressed(path, **payload)


def _load_npz_dataset(path: str) -> LoadedDataset:
    with np.load(path, allow_pickle=True) as z:
        out: LoadedDataset = {}
        for k in z.files:
            val = z[k]
            key = "angles" if k == _ANGLES_KEY else k
            if isinstance(val, np.ndarray) and val.shape == () and val.dtype == object:
                out[key] = val.item()
            else:
                out[key] = val
        return out


def load_npz(path: str) -> LoadedNXTomo:
    """Load a compressed NPZ payload using the same typed shape as NXtomo."""
    return LoadedNXTomo.from_dataset(_load_npz_dataset(path))
