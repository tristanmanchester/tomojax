from __future__ import annotations

import numpy as np

from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata


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
    np.savez_compressed(path, **payload)


def _load_npz_dataset(path: str) -> LoadedDataset:
    with np.load(path, allow_pickle=True) as z:
        out: LoadedDataset = {}
        for k in z.files:
            val = z[k]
            if isinstance(val, np.ndarray) and val.shape == () and val.dtype == object:
                out[k] = val.item()
            else:
                out[k] = val
        return out


def load_npz(path: str) -> LoadedNXTomo:
    """Load a compressed NPZ payload using the same typed shape as NXtomo."""
    return LoadedNXTomo.from_dataset(_load_npz_dataset(path))
