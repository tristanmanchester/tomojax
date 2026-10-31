from __future__ import annotations

import numpy as np

from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata

# The file's key for the view angles (``NXTomoMetadata.angles``); earlier
# versions wrote it, so it stays.
_ANGLES_KEY = "thetas_deg"
_AXES_KEY = "volume_axes_order"


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
    if "volume" in payload:
        payload[_AXES_KEY] = "zyx"
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
    if "volume" in out and str(out.pop(_AXES_KEY, "")) != "zyx":
        raise ValueError(f"{path}: volume has no zyx axes tag; declare it or re-export the file")
    return out


def load_npz(path: str) -> LoadedNXTomo:
    """Load a compressed NPZ payload using the same typed shape as NXtomo."""
    return LoadedNXTomo.from_dataset(_load_npz_dataset(path))
