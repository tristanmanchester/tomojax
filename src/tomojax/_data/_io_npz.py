from __future__ import annotations

import numpy as np

from ._io_nxtomo import load_nxtomo, save_nxtomo
from ._io_types import DatasetValue, LoadedDataset, LoadedNXTomo, NXTomoMetadata


def save_npz(
    path: str,
    projections: np.ndarray,
    *,
    metadata: NXTomoMetadata | None = None,
    **meta: DatasetValue,
) -> None:
    """Write a typed TomoJAX payload to compressed NPZ.

    ``metadata`` is the preferred contract and mirrors ``save_nxtomo``. Extra
    keyword metadata is retained as a compatibility path for older callers and
    overrides fields derived from ``metadata`` when both are supplied.
    """
    payload: LoadedDataset = {}
    if metadata is not None:
        payload.update(
            LoadedNXTomo(
                projections=np.asarray(projections),
                metadata=metadata,
            ).to_dataset_dict()
        )
    payload.update(meta)
    payload["projections"] = np.asarray(projections)
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


def convert(in_path: str, out_path: str) -> None:
    """Convert between .npz and .nxs based on file extension."""
    if in_path.endswith(".npz") and out_path.endswith((".nxs", ".h5", ".hdf5")):
        data = load_npz(in_path)
        save_nxtomo(
            out_path,
            data.projections,
            metadata=data.copy_metadata(),
        )
    elif in_path.endswith((".nxs", ".h5", ".hdf5")) and out_path.endswith(".npz"):
        data = load_nxtomo(in_path)
        save_npz(out_path, **data.to_dataset_dict())
    else:
        raise ValueError("Unsupported conversion. Use .npz <-> .nxs/.h5/.hdf5")
