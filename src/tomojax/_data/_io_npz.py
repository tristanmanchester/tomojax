from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tomojax import progress
from tomojax.core.files import replaced_when_complete

from ._io_helpers import sizes_of
from ._io_types import LoadedDataset, LoadedNXTomo, NXTomoMetadata

# The file's key for the view angles (``NXTomoMetadata.angles``); earlier
# versions wrote it, so it stays.
_ANGLES_KEY = "thetas_deg"
_AXES_KEY = "volume_axes_order"


def save_npz(
    path: str,
    projections: np.ndarray | None,
    *,
    metadata: NXTomoMetadata,
) -> None:
    """Write a typed TomoJAX payload to compressed NPZ.

    ``metadata`` mirrors the required ``save_nxtomo`` persistence contract; None
    projections save a reconstruction without its scan.
    """
    payload: LoadedDataset = LoadedNXTomo(
        projections=None if projections is None else np.asarray(projections),
        metadata=metadata,
    ).to_dataset_dict()
    if "angles" in payload:
        payload[_ANGLES_KEY] = payload.pop("angles")
    if "volume" in payload:
        payload[_AXES_KEY] = "zyx"
    parts = sizes_of(projections, metadata.volume)
    with replaced_when_complete(path, parts) as temporary, temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _load_npz_dataset(path: str) -> LoadedDataset:
    with np.load(path, allow_pickle=True) as z:
        out: LoadedDataset = {}
        for k in z.files:
            if k == "projections" and progress.watching():
                # NPZ stores one compressed NPY member per array. Its header
                # supplies the view count before NumPy inflates the pixel data.
                with ZipFile(path) as archive, archive.open("projections.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    header = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, _, _ = header(stream)
                views = shape[0] if shape else 0
                with progress.task("load", Path(path).name, views, "views"):
                    val = z[k]
                    active = progress.for_unit("views")
                    if active is not None:
                        active.advance(views, {})
                    progress.check_cancelled()
            else:
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
