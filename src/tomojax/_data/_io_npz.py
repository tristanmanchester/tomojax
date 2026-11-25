from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
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
_JSON_FIELDS = {
    "grid",
    "detector",
    "source",
    "geometry_meta",
    "align_gauge",
    "geometry_calibration",
    "misalign_spec",
    "simulation_artefacts",
}


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
        arrays = {
            k: json.dumps(v) if k in _JSON_FIELDS else np.asarray(v) for k, v in payload.items()
        }
        np.savez_compressed(stream, allow_pickle=False, **arrays)


def npz_layout(
    path: Path, *, volume_only: bool = False
) -> tuple[LoadedNXTomo, dict[str, np.ndarray]]:
    """The archive's record and its saved arrays, read whole."""
    with ZipFile(path) as archive:
        payload = LoadedNXTomo.from_dataset(_load_npz_dataset(archive, volume_only=volume_only))
    if TYPE_CHECKING:
        # NPZ members are read whole, so there is no lazy HDF5 projection stack.
        assert payload.projections is None or isinstance(payload.projections, np.ndarray)
    arrays = {"projections": payload.projections, "volume": payload.metadata.volume}
    return payload, {key: value for key, value in arrays.items() if value is not None}


def _load_npz_dataset(archive: ZipFile, *, volume_only: bool = False) -> LoadedDataset:
    out: LoadedDataset = {}
    known = {
        *NXTomoMetadata.__dataclass_fields__,
        _ANGLES_KEY,
        _AXES_KEY,
        "source",
        "projections",
        "disk_volume_axes_order",
    }
    wanted = {"volume", "grid", "geometry_meta", _AXES_KEY} if volume_only else known
    for member in archive.infolist():
        k = member.filename.removesuffix(".npy")
        if k not in wanted:
            continue
        with archive.open(member) as stream:
            if k == "projections" and progress.watching():
                version = np.lib.format.read_magic(stream)
                header = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, _ = header(stream)
                stream.seek(0)
                views = shape[0] if shape else 0
                with progress.task("load", Path(str(archive.filename)).name, views, "views"):
                    out[k] = np.lib.format.read_array(stream, allow_pickle=False)
                    if active := progress.for_unit("views"):
                        active.advance(views, {})
                    progress.check_cancelled()
                continue
            val = np.lib.format.read_array(stream, allow_pickle=False)
            out["angles" if k == _ANGLES_KEY else k] = (
                json.loads(val.item()) if k in _JSON_FIELDS else val
            )
    if "volume" in out and str(out.pop(_AXES_KEY, "")) != "zyx":
        raise ValueError(
            f"{archive.filename}: volume has no zyx axes tag; declare it or re-export the file"
        )
    return out


def load_npz(path: str) -> LoadedNXTomo:
    """Load a compressed NPZ payload using the same typed shape as NXtomo."""
    with ZipFile(path) as archive:
        return LoadedNXTomo.from_dataset(_load_npz_dataset(archive))
