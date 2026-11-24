from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
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
        arrays = {k: json.dumps(v) if k in _JSON_FIELDS else v for k, v in payload.items()}
        np.savez_compressed(stream, allow_pickle=False, **arrays)


def _load_npz_dataset(
    archive: ZipFile, *, mapped: dict[str, np.ndarray] | None = None
) -> LoadedDataset:
    out: LoadedDataset = dict(mapped or {})
    for member in archive.infolist():
        k = member.filename.removesuffix(".npy")
        with archive.open(member) as stream:
            if k in ("projections", "volume"):
                version = np.lib.format.read_magic(stream)
                header = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, dtype = header(stream)
                if dtype.hasobject:
                    raise ValueError(f"{archive.filename}: object arrays are unsupported")
                if mapped is not None:
                    continue
                stream.seek(0)
                if k == "projections" and progress.watching():
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


@contextmanager
def mapped_npz(path: str, *, member: str | None = None) -> Iterator[LoadedNXTomo]:
    """Map the primary array of a TomoJAX NPZ file for bounded random reads.

    Parameters
    ----------
    path : str
        Saved TomoJAX NPZ dataset.
    member : str or None, optional
        Select volume or projections. By default prefer the stored volume.

    Yields
    ------
    LoadedNXTomo
        The stored volume (z, y, x), or projections (view, v, u) for a scan,
        with metadata. The temporary pixel file lasts for the context's lifetime.

    Raises
    ------
    ValueError
        If the array or its saved volume axis declaration is invalid.
    """
    with ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        key = member or ("volume" if "volume.npy" in archive.namelist() else "projections")
        if key + ".npy" not in archive.namelist():
            raise ValueError(f"{path}: no {key} array")
        filename = Path(directory) / "pixels.npy"
        with archive.open(key + ".npy") as src, filename.open("wb") as dest:
            shutil.copyfileobj(src, dest, length=1024 * 1024)
        pixels = np.load(filename, mmap_mode="r", allow_pickle=False)
        fields = _load_npz_dataset(archive, mapped={key: pixels})
        fields.pop("volume", None)
        dataset = LoadedNXTomo.from_dataset(fields)
        dataset.metadata.volume = pixels if key == "volume" else None
        yield dataset
