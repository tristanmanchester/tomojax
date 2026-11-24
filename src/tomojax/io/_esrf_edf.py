"""ESRF SPEC EDF acquisitions, using acquisition .info or XML metadata."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import numpy as np

from ._acquisition import metadata_candidates
from ._datasets import ProjectionDataset
from ._edf import EdfFrames

if TYPE_CHECKING:
    from ._acquisition import RawAcquisition


def _metadata(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        head = stream.read(65536)
    if head.lstrip().startswith(b"<"):
        try:
            root = ElementTree.fromstring(head)
        except ElementTree.ParseError:
            return None
        acquisition = root.find("acquisition")
        if acquisition is None:
            return None
        fields = {node.tag.lower(): node.text.strip() for node in acquisition.iter() if node.text}
    else:
        fields = {
            key.strip().lower(): value.split("#", 1)[0].strip()
            for line in head.decode("utf-8", errors="replace").splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        }
    return fields if {"tomo_n", "dim_1", "dim_2"} <= fields.keys() else None


def _projection_files(owner: Path, fields: dict[str, str]) -> list[Path]:
    declared = fields.get("scanname", fields.get("prefix"))
    # Match the recorded prefix before parsing the image number: scan names
    # themselves can end in digits, and other scans can share this directory.
    pattern = (
        rf"({re.escape(declared.rstrip('_'))}_?)(\d+)\.edf" if declared else r"(.+?)(\d+)\.edf"
    )
    groups: dict[str, dict[int, Path]] = {}
    for file in owner.parent.iterdir():
        match = re.fullmatch(pattern, file.name, re.IGNORECASE)
        calibration = re.search(
            r"(?:^|_)(?:ref(?:HST)?|dark(?:end|HST)?|avgdarkfile)_?\d+(?:_\d+)?\.edf$",
            file.name,
            re.IGNORECASE,
        )
        if match and not calibration and "_slice_" not in file.name:
            prefix, index = match.groups()
            groups.setdefault(prefix.rstrip("_"), {})[int(index)] = file
    name = (declared or owner.stem).rstrip("_")
    if name in groups:
        frames = groups[name]
    elif len(groups) == 1:
        frames = next(iter(groups.values()))
    else:
        raise ValueError(f"{owner}: cannot identify one EDF projection series")
    # Extra closing/alignment images are not tomography projections.
    first = min(frames)
    indices = range(first, first + int(fields["tomo_n"]))
    missing = next((i for i in indices if i not in frames), None)
    if missing is not None:
        raise ValueError(f"{owner}: missing EDF projection {missing} in the declared series")
    return [frames[i] for i in indices]


def _calibrations(owner: Path, fields: dict[str, str]):
    """Keep fractional dark means; nxtomomill's integer NX output quantizes them."""
    files = sorted(owner.parent.iterdir())
    dark_files = [f for f in files if re.fullmatch(r"darkend\d+\.edf", f.name)]
    accumulated = bool(dark_files)
    if not dark_files:
        dark_files = [f for f in files if re.fullmatch(r"dark(?:HST)?\d*\.edf", f.name)]
    darks = EdfFrames.open(dark_files) if dark_files else None
    if accumulated:
        # SPEC darkend files contain sums, unlike already reduced dark.edf.
        # Without the exposure count, the summed field is not a dark level.
        darks = (
            np.asarray(darks, np.float32) * (len(dark_files) / int(fields["dark_n"]))
            if int(fields.get("dark_n", 0)) > 0
            else None
        )
    references = [
        (int(m[2]), int(m[1]), f)
        for f in files
        if (m := re.fullmatch(r"ref(\d+)_(\d+)\.edf", f.name))
    ]
    if not references:
        references = [
            (int(m[1]), 0, f) for f in files if (m := re.fullmatch(r"refHST(\d+)\.edf", f.name))
        ]
    references.sort()
    flats = EdfFrames.open([f for _, _, f in references]) if references else None
    positions = np.asarray([p for p, _, _ in references], np.float64) if references else None
    return flats, darks, positions


def _acquisition(owner: Path, fields: dict[str, str]) -> RawAcquisition:
    from ._acquisition import RawAcquisition

    counts = EdfFrames.open(_projection_files(owner, fields))
    _, rows, cols = counts.shape
    recorded = (int(fields["dim_2"]), int(fields["dim_1"]))
    if recorded != (rows, cols):
        raise ValueError(
            f"{owner}: stored EDF images are {(rows, cols)}, but scan metadata records {recorded}; "
            "cropped or resized images need updated geometry metadata"
        )
    # SPEC records effective pixels in micrometres, propagation distance in mm.
    pixel = float(fields["pixelsize"]) * 1e-3 if "pixelsize" in fields else None
    flats, darks, positions = _calibrations(owner, fields)
    acquisition: dict[str, object] = {
        "layout": "esrf_edf",
        "metadata_path": str(owner),
        "length_unit": "mm",
        "detector_pixel_size": [pixel, pixel],
        "scan_range_deg": float(fields["scanrange"]) if "scanrange" in fields else None,
    }
    if "distance" in fields:
        acquisition["sample_detector_distance"] = float(fields["distance"])
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        # A scan range does not establish its starting angle; continuous scans
        # can also have stale motor positions in individual EDF headers.
        angles=np.empty(0, np.float32),
        detector=None,
        source_path=str(owner),
        source_format="esrf_edf",
        sample_name=fields.get("scanname", owner.stem),
        geometry_metadata={
            "acquisition": acquisition,
            "detector_center_status": "not_recorded",
        },
    )
    return RawAcquisition(record, counts, flats=flats, darks=darks, flat_positions=positions)


def read_edf_scan(path: Path) -> RawAcquisition | None:
    """Recognize SPEC metadata by content; EDF image names only locate its frames."""
    if path.is_dir():
        candidates = [(f, m) for f in metadata_candidates(path) if (m := _metadata(f)) is not None]
        # SPEC .info overrides the corresponding acquisition XML, as in edf2nx.
        info_stems = {p.stem for p, _ in candidates if p.suffix == ".info"}
        candidates = [
            (p, m) for p, m in candidates if p.suffix != ".xml" or p.stem not in info_stems
        ]
        if len(candidates) > 1:
            raise ValueError(f"{path}: several EDF acquisition records; select the metadata file")
        return _acquisition(*candidates[0]) if candidates else None
    from ._pyhst import read_pyhst

    pyhst = read_pyhst(path)
    if pyhst is not None:
        return pyhst
    fields = _metadata(path)
    return _acquisition(path, fields) if fields is not None else None
