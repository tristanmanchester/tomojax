"""Read Nikon (X-Tek) lab CT scans: an ``.xtekct`` parameter file and TIFF projections."""
# pyright: reportUnknownMemberType=false, reportAny=false

from __future__ import annotations

import configparser
from dataclasses import replace
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.core.geometry.base import Detector
from tomojax.core.geometry.cone import ConeBeam

from ._acquisition import metadata_candidates
from ._datasets import ProjectionDataset
from ._tiff import tiff_files

if TYPE_CHECKING:
    from os import PathLike

    from ._acquisition import RawAcquisition

LOG = logging.getLogger(__name__)


def _parameters(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # pyright: ignore[reportAttributeAccessIssue]
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    parser.read_string(text if text.lstrip().startswith("[") else "[XTekCT]\n" + text)
    section = next((s for s in parser.sections() if s.lower() == "xtekct"), None)
    if section is None:
        raise ValueError(f"{path} has no [XTekCT] section")
    params = dict(parser[section])
    for name in parser.sections():
        if name.lower() == "ctpro" and "AngleFile_Use" in parser[name]:
            params["AngleFile_Use"] = parser[name]["AngleFile_Use"]
    return params


def _number(params: dict[str, str], key: str, default: float | None = None) -> float:
    value = params.get(key)
    if value is None or not value.strip():
        if default is None:
            raise ValueError(f"the .xtekct file has no {key}")
        return float(default)
    return float(value)


def _angles(file: Path, params: dict[str, str], views: int) -> np.ndarray:
    """Angles from this scan's ``_ctdata.txt``, else the nominal steps."""
    name = params.get("Name", "").strip() or file.stem
    ctdata = file.parent / f"{name}_ctdata.txt"
    if ctdata.is_file() and params.get("AngleFile_Use", "true").lower() != "false":
        rows: list[tuple[int, float]] = []
        for line in ctdata.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except (IndexError, ValueError):
                continue
        if len(rows) == views:
            return np.asarray([angle for _, angle in sorted(rows)], np.float64)
        LOG.warning(
            "%s lists %d angles for %d projections; pass angles=...", ctdata, len(rows), views
        )
        return np.empty(0, np.float64)
    if not {"InitialAngle", "AngularStep"} <= params.keys():
        return np.empty(0, np.float64)
    start = _number(params, "InitialAngle")
    step = _number(params, "AngularStep")
    return start + step * np.arange(views)


def _projection_files(directory: Path, params: dict[str, str]) -> list[Path]:
    directory = directory / params.get("InputFolderName", "").strip()
    name = params.get("Name", "").strip()
    prefix = name + params.get("InputSeparator", "_") if name else ""
    files = [
        file
        for file in tiff_files(directory)
        if file.name.startswith(prefix) and (not name or file.stem[len(prefix) :].isdecimal())
    ]
    if not files:
        raise ValueError(f"no projection TIFFs named {prefix}NNNN.tif in {directory}")
    declared = int(_number(params, "Projections", len(files)))
    closing = len(files) == declared + 1 and np.isclose(
        abs(_number(params, "AngularStep", 0) * declared), 360.0
    )
    if closing:
        LOG.warning(
            "%s: excluding closing-turn frame %s after %d declared views",
            directory,
            files[-1].name,
            declared,
        )
        files = files[:-1]
    if len(files) != declared:
        raise ValueError(
            f"{directory}: {len(files)} projection files for {declared} declared views"
        )
    return files


def _axis(
    params: dict[str, str], height: float, beam: ConeBeam
) -> tuple[float, list[float]] | None:
    """Nikon's calibrated rotation axis in our right/up detector frame.

    Old COR values are detector-plane millimetres at the full panel's ends;
    ObjectOffsetX is already in object-plane millimetres. ObjectRoll and
    ObjectTilt are degrees, not a rotation of the detector itself.
    """
    modern = {"ObjectOffsetX", "ObjectRoll", "ObjectTilt"} <= params.keys()
    legacy = {"CentreOfRotationBottom", "CentreOfRotationTop"} <= params.keys()
    if not modern and not legacy:
        return None
    # Legacy files define an in-plane two-point axis; modern files add tilt.
    bottom = _number(params, "CentreOfRotationBottom", 0.0)
    top = _number(params, "CentreOfRotationTop", bottom)
    offset = _number(params, "ObjectOffsetX", (bottom + top) / (2 * beam.magnification))
    roll = (
        np.deg2rad(float(params["ObjectRoll"]))
        if "ObjectRoll" in params
        else -np.arctan2(top - bottom, height)
    )
    tilt = np.deg2rad(_number(params, "ObjectTilt", 0.0))
    return offset, [
        -float(np.sin(roll)),
        float(np.sin(tilt) * np.cos(roll)),
        float(np.cos(tilt) * np.cos(roll)),
    ]


def load_nikon_xtekct(
    path: str | PathLike[str],
    *,
    absorption: bool = True,
    reverse_angles: bool = False,
    darks: np.ndarray | float | None = None,
) -> ProjectionDataset:
    """Load a Nikon scan as a cone-beam dataset.

    Reads source and detector distances, detector pixels and offsets, the
    white level from ``path`` (an ``.xtekct``
    file), the projections ``<Name>_NNNN.tif`` beside it, and the angles from
    ``<Name>_ctdata.txt`` when present (else ``InitialAngle`` and ``AngularStep``).
    Without ``Name``, the angle sidecar uses the parameter file's basename.
    Missing nominal angle settings or an incomplete measured sidecar require
    ``angles=`` through ``load_frames``; a missing WhiteLevel requires
    ``white_level=`` or measured ``flats=`` instead of the integer dtype maximum.
    Dark calibration is not stored in this format; supply ``darks`` explicitly
    (zero only for already dark-corrected counts). Projections become
    absorption ``-log((I - dark) / (WhiteLevel - dark))`` unless
    ``absorption`` is False, with image rows flipped so detector v points up.
    Lengths stay in the file's units (mm).
    Use ``tomojax.load_frames`` to inspect metadata or crop before reading pixels.

    Vendor reconstruction dimensions, pitches and ROI offsets are retained in
    source metadata, not applied: TomoJAX derives its numerical field-of-view
    grid from the acquisition unless the caller supplies a grid.
    The calibrated rotation axis comes from ObjectOffsetX/Roll/Tilt, or the
    older CentreOfRotationTop/Bottom detector-plane offsets. An extra closing
    frame on a full turn is excluded according to the declared view count.
    ``reverse_angles`` negates the angles, for a stage turning the other way;
    check the handedness of the reconstruction on a known sample.

    Parameters
    ----------
    path : str or path-like
        Nikon .xtekct parameter file beside its projection TIFFs.
    absorption : bool, optional
        Convert counts to -log(I / WhiteLevel); defaults to True.
    reverse_angles : bool, optional
        Negate the recorded angles to reverse the stage convention.
    darks : numpy.ndarray, float or None, optional
        Measured dark level or frames, required when absorption is True.

    Returns
    -------
    ProjectionDataset
        Cone-beam scan with flipped detector rows, geometry in mm and recorded angles.

    Raises
    ------
    ValueError
        If required scan parameters or projection TIFFs are missing or inconsistent.
    OSError
        If the parameter file or a projection cannot be read.
    """
    raw = _raw_nikon(Path(path))
    if absorption:
        if raw.white_level is None:
            raise ValueError("Missing WhiteLevel: use tomojax.load_frames(path, white_level=...)")
        if darks is None:
            raise ValueError("Missing dark calibration: pass darks=... or use load_frames")
    projections = np.asarray(raw.counts, np.float32)
    if absorption:
        assert raw.white_level is not None
        assert darks is not None
        dark = np.asarray(darks, np.float32)
        if dark.ndim == 3:
            dark = dark.mean(axis=0)
        projections = -np.log(
            np.maximum((projections - dark) / np.maximum(raw.white_level - dark, 1e-6), 1e-6)
        )
    return replace(
        raw.record,
        projections=projections,
        angles=-raw.record.angles if reverse_angles else raw.record.angles,
    )


def _raw_nikon(file: Path) -> RawAcquisition:
    from ._acquisition import RawAcquisition
    from ._tiff import TiffFrames

    params = _parameters(file)
    files = _projection_files(file.parent, params)
    counts = TiffFrames.from_files(files, flip_rows=True)
    if len(counts) != len(files):
        raise ValueError("Nikon projection TIFFs must hold one image per file")
    white = _number(params, "WhiteLevel") if "WhiteLevel" in params else None
    record = _nikon_record(file, params, counts.shape, white)
    return RawAcquisition(record=record, counts=counts, white_level=white)


def _nikon_record(
    file: Path, params: dict[str, str], shape: tuple[int, int, int], white: float | None
) -> ProjectionDataset:
    """Keep metadata independent of pixel storage, including in Scan.source."""
    views, nv, nu = shape
    expected = (
        int(_number(params, "DetectorPixelsY", nv)),
        int(_number(params, "DetectorPixelsX", nu)),
    )
    if expected != (nv, nu):
        LOG.warning("projections are %s pixels; the .xtekct file lists %s", (nv, nu), expected)
    angles = _angles(file, params, views)
    detector_keys = (
        "DetectorPixelSizeX",
        "DetectorPixelSizeY",
        "DetectorOffsetX",
        "DetectorOffsetY",
    )
    detector = None
    if all(key in params for key in detector_keys):
        du, dv, u, v = (_number(params, key) for key in detector_keys)
        detector = Detector(nu, nv, du, dv, center=(u, v))
    beam = (
        ConeBeam(_number(params, "SrcToObject"), _number(params, "SrcToDetector"))
        if {"SrcToObject", "SrcToDetector"} <= params.keys()
        else None
    )
    axis = None
    if beam is not None and "DetectorPixelSizeY" in params:
        calibrated = _axis(params, nv * _number(params, "DetectorPixelSizeY"), beam)
        if calibrated is not None:
            offset, axis = calibrated
            beam = replace(beam, axis_offset=offset)
    recorded = {
        key: params[key]
        for key in (
            "ObjectOffsetX",
            "CentreOfRotationTop",
            "CentreOfRotationBottom",
            "ObjectRoll",
            "ObjectTilt",
            *detector_keys,
            "SrcToObject",
            "SrcToDetector",
        )
        if key in params
    }
    metadata: dict[str, object] = {
        "axis_status": "recorded" if axis is not None else "not_recorded",
        "grid_status": "derived",
        "ingest_source": "nikon_xtekct",
        "nikon_xtekct": {
            "path": str(file),
            "white_level": white,
            "projection_count": views,
            "volume": {
                key: float(value)
                for key, value in params.items()
                if key.startswith(("Voxels", "VoxelSize"))
                or key in ("OffsetX", "OffsetY", "OffsetZ")
            },
            **recorded,
        },
    }
    if beam is not None:
        metadata["cone_beam"] = beam.to_dict()
    if axis is not None:
        metadata["axis_unit_lab"] = axis
    return ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), shape),
        angles=angles.astype(np.float32),
        detector=detector,
        grid=None,
        geometry_type="cone",
        geometry_metadata=cast("dict[str, object]", metadata),
        sample_name=params.get("Name") or None,
        source_path=str(file),
        source_format="nikon_xtekct",
    )


def read_nikon(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    """Recognize the XTekCT section by content, irrespective of the file suffix."""

    def matches(file: Path) -> bool:
        with file.open("rb") as source:
            header = source.read(4096).decode("utf-8-sig", errors="replace")
        return header.lstrip().lower().startswith("[xtekct]")

    candidates = metadata_candidates(path)
    found = [p for p in candidates if matches(p)]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError(f"{path}: several Nikon parameter files; name one explicitly")
    if data_path is not None:
        raise ValueError("a Nikon acquisition has no HDF5 data_path")
    return _raw_nikon(found[0])


__all__ = ["load_nikon_xtekct"]
