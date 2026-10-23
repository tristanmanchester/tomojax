"""Read Nikon (X-Tek) lab CT scans: an ``.xtekct`` file and TIFF projections.

``InitialAngle`` and ``AngularStep`` set nominal angles; ``.ang`` and
``_ctdata.txt`` sidecars are not used. Nikon's top-right panel is converted to
right/up detector coordinates with its object rotation, without a reflection.
"""
# pyright: reportUnknownMemberType=false, reportAny=false

from __future__ import annotations

import configparser
from dataclasses import replace
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, cast

import numpy as np

from tomojax.core.geometry.base import Detector
from tomojax.core.geometry.cone import ConeBeam
from tomojax.core.geometry.transforms import align_u_to_v

from ._acquisition import metadata_candidates
from ._datasets import ProjectionDataset
from ._tiff import tiff_files

if TYPE_CHECKING:
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
        for key in ("AngleFile_Use", "Filter_Material", "Filter_ThicknessMM", "XraykV"):
            if name.lower() in {"ctpro", "xrays"} and key in parser[name]:
                params[key] = parser[name][key]
    return params


def _beam_hardening(file: Path, params: dict[str, str]) -> dict[str, object] | None:
    """Record CT Pro coefficients (lowest power first) and tube settings, unapplied.

    Their vendor intensity domain and scaling are not specified by this reader.
    The preset index comes from the ``.ctprofile.xml`` beside the scan, when present.
    """
    keys = [f"CoefX{n}" for n in range(5)]
    record: dict[str, object] = {"vendor": "nikon"}
    if all(params.get(key, "").strip() for key in keys):
        record["coefficients"] = [float(params[key]) for key in keys]
    for key in ("Scale", "Scaling"):
        if params.get(key, "").strip():
            record[key.lower()] = float(params[key])
    lut = params.get("BeamHardeningLUTFile", "").strip()
    if lut:
        record["lut"] = lut
    name = params.get("Name", "").strip() or file.stem
    profile = file.parent / f"{name}.ctprofile.xml"
    if profile.is_file():
        found = re.search(
            r"<BeamHardeningPreset>\s*(\d+)\s*</BeamHardeningPreset>",
            profile.read_text(encoding="utf-8-sig", errors="replace"),
        )
        if found:
            record["preset"] = int(found.group(1))
    if params.get("XraykV", "").strip():
        record["kv"] = float(params["XraykV"])
    material = params.get("Filter_Material", "").strip()
    if material and material.lower() != "none":
        record["filter"] = [material, float(params.get("Filter_ThicknessMM") or 0.0)]
    return record if len(record) > 1 else None


def _number(params: dict[str, str], key: str, default: float | None = None) -> float:
    value = params.get(key)
    if value is None or not value.strip():
        if default is None:
            raise ValueError(f"the .xtekct file has no {key}")
        return float(default)
    return float(value)


def _angles(params: dict[str, str], views: int) -> np.ndarray:
    """InitialAngle and AngularStep set the nominal schedule; sidecars are not used."""
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
    """Nikon's axis in the right/up detector frame, matching CIL's Nikon reader.

    Old COR values are detector-plane millimetres at the full panel's ends;
    ObjectOffsetX is already in object-plane millimetres. Both horizontal
    offsets have the opposite sign to our rightward detector u. ObjectRoll
    and ObjectTilt are degrees, not a rotation of the detector itself.
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
    return -offset, [
        float(np.sin(roll)),
        float(np.sin(tilt) * np.cos(roll)),
        float(np.cos(tilt) * np.cos(roll)),
    ]


def _raw_nikon(file: Path) -> RawAcquisition:
    from ._acquisition import RawAcquisition
    from ._tiff import TiffFrames

    params = _parameters(file)
    files = _projection_files(file.parent, params)
    # Nikon uses a top-right panel origin, not the top-left image convention.
    counts = TiffFrames.from_files(files, flip_rows=True, flip_cols=True)
    if len(counts) != len(files):
        raise ValueError("Nikon projection TIFFs must hold one image per file")
    white = _number(params, "WhiteLevel") if "WhiteLevel" in params else None
    record = _nikon_record(file, params, counts.shape, white)
    return RawAcquisition(record=record, counts=counts, white_level=white)


def _geometry(
    params: dict[str, str], nv: int, nu: int
) -> tuple[Detector | None, ConeBeam | None, list[float] | None, float]:
    """Convert the recorded panel and axis, retaining CIL's object-frame zero angle.

    The angle origin includes the in-plane rotation after axis alignment;
    panel offsets can change that frame even with a vertical rotation axis.
    """
    detector_keys = (
        "DetectorPixelSizeX",
        "DetectorPixelSizeY",
        "DetectorOffsetX",
        "DetectorOffsetY",
    )
    detector = None
    if all(key in params for key in detector_keys):
        du, dv, u, v = (_number(params, key) for key in detector_keys)
        detector = Detector(nu, nv, du, dv, center=(-u, v))
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
    # CIL first aligns the axis to z, then the source-to-panel centre line
    # to +y. Absorb that second rotation in our object angle, not in pixels.
    origin = 180.0
    if detector is not None and beam is not None and axis is not None:
        align = align_u_to_v(np.asarray(axis), np.array([0.0, 0.0, 1.0]))
        direction = align @ np.array(
            [detector.center[0], beam.source_to_detector, detector.center[1]]
        )
        origin -= float(np.rad2deg(np.arctan2(direction[0], direction[1])))
    return detector, beam, axis, origin


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
    angles = _angles(params, views)
    detector_keys = (
        "DetectorPixelSizeX",
        "DetectorPixelSizeY",
        "DetectorOffsetX",
        "DetectorOffsetY",
    )
    detector, beam, axis, origin = _geometry(params, nv, nu)
    # CIL NikonDataReader: angle = 180 - (InitialAngle + step * view).
    # Together with both panel-axis reversals this preserves physical chirality.
    angles = origin - angles
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
            # Keep vendor scaling/filter settings without guessing their units
            # or applying them as detector calibration.
            "parameters": params,
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
    hardening = _beam_hardening(file, params)
    if hardening is not None:
        metadata["beam_hardening"] = hardening
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
