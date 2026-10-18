"""Geometrically and flat-field corrected SkyScan acquisition logs and TIFFs."""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np

from tomojax.geometry import ConeBeam, Grid

from ._acquisition import RawAcquisition, metadata_candidates
from ._datasets import ProjectionDataset
from ._tiff import TiffFrames, tiff_files


def _parameters(path: Path) -> configparser.ConfigParser | None:
    with path.open("rb") as source:
        header = source.read(4096).decode("utf-8-sig", errors="replace")
    if not header.lstrip().startswith("[System]") or "Scanner=" not in header:
        return None
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(path.read_text(encoding="utf-8-sig", errors="replace"))
    if not parser["System"].get("Scanner", "").lower().startswith("skyscan"):
        return None
    return parser


def _acquisition_log(path: Path) -> tuple[Path, configparser.ConfigParser] | None:
    candidates = sorted(metadata_candidates(path))
    found = [(p, params) for p in candidates if (params := _parameters(p)) is not None]
    if not found:
        return None
    if len(found) > 1:
        # A drift-reference log may accompany the acquisition, sharing its
        # prefix but describing IIF reference images instead of the TIFF stack.
        matched = [(p, params) for p, params in found if _projection_files(p, params)[1]]
        if len(matched) == 1:
            return matched[0]
    if len(found) != 1:
        raise ValueError(f"{path}: several SkyScan logs; name the acquisition log explicitly")
    return found[0]


def _projection_files(file: Path, params: configparser.ConfigParser) -> tuple[list[Path], bool]:
    acquisition = params["Acquisition"]
    prefix = acquisition["Filename Prefix"].strip()
    files = [
        p
        for p in tiff_files(file.parent)
        if p.stem.startswith(prefix) and p.stem[len(prefix) :].isdecimal()
    ]
    files.sort(key=lambda p: int(p.stem[len(prefix) :]))
    # SkyScan counts the separate arc/scout image, not an additional angular view.
    arc = file.parent / f"{prefix}_arc.tif"
    expected = int(acquisition["Number of Files"]) - int(arc.is_file())
    return files, len(files) == expected


def read_skyscan(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    """Recognize a SkyScan log by content and retain its ordered TIFF counts lazily.

    Supported exports have acquisition-time geometric and flat-field corrections
    enabled. Counts are interpreted in corrected pixel coordinates; camera
    aspect/rotation and random-movement settings are retained as provenance,
    not reapplied. Native-resolution grains controls gave no consistent roll
    preference, and detector aspect was confounded with the object z scale;
    this convention is not a vendor calibration certificate. Detector v points
    up, opposite TIFF rows. Lengths are mm; the log's image pixel size is the
    reconstruction voxel size at the object, not the detector pitch.
    """
    found = _acquisition_log(path)
    if found is None:
        return None
    if data_path is not None:
        raise ValueError("a SkyScan acquisition has no HDF5 data_path")
    file, params = found
    acquisition, system = params["Acquisition"], params["System"]
    for correction in ("Geometrical Correction", "Flat Field Correction"):
        if acquisition[correction].strip().upper() != "ON":
            raise ValueError(f"SkyScan import requires acquisition-time {correction}=ON")
    prefix = acquisition["Filename Prefix"].strip()
    files, complete = _projection_files(file, params)
    if not complete:
        raise ValueError(f"{file}: numbered TIFF count disagrees with acquisition metadata")
    counts = TiffFrames.from_files(files, flip_rows=True)
    nv, nu = counts.shape[1:]
    if (nv, nu) != (int(acquisition["Number of Rows"]), int(acquisition["Number of Columns"])):
        raise ValueError(f"{file}: projection dimensions disagree with acquisition metadata")
    binning = acquisition.get("Camera binning")
    resize = acquisition.getfloat("Resized in image", fallback=None)
    camera_pitch = system.getfloat("Camera Pixel Size (um)", fallback=None)
    pitches = [None, None]
    if binning is not None and resize is not None and camera_pitch is not None:
        pitches = [camera_pitch * 1e-3 * float(v) * resize for v in binning.split("x")]
    # Optical Axis supplies only vertical placement, not horizontal axis calibration.
    # Preserve the partial record; the caller must provide a complete detector.
    axis = acquisition.getfloat("Optical Axis (line)", fallback=None)
    sod = acquisition.getfloat("Object to Source (mm)", fallback=None)
    sdd = acquisition.getfloat("Camera to Source (mm)", fallback=None)
    beam = ConeBeam(sod, sdd) if sod is not None and sdd is not None else None
    image_pixel = acquisition.getfloat("Image Pixel Size (um)", fallback=None)
    voxel = None if image_pixel is None else image_pixel * 1e-3
    ratio = None if binning is None else [float(v) for v in binning.split("x")]
    grid = (
        None
        if ratio is None or voxel is None
        else Grid(nu, nu, nv, voxel, voxel, voxel * ratio[1] / ratio[0])
    )
    indices = np.asarray([int(p.stem[len(prefix) :]) for p in files], np.float64)
    direction = acquisition.get("Rotation Direction", "").strip().upper()
    if direction and direction not in {"CC", "CW"}:
        raise ValueError(f"unknown SkyScan rotation direction {direction!r}")
    angles = (
        indices * float(acquisition["Rotation Step (deg)"])
        if direction and "Rotation Step (deg)" in acquisition
        else np.empty(0, np.float64)
    )
    if direction == "CW":
        angles = -angles
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), counts.shape),
        angles=angles.astype(np.float32),
        detector=None,
        grid=grid,
        geometry_type="cone",
        geometry_metadata={
            "cone_beam": None if beam is None else beam.to_dict(),
            "ingest_source": "skyscan",
            "detector_center_status": "not_recorded",
            "acquisition": {
                "detector_pixel_size": pitches,
                "camera_pixel_size_um": camera_pitch,
                "camera_binning": binning,
                "resize": resize,
                "optical_axis_line": axis,
            },
            "skyscan": {section: dict(params[section]) for section in params.sections()},
        },
        sample_name=prefix,
        source_path=str(file),
        source_format="skyscan",
    )
    # Bit depth describes storage, not the calibrated incident-beam level.
    return RawAcquisition(record, counts)
